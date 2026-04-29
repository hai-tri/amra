#!/usr/bin/env python3
"""
Optuna multi-objective NSGA-II sweep over (epsilon, num_layers, k_w, k_r).

Loads model + extracts undefended refusal direction + measures undefended
attack baselines + undefended utility once, then runs `n_trials` configs by
restoring weight snapshots between trials.  Each trial measures both:

    refusal_gap  = (undef_base − def_arditi) + (undef_pca8 − def_pca8)
    utility_loss = composite of MATH500 / MMLU / Pile-BPB regressions

NSGA-II returns a Pareto front over (refusal_gap, utility_loss) instead of
collapsing them into a single weighted objective.  Single-objective TPE on
refusal_gap alone is also available via --objective refusal_only.

Trial results are logged to an Optuna SQLite study + a flat CSV with all
metrics.
"""

import argparse
import csv
import datetime
import gc
import json
import math
import os
import sys
import tempfile

os.environ.setdefault("MPLCONFIGDIR", "/tmp/mplconfig")
os.environ.setdefault("FONTCONFIG_PATH", "/tmp/fontconfig")
os.makedirs(os.environ["MPLCONFIGDIR"], exist_ok=True)
os.makedirs(os.environ["FONTCONFIG_PATH"], exist_ok=True)

import optuna
import torch

# GH200 / H100 throughput: TF32 + cuDNN benchmark with negligible numeric impact
torch.backends.cuda.matmul.allow_tf32 = True
torch.backends.cudnn.allow_tf32 = True
torch.backends.cudnn.benchmark = True
torch.set_float32_matmul_precision("high")

REPO_DIR = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.insert(0, REPO_DIR)
sys.path.insert(0, os.path.join(REPO_DIR, "refusal_direction"))
sys.path.insert(0, os.path.join(REPO_DIR, "scripts", "eval"))

from pipeline.model_utils.model_factory import construct_model_base
from pipeline.submodules.generate_directions import generate_directions
from pipeline.submodules.select_direction import select_direction, get_refusal_scores

from apply_obfuscation import apply_obfuscation
from attacks.evaluate_abliteration import evaluate_abliteration_resistance
from obfuscation_config import ObfuscationConfig
from quick_attack_test import (
    CONFIGS,
    SYSTEM_PROMPT,
    _build_arditi_hooks,
    _build_pca_hooks,
    _extract_pca_directions,
    _measure_utility,
    _restore,
    _save,
)
from run_obfuscation_pipeline import filter_data, load_mlabonne_datasets

import functools
from quick_attack_test import _QWEN3_TEMPLATE


EPSILON_CHOICES = [0.005, 0.025, 0.2, 0.5]
LAYER_CHOICES = [10, 15, 20, 25, 30, 35, 40]
K_W_CHOICES = [1, 2, 4, 8, 16]
K_R_CHOICES = [1, 2, 4, 8, 16]

CSV_HEADER = [
    "trial", "model", "epsilon", "num_layers", "num_writer_directions",
    "num_reader_directions", "avg_cos_sim",
    "ref_undef_base", "ref_undef_arditi", "ref_undef_pca8",
    "ref_def_base", "ref_def_arditi", "ref_def_pca8",
    "arditi_gap", "pca8_gap", "refusal_gap",
    "bpb_undef", "mmlu_undef", "math500_undef",
    "bpb_def", "mmlu_def", "math500_def",
    "bpb_loss", "mmlu_loss", "math500_loss", "utility_loss",
    "status",
]


def _setup_tokenizer(model_base, model_id):
    is_qwen3 = "qwen3" in model_id.lower()
    is_gemma = "gemma" in model_id.lower()
    if is_qwen3:
        tok = model_base.tokenizer
        orig = tok.apply_chat_template
        def _no_think(messages, **kw):
            kw.setdefault("enable_thinking", False)
            return orig(messages, **kw)
        tok.apply_chat_template = _no_think
        def _qwen3_tok(instructions, outputs=None, system=None):
            prompts = [_QWEN3_TEMPLATE.format(instruction=i) for i in instructions]
            if outputs is not None:
                prompts = [p + o for p, o in zip(prompts, outputs)]
            return tok(prompts, padding=True, truncation=False, return_tensors="pt")
        model_base.tokenize_instructions_fn = functools.partial(
            _qwen3_tok, system=SYSTEM_PROMPT
        )
    elif not is_gemma:
        model_base.tokenize_instructions_fn = functools.partial(
            model_base.tokenize_instructions_fn, system=SYSTEM_PROMPT
        )


def _drop(better, worse):
    """Loss when ``better`` should exceed ``worse`` (e.g., undef accuracy vs def).

    Returns max(0, better − worse), or 0 if either is missing/non-finite.
    """
    if better is None or worse is None:
        return 0.0
    if not (math.isfinite(better) and math.isfinite(worse)):
        return 0.0
    return max(0.0, better - worse)


def _utility_loss(util_undef, util_def):
    """Composite utility loss: BPB increase + MMLU drop + MATH500 drop.

    Returns dict with per-metric and total losses.  Components are normalised
    to the undefended scale so each metric contributes proportionally.
    """
    bpb_undef = util_undef.get("bpb"); bpb_def = util_def.get("bpb")
    mmlu_undef = util_undef.get("mmlu"); mmlu_def = util_def.get("mmlu")
    m500_undef = util_undef.get("math500"); m500_def = util_def.get("math500")

    # BPB is cross-entropy: lower is better, so loss = def − undef.
    # MMLU and MATH500 are accuracies: higher is better, so loss = undef − def.
    bpb_loss = _drop(bpb_def, bpb_undef)
    mmlu_loss = _drop(mmlu_undef, mmlu_def)
    m500_loss = _drop(m500_undef, m500_def)

    bpb_norm = bpb_loss / max(1e-3, bpb_undef or 1.0)
    mmlu_norm = mmlu_loss / max(1e-3, mmlu_undef or 1.0)
    m500_norm = m500_loss / max(1e-3, m500_undef or 1.0)

    return {
        "bpb_loss": bpb_loss, "mmlu_loss": mmlu_loss, "math500_loss": m500_loss,
        "utility_loss": bpb_norm + mmlu_norm + m500_norm,
    }


def main():
    pa = argparse.ArgumentParser()
    pa.add_argument("--model", choices=list(CONFIGS.keys()), required=True)
    pa.add_argument("--n_trials", type=int, default=30)
    pa.add_argument("--n", type=int, default=20,
                    help="Harmful/harmless prompts for attack eval per trial")
    pa.add_argument("--num_calibration_prompts", type=int, default=64)
    pa.add_argument("--attack_batch_size", type=int, default=64)
    pa.add_argument("--forward_batch_size", type=int, default=64)
    pa.add_argument("--bpb_batches", type=int, default=32)
    pa.add_argument("--mmlu_n", type=int, default=200)
    pa.add_argument("--math500_n", type=int, default=200)
    pa.add_argument("--utility_batch_size", type=int, default=8,
                    help="lm-harness batch size for MMLU / MATH500")
    pa.add_argument("--objective", choices=["multi", "refusal_only"], default="multi",
                    help="multi = NSGA-II Pareto over (refusal_gap, utility_loss); "
                         "refusal_only = TPE on refusal_gap (skips utility eval)")
    pa.add_argument("--seed", type=int, default=42)
    pa.add_argument("--study_name", default=None)
    pa.add_argument("--storage", default=None,
                    help="Optuna storage URL (default = SQLite next to CSV)")
    pa.add_argument("--output_dir",
                    default=os.path.join(REPO_DIR, "results", "optuna_sweep"))
    args = pa.parse_args()

    measure_utility = (args.objective == "multi")

    os.makedirs(args.output_dir, exist_ok=True)
    ts = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
    csv_path = os.path.join(args.output_dir, f"optuna_{args.model}_{ts}.csv")
    db_path = os.path.join(args.output_dir, f"optuna_{args.model}_{ts}.db")
    study_name = args.study_name or f"aprs_{args.model}_{ts}"
    storage = args.storage or f"sqlite:///{db_path}"

    model_id, _, _ = CONFIGS[args.model]
    print(f"[optuna] model={model_id} trials={args.n_trials} "
          f"objective={args.objective} csv={csv_path}")
    print(f"[optuna] study={study_name} storage={storage}")

    # ── One-time model load + direction extraction + undef baselines ────────
    model_base = construct_model_base(model_id)
    _setup_tokenizer(model_base, model_id)

    harmful_train, harmless_train, harmful_val, harmless_val = load_mlabonne_datasets(
        n_train=400, n_val=100,
    )
    print("\n[optuna] filtering data ...")
    harmful_train, harmless_train, harmful_val, harmless_val = filter_data(
        model_base, harmful_train, harmless_train, harmful_val, harmless_val,
    )
    print(f"  harmful_train={len(harmful_train)}  harmless_train={len(harmless_train)}")
    print(f"  harmful_val={len(harmful_val)}    harmless_val={len(harmless_val)}")

    print("\n[optuna] extracting undefended refusal direction ...")
    with tempfile.TemporaryDirectory() as tmp:
        mean_diffs_train = generate_directions(
            model_base, harmful_train, harmless_train, artifact_dir=tmp,
        )
        pos, layer, direction = select_direction(
            model_base, harmful_val, harmless_val, mean_diffs_train, artifact_dir=tmp,
        )
        with open(os.path.join(tmp, "direction_evaluations.json")) as f:
            ablation_scores = json.load(f)
    print(f"  direction: pos={pos}, layer={layer}, ||r||={direction.norm():.4f}")

    harmful_test = harmful_val[:args.n]
    harmless_test = harmless_val[:args.n]

    def _refusal(pre, post):
        scores = get_refusal_scores(
            model_base.model,
            harmful_test,
            model_base.tokenize_instructions_fn,
            model_base.refusal_toks,
            fwd_pre_hooks=pre,
            fwd_hooks=post,
            batch_size=args.attack_batch_size,
        )
        return scores.mean().item()

    print("\n[optuna] measuring undefended attack baselines ...")
    ref_undef_base = _refusal([], [])
    undef_arditi_res = evaluate_abliteration_resistance(
        model=model_base.model,
        tokenizer=model_base.tokenizer,
        tokenize_fn=model_base.tokenize_instructions_fn,
        block_modules=model_base.model_block_modules,
        attn_modules=model_base.model_attn_modules,
        mlp_modules=model_base.model_mlp_modules,
        harmful_prompts=harmful_test,
        benign_prompts=harmless_test,
        original_direction=direction,
        refusal_toks=model_base.refusal_toks,
        batch_size=args.attack_batch_size,
    )
    ref_undef_arditi = undef_arditi_res["arditi_refusal_score"]
    undef_pca_dirs, _ = _extract_pca_directions(
        model_base.model,
        model_base.tokenizer,
        model_base.tokenize_instructions_fn,
        model_base.model_block_modules,
        harmful_test,
        harmless_test,
        top_k=8,
        batch_size=args.attack_batch_size,
    )
    up_pre, up_post = _build_pca_hooks(
        model_base.model_block_modules,
        model_base.model_attn_modules,
        model_base.model_mlp_modules,
        undef_pca_dirs,
    )
    ref_undef_pca8 = _refusal(up_pre, up_post)
    print(f"  undef_base={ref_undef_base:.4f}  undef_arditi={ref_undef_arditi:.4f}  "
          f"undef_pca8={ref_undef_pca8:.4f}")

    util_undef = {"bpb": None, "mmlu": None, "math500": None}
    if measure_utility:
        print("\n[optuna] measuring undefended utility (BPB + MMLU + MATH500) ...")
        util_undef = _measure_utility(
            model_base, [], [], args.bpb_batches, args.mmlu_n,
            args.math500_n, args.utility_batch_size,
        )
        print(f"  undef bpb={util_undef.get('bpb'):.4f}  "
              f"mmlu={util_undef.get('mmlu'):.4f}  "
              f"math500={util_undef.get('math500'):.4f}")

    clean_snapshot = _save(model_base.model)

    # CSV header
    if not os.path.exists(csv_path):
        with open(csv_path, "w", newline="") as f:
            csv.writer(f).writerow(CSV_HEADER)

    # ── Objective ───────────────────────────────────────────────────────────
    def objective(trial: optuna.trial.Trial):
        epsilon = trial.suggest_categorical("epsilon", EPSILON_CHOICES)
        n_layers = trial.suggest_categorical("num_layers", LAYER_CHOICES)
        k_w = trial.suggest_categorical("k_w", K_W_CHOICES)
        k_r = trial.suggest_categorical("k_r", K_R_CHOICES)

        print("\n" + "=" * 72)
        print(f"[trial {trial.number}] eps={epsilon} layers={n_layers} "
              f"k_w={k_w} k_r={k_r}")
        print("=" * 72)

        _restore(model_base.model, clean_snapshot)
        if torch.cuda.is_available():
            torch.cuda.empty_cache()

        row = {
            "trial": trial.number, "model": os.path.basename(model_id).lower(),
            "epsilon": epsilon, "num_layers": n_layers,
            "num_writer_directions": k_w, "num_reader_directions": k_r,
            "ref_undef_base": ref_undef_base,
            "ref_undef_arditi": ref_undef_arditi,
            "ref_undef_pca8": ref_undef_pca8,
            "bpb_undef": util_undef.get("bpb"),
            "mmlu_undef": util_undef.get("mmlu"),
            "math500_undef": util_undef.get("math500"),
        }

        try:
            cfg = ObfuscationConfig(
                epsilon=epsilon,
                num_pertinent_layers=n_layers,
                num_calibration_prompts=args.num_calibration_prompts,
                seed=args.seed,
                projection_mode="full",
                per_layer_direction=True,
                writer_output_directions=True,
                num_writer_directions=k_w,
                num_reader_directions=k_r,
                forward_batch_size=args.forward_batch_size,
            )
            obf = apply_obfuscation(
                model=model_base.model,
                tokenize_fn=model_base.tokenize_instructions_fn,
                harmful_prompts=harmful_train,
                harmless_prompts=harmless_train,
                mean_diffs=mean_diffs_train,
                selected_pos=pos,
                selected_layer=layer,
                direction=direction,
                cfg=cfg,
                ablation_scores=ablation_scores,
            )
            pertinent = obf["pertinent_layers"]

            ref_def_base = _refusal([], [])
            def_arditi_res = evaluate_abliteration_resistance(
                model=model_base.model,
                tokenizer=model_base.tokenizer,
                tokenize_fn=model_base.tokenize_instructions_fn,
                block_modules=model_base.model_block_modules,
                attn_modules=model_base.model_attn_modules,
                mlp_modules=model_base.model_mlp_modules,
                harmful_prompts=harmful_test,
                benign_prompts=harmless_test,
                original_direction=direction,
                refusal_toks=model_base.refusal_toks,
                batch_size=args.attack_batch_size,
                pertinent_layers=list(pertinent),
            )
            ref_def_arditi = def_arditi_res["arditi_refusal_score"]
            avg_cos_sim = def_arditi_res["mean_cos_sim"]

            def_pca_dirs, _ = _extract_pca_directions(
                model_base.model,
                model_base.tokenizer,
                model_base.tokenize_instructions_fn,
                model_base.model_block_modules,
                harmful_test,
                harmless_test,
                top_k=8,
                batch_size=args.attack_batch_size,
            )
            dp_pre, dp_post = _build_pca_hooks(
                model_base.model_block_modules,
                model_base.model_attn_modules,
                model_base.model_mlp_modules,
                def_pca_dirs,
            )
            ref_def_pca8 = _refusal(dp_pre, dp_post)

            arditi_gap = ref_undef_base - ref_def_arditi
            pca8_gap = ref_undef_pca8 - ref_def_pca8
            refusal_gap = arditi_gap + pca8_gap

            row.update({
                "avg_cos_sim": avg_cos_sim,
                "ref_def_base": ref_def_base,
                "ref_def_arditi": ref_def_arditi,
                "ref_def_pca8": ref_def_pca8,
                "arditi_gap": arditi_gap,
                "pca8_gap": pca8_gap,
                "refusal_gap": refusal_gap,
            })
            print(f"  def_base={ref_def_base:.4f}  def_arditi={ref_def_arditi:.4f}  "
                  f"def_pca8={ref_def_pca8:.4f}")
            print(f"  arditi_gap={arditi_gap:.4f}  pca8_gap={pca8_gap:.4f}  "
                  f"refusal_gap={refusal_gap:.4f}")

            util_def = {"bpb": None, "mmlu": None, "math500": None}
            utility_loss = 0.0
            if measure_utility:
                util_def = _measure_utility(
                    model_base, [], [], args.bpb_batches, args.mmlu_n,
                    args.math500_n, args.utility_batch_size,
                )
                losses = _utility_loss(util_undef, util_def)
                utility_loss = losses["utility_loss"]
                row.update({
                    "bpb_def": util_def.get("bpb"),
                    "mmlu_def": util_def.get("mmlu"),
                    "math500_def": util_def.get("math500"),
                    "bpb_loss": losses["bpb_loss"],
                    "mmlu_loss": losses["mmlu_loss"],
                    "math500_loss": losses["math500_loss"],
                    "utility_loss": utility_loss,
                })
                print(f"  def bpb={util_def.get('bpb'):.4f}  "
                      f"mmlu={util_def.get('mmlu'):.4f}  "
                      f"math500={util_def.get('math500'):.4f}")
                print(f"  utility_loss={utility_loss:.4f}")

            row["status"] = "ok"
            return (refusal_gap, utility_loss) if measure_utility else refusal_gap
        except Exception as exc:
            import traceback
            traceback.print_exc()
            row["status"] = f"error: {exc!s}"
            return (float("inf"), float("inf")) if measure_utility else float("inf")
        finally:
            with open(csv_path, "a", newline="") as f:
                csv.writer(f).writerow([row.get(k, "") for k in CSV_HEADER])
            _restore(model_base.model, clean_snapshot)
            gc.collect()
            if torch.cuda.is_available():
                torch.cuda.empty_cache()

    if measure_utility:
        # Multi-objective TPE: sample-efficient at small budgets (~30 trials)
        # where NSGA-II would still be filling its initial population.
        sampler = optuna.samplers.TPESampler(seed=args.seed, multivariate=True)
        study = optuna.create_study(
            study_name=study_name, storage=storage,
            sampler=sampler, directions=["minimize", "minimize"],
            load_if_exists=True,
        )
    else:
        sampler = optuna.samplers.TPESampler(seed=args.seed)
        study = optuna.create_study(
            study_name=study_name, storage=storage,
            sampler=sampler, direction="minimize", load_if_exists=True,
        )

    study.optimize(objective, n_trials=args.n_trials, show_progress_bar=False)

    print("\n" + "=" * 72)
    if measure_utility:
        pareto = study.best_trials
        print(f"[optuna] done. Pareto front has {len(pareto)} trials:")
        for t in sorted(pareto, key=lambda x: x.values[0]):
            print(f"  trial #{t.number}  refusal_gap={t.values[0]:.4f}  "
                  f"utility_loss={t.values[1]:.4f}  params={t.params}")
    else:
        print(f"[optuna] done. best trial #{study.best_trial.number} "
              f"objective={study.best_trial.value:.4f}")
        print(f"  params={study.best_trial.params}")
    print(f"  csv={csv_path}")
    print(f"  db={db_path}")


if __name__ == "__main__":
    main()
