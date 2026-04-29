#!/usr/bin/env python3
"""
Optuna TPE sweep over (epsilon, num_layers, k_w, k_r) for APRS.

Loads model + extracts undefended refusal direction + measures undefended
attack baselines once, then runs `n_trials` TPE-suggested configs by
restoring weight snapshots between trials.  Objective minimises the
composite refusal gap

    (undef_base − def_arditi) + (undef_pca8 − def_pca8)

against the contemporary-weight Arditi and PCA-8 attacks.  Trial results are
logged to an Optuna SQLite study + a flat CSV mirroring sweep_epsilon_k.py.
Pair with `validate_top_k.py` to re-run top-k winners with full utility.
"""

import argparse
import csv
import datetime
import gc
import json
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
    "arditi_gap", "pca8_gap", "objective", "status",
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


def main():
    pa = argparse.ArgumentParser()
    pa.add_argument("--model", choices=list(CONFIGS.keys()), required=True)
    pa.add_argument("--n_trials", type=int, default=30)
    pa.add_argument("--n", type=int, default=20,
                    help="Harmful/harmless prompts for attack eval per trial")
    pa.add_argument("--num_calibration_prompts", type=int, default=64)
    pa.add_argument("--attack_batch_size", type=int, default=64)
    pa.add_argument("--forward_batch_size", type=int, default=64)
    pa.add_argument("--seed", type=int, default=42)
    pa.add_argument("--study_name", default=None)
    pa.add_argument("--storage", default=None,
                    help="Optuna storage URL (e.g. sqlite:///study.db); default = SQLite next to CSV")
    pa.add_argument("--output_dir",
                    default=os.path.join(REPO_DIR, "results", "optuna_sweep"))
    args = pa.parse_args()

    os.makedirs(args.output_dir, exist_ok=True)
    ts = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
    csv_path = os.path.join(args.output_dir, f"optuna_{args.model}_{ts}.csv")
    db_path = os.path.join(args.output_dir, f"optuna_{args.model}_{ts}.db")
    study_name = args.study_name or f"aprs_{args.model}_{ts}"
    storage = args.storage or f"sqlite:///{db_path}"

    model_id, _, _ = CONFIGS[args.model]
    print(f"[optuna] model={model_id} trials={args.n_trials} csv={csv_path}")
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

    print("\n[optuna] measuring undefended baselines ...")
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

    clean_snapshot = _save(model_base.model)

    # CSV header (first writer)
    if not os.path.exists(csv_path):
        with open(csv_path, "w", newline="") as f:
            csv.writer(f).writerow(CSV_HEADER)

    # ── Objective ───────────────────────────────────────────────────────────
    def objective(trial: optuna.trial.Trial) -> float:
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
            "ref_undef_base": ref_undef_base, "ref_undef_arditi": ref_undef_arditi,
            "ref_undef_pca8": ref_undef_pca8,
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
            objective_val = arditi_gap + pca8_gap

            row.update({
                "avg_cos_sim": avg_cos_sim,
                "ref_def_base": ref_def_base,
                "ref_def_arditi": ref_def_arditi,
                "ref_def_pca8": ref_def_pca8,
                "arditi_gap": arditi_gap,
                "pca8_gap": pca8_gap,
                "objective": objective_val,
                "status": "ok",
            })
            print(f"  def_base={ref_def_base:.4f}  def_arditi={ref_def_arditi:.4f}  "
                  f"def_pca8={ref_def_pca8:.4f}")
            print(f"  arditi_gap={arditi_gap:.4f}  pca8_gap={pca8_gap:.4f}  "
                  f"objective={objective_val:.4f}")
            return objective_val
        except Exception as exc:
            import traceback
            traceback.print_exc()
            row["status"] = f"error: {exc!s}"
            return float("inf")
        finally:
            with open(csv_path, "a", newline="") as f:
                csv.writer(f).writerow([row.get(k, "") for k in CSV_HEADER])
            _restore(model_base.model, clean_snapshot)
            gc.collect()
            if torch.cuda.is_available():
                torch.cuda.empty_cache()

    sampler = optuna.samplers.TPESampler(seed=args.seed)
    study = optuna.create_study(
        study_name=study_name, storage=storage,
        sampler=sampler, direction="minimize", load_if_exists=True,
    )
    study.optimize(objective, n_trials=args.n_trials, show_progress_bar=False)

    print("\n" + "=" * 72)
    print(f"[optuna] done. best trial #{study.best_trial.number} "
          f"objective={study.best_trial.value:.4f}")
    print(f"  params={study.best_trial.params}")
    print(f"  csv={csv_path}")
    print(f"  db={db_path}")


if __name__ == "__main__":
    main()
