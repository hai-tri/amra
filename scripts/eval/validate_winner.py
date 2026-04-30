#!/usr/bin/env python3
"""
Validate a single APRS config with full MATH500 + MMLU at multiple seeds.

Loads the model + extracts undefended refusal direction once, applies the
defense once, then runs full MATH500 (n=500), MMLU (n configurable), and
Pile BPB across `seeds` LM-Eval-Harness seeds. Reports mean ± std per
metric.

Use to validate Optuna sweep winners with paper-quality numbers.
"""

import argparse
import csv
import datetime
import functools
import json
import math
import os
import statistics
import sys
import tempfile

os.environ.setdefault("MPLCONFIGDIR", "/tmp/mplconfig")
os.environ.setdefault("FONTCONFIG_PATH", "/tmp/fontconfig")
os.makedirs(os.environ["MPLCONFIGDIR"], exist_ok=True)
os.makedirs(os.environ["FONTCONFIG_PATH"], exist_ok=True)

import torch

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
from benchmarks.evaluate_lm_harness import run_lm_harness
from pipeline.submodules.evaluate_loss import evaluate_loss
from obfuscation_config import ObfuscationConfig
from quick_attack_test import (
    CONFIGS, SYSTEM_PROMPT, _QWEN3_TEMPLATE, _build_pca_hooks,
    _extract_pca_directions, _restore, _save,
)
from run_obfuscation_pipeline import filter_data, load_mlabonne_datasets


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


def _measure_one_seed(model_base, fwd_pre, fwd_post, mmlu_n, math500_n,
                      bpb_batches, batch_size, seed):
    out = {"bpb": None, "mmlu": None, "math500": None}
    if bpb_batches and bpb_batches > 0:
        bpb_res = evaluate_loss(
            model_base, fwd_pre_hooks=fwd_pre, fwd_hooks=fwd_post,
            batch_size=batch_size, n_batches=bpb_batches, dataset_labels=["pile"],
        )
        pile = bpb_res["pile"]
        out["bpb"] = pile.get("bpb") or pile["ce_loss"] / math.log(2)
    if mmlu_n and mmlu_n > 0:
        mmlu_res = run_lm_harness(
            model=model_base.model, tokenizer=model_base.tokenizer,
            tasks=["mmlu"], n_samples=mmlu_n, batch_size=batch_size,
            seed=seed, fwd_pre_hooks=fwd_pre, fwd_hooks=fwd_post,
        )
        out["mmlu"] = mmlu_res.get("mmlu", {}).get("acc")
    if math500_n and math500_n > 0:
        math_res = run_lm_harness(
            model=model_base.model, tokenizer=model_base.tokenizer,
            tasks=["math500"], n_samples=math500_n, batch_size=batch_size,
            seed=seed, fwd_pre_hooks=fwd_pre, fwd_hooks=fwd_post,
        )
        out["math500"] = math_res.get("math500", {}).get("exact_match")
    return out


def _agg(values, fmt="{:.4f}"):
    if not values:
        return "—"
    if len(values) == 1:
        return fmt.format(values[0])
    return f"{fmt.format(statistics.mean(values))} ± {fmt.format(statistics.stdev(values))}"


def main():
    pa = argparse.ArgumentParser()
    pa.add_argument("--model", required=True, choices=list(CONFIGS.keys()))
    pa.add_argument("--epsilon", type=float, required=True)
    pa.add_argument("--num_layers", type=int, required=True)
    pa.add_argument("--k_w", type=int, required=True)
    pa.add_argument("--k_r", type=int, required=True)
    pa.add_argument("--seeds", default="42,43,44",
                    help="Comma-separated lm-harness seeds")
    pa.add_argument("--math500_n", type=int, default=500)
    pa.add_argument("--mmlu_n", type=int, default=2000)
    pa.add_argument("--bpb_batches", type=int, default=128)
    pa.add_argument("--batch_size", type=int, default=8)
    pa.add_argument("--num_calibration_prompts", type=int, default=64)
    pa.add_argument("--forward_batch_size", type=int, default=64)
    pa.add_argument("--output_dir",
                    default=os.path.join(REPO_DIR, "results", "winner_validation"))
    args = pa.parse_args()

    seeds = [int(s) for s in args.seeds.split(",") if s.strip()]
    os.makedirs(args.output_dir, exist_ok=True)
    ts = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
    out_path = os.path.join(
        args.output_dir,
        f"validate_{args.model}_eps{args.epsilon}_L{args.num_layers}_"
        f"kw{args.k_w}_kr{args.k_r}_{ts}.json",
    )
    print(f"[validate] {args.model} ε={args.epsilon} L={args.num_layers} "
          f"k_w={args.k_w} k_r={args.k_r}  seeds={seeds}")
    print(f"[validate] math500_n={args.math500_n} mmlu_n={args.mmlu_n} "
          f"bpb_batches={args.bpb_batches}")

    # ── Setup once ──────────────────────────────────────────────────────────
    model_id, _, _ = CONFIGS[args.model]
    model_base = construct_model_base(model_id)
    _setup_tokenizer(model_base, model_id)

    harmful_train, harmless_train, harmful_val, harmless_val = load_mlabonne_datasets(
        n_train=400, n_val=100,
    )
    print("[validate] filtering data ...")
    harmful_train, harmless_train, harmful_val, harmless_val = filter_data(
        model_base, harmful_train, harmless_train, harmful_val, harmless_val,
    )

    print("[validate] extracting undefended refusal direction ...")
    with tempfile.TemporaryDirectory() as tmp:
        mean_diffs_train = generate_directions(
            model_base, harmful_train, harmless_train, artifact_dir=tmp,
        )
        pos, layer, direction = select_direction(
            model_base, harmful_val, harmless_val, mean_diffs_train, artifact_dir=tmp,
        )
        with open(os.path.join(tmp, "direction_evaluations.json")) as f:
            ablation_scores = json.load(f)

    clean_snapshot = _save(model_base.model)

    # ── Undefended utility (multiple seeds) ─────────────────────────────────
    print(f"\n[validate] undefended utility across {len(seeds)} seeds ...")
    undef_results = {"bpb": [], "mmlu": [], "math500": []}
    for s in seeds:
        print(f"  seed={s}")
        r = _measure_one_seed(
            model_base, [], [], args.mmlu_n, args.math500_n,
            args.bpb_batches, args.batch_size, s,
        )
        for k in undef_results:
            if r.get(k) is not None:
                undef_results[k].append(r[k])
        print(f"    bpb={r['bpb']}  mmlu={r['mmlu']}  math500={r['math500']}")

    # ── Apply defense once ──────────────────────────────────────────────────
    print(f"\n[validate] applying defense ε={args.epsilon} L={args.num_layers} "
          f"k_w={args.k_w} k_r={args.k_r} ...")
    cfg = ObfuscationConfig(
        epsilon=args.epsilon,
        num_pertinent_layers=args.num_layers,
        num_calibration_prompts=args.num_calibration_prompts,
        seed=42,
        projection_mode="full",
        per_layer_direction=True,
        writer_output_directions=True,
        num_writer_directions=args.k_w,
        num_reader_directions=args.k_r,
        forward_batch_size=args.forward_batch_size,
    )
    apply_obfuscation(
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

    # ── Defended utility (multiple seeds) ───────────────────────────────────
    print(f"\n[validate] defended utility across {len(seeds)} seeds ...")
    def_results = {"bpb": [], "mmlu": [], "math500": []}
    for s in seeds:
        print(f"  seed={s}")
        r = _measure_one_seed(
            model_base, [], [], args.mmlu_n, args.math500_n,
            args.bpb_batches, args.batch_size, s,
        )
        for k in def_results:
            if r.get(k) is not None:
                def_results[k].append(r[k])
        print(f"    bpb={r['bpb']}  mmlu={r['mmlu']}  math500={r['math500']}")

    # ── Report ──────────────────────────────────────────────────────────────
    print("\n" + "=" * 72)
    print(f"VALIDATION SUMMARY — {args.model}")
    print(f"  config: ε={args.epsilon}, L={args.num_layers}, "
          f"k_w={args.k_w}, k_r={args.k_r}")
    print(f"  seeds: {seeds}")
    print(f"  MATH500 n={args.math500_n}, MMLU n={args.mmlu_n}, "
          f"BPB batches={args.bpb_batches}")
    print("=" * 72)
    print(f"{'metric':<10} {'undefended':>20} {'defended':>20} {'Δ':>15}")
    for k, fmt in [("bpb", "{:.4f}"), ("mmlu", "{:.4f}"), ("math500", "{:.4f}")]:
        u = undef_results[k]; d = def_results[k]
        delta = (statistics.mean(d) - statistics.mean(u)) if (u and d) else None
        print(f"{k:<10} {_agg(u, fmt):>20} {_agg(d, fmt):>20} "
              f"{f'{delta:+.4f}' if delta is not None else '—':>15}")

    out = {
        "model": model_id, "model_key": args.model,
        "config": {
            "epsilon": args.epsilon, "num_layers": args.num_layers,
            "k_w": args.k_w, "k_r": args.k_r,
        },
        "seeds": seeds, "math500_n": args.math500_n,
        "mmlu_n": args.mmlu_n, "bpb_batches": args.bpb_batches,
        "undefended": undef_results, "defended": def_results,
    }
    with open(out_path, "w") as f:
        json.dump(out, f, indent=2)
    print(f"\n[validate] saved → {out_path}")


if __name__ == "__main__":
    main()
