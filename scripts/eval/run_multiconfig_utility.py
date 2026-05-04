#!/usr/bin/env python3
"""
Multi-config utility runner: loads model once, measures undefended baseline
once, then evaluates each (epsilon, num_layers, k_w, k_r) config in sequence.
Avoids the per-config model reload overhead of validate_winner.py.
"""

import argparse
import csv
import datetime
import functools
import json
import math
import os
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
from benchmarks.evaluate_lm_harness import run_lm_harness
from pipeline.submodules.evaluate_loss import evaluate_loss
from obfuscation_config import ObfuscationConfig
from quick_attack_test import CONFIGS, SYSTEM_PROMPT, _QWEN3_TEMPLATE, _restore, _save
from run_obfuscation_pipeline import filter_data, load_mlabonne_datasets


QWEN_CONFIGS = [
    # (epsilon, num_layers, k_w, k_r, label)
    (0.5,  25, 2,  4,  "trial7"),
    (0.2,  25, 4,  1,  "trial5"),
    (0.5,  10, 4,  1,  "trial12"),
    (0.5,  40, 4, 16,  "trial26"),
]


def _setup_tokenizer(model_base, model_id):
    if "qwen3" in model_id.lower():
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
        model_base.tokenize_instructions_fn = functools.partial(_qwen3_tok, system=SYSTEM_PROMPT)
    elif "gemma" not in model_id.lower():
        model_base.tokenize_instructions_fn = functools.partial(
            model_base.tokenize_instructions_fn, system=SYSTEM_PROMPT
        )


def _measure_utility(model_base, fwd_pre, fwd_post, args, seed=42):
    out = {}
    if args.bpb_batches > 0:
        res = evaluate_loss(
            model_base, fwd_pre_hooks=fwd_pre, fwd_hooks=fwd_post,
            batch_size=args.batch_size, n_batches=args.bpb_batches,
            dataset_labels=["pile"],
        )
        pile = res["pile"]
        out["bpb"] = pile.get("bpb") or pile["ce_loss"] / math.log(2)
    if args.mmlu_n > 0:
        r = run_lm_harness(
            model=model_base.model, tokenizer=model_base.tokenizer,
            tasks=["mmlu"], n_samples=args.mmlu_n, batch_size=args.batch_size,
            seed=seed, fwd_pre_hooks=fwd_pre, fwd_hooks=fwd_post,
        )
        out["mmlu"] = r.get("mmlu", {}).get("acc")
    if args.math500_n > 0:
        r = run_lm_harness(
            model=model_base.model, tokenizer=model_base.tokenizer,
            tasks=["math500"], n_samples=args.math500_n, batch_size=args.batch_size,
            seed=seed, fwd_pre_hooks=fwd_pre, fwd_hooks=fwd_post,
        )
        out["math500"] = r.get("math500", {}).get("exact_match")
    return out


def main():
    pa = argparse.ArgumentParser()
    pa.add_argument("--model", default="qwen", choices=list(CONFIGS.keys()))
    pa.add_argument("--math500_n", type=int, default=200)
    pa.add_argument("--mmlu_n", type=int, default=200)
    pa.add_argument("--bpb_batches", type=int, default=32)
    pa.add_argument("--batch_size", type=int, default=64)
    pa.add_argument("--num_calibration_prompts", type=int, default=64)
    pa.add_argument("--forward_batch_size", type=int, default=64)
    pa.add_argument("--seed", type=int, default=42)
    pa.add_argument("--output_dir",
                    default=os.path.join(REPO_DIR, "results", "winner_validation"))
    args = pa.parse_args()

    os.makedirs(args.output_dir, exist_ok=True)
    model_id, _, _ = CONFIGS[args.model]

    print(f"[multi] model={model_id}  batch_size={args.batch_size}")
    print(f"[multi] math500_n={args.math500_n}  mmlu_n={args.mmlu_n}  bpb_batches={args.bpb_batches}")

    model_base = construct_model_base(model_id)
    _setup_tokenizer(model_base, model_id)

    harmful_train, harmless_train, harmful_val, harmless_val = load_mlabonne_datasets(n_train=400, n_val=100)
    print("[multi] filtering data ...")
    harmful_train, harmless_train, harmful_val, harmless_val = filter_data(
        model_base, harmful_train, harmless_train, harmful_val, harmless_val,
    )

    print("[multi] extracting refusal direction ...")
    with tempfile.TemporaryDirectory() as tmp:
        mean_diffs_train = generate_directions(model_base, harmful_train, harmless_train, artifact_dir=tmp)
        pos, layer, direction = select_direction(
            model_base, harmful_val, harmless_val, mean_diffs_train, artifact_dir=tmp,
        )
        with open(os.path.join(tmp, "direction_evaluations.json")) as f:
            ablation_scores = json.load(f)

    clean_snapshot = _save(model_base.model)

    print("\n[multi] measuring undefended utility ...")
    undef = _measure_utility(model_base, [], [], args, args.seed)
    print(f"  bpb={undef.get('bpb')}  mmlu={undef.get('mmlu')}  math500={undef.get('math500')}")

    results = []
    for eps, n_layers, k_w, k_r, label in QWEN_CONFIGS:
        print(f"\n[multi] === {label}: ε={eps} L={n_layers} k_w={k_w} k_r={k_r} ===")
        _restore(model_base.model, clean_snapshot)
        if torch.cuda.is_available():
            torch.cuda.empty_cache()

        cfg = ObfuscationConfig(
            epsilon=eps,
            num_pertinent_layers=n_layers,
            num_calibration_prompts=args.num_calibration_prompts,
            seed=42,
            projection_mode="full",
            per_layer_direction=True,
            writer_output_directions=True,
            num_writer_directions=k_w,
            num_reader_directions=k_r,
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

        print(f"  measuring defended utility ...")
        defended = _measure_utility(model_base, [], [], args, args.seed)
        print(f"  bpb={defended.get('bpb')}  mmlu={defended.get('mmlu')}  math500={defended.get('math500')}")

        bpb_loss   = (defended.get("bpb", 0) or 0) - (undef.get("bpb", 0) or 0)
        mmlu_loss  = (undef.get("mmlu", 0) or 0) - (defended.get("mmlu", 0) or 0)
        m500_loss  = (undef.get("math500", 0) or 0) - (defended.get("math500", 0) or 0)
        bpb_u   = undef.get("bpb") or 1.0
        mmlu_u  = undef.get("mmlu") or 1.0
        m500_u  = undef.get("math500") or 1.0
        utility_loss = bpb_loss / bpb_u + mmlu_loss / mmlu_u + m500_loss / m500_u

        row = {
            "label": label, "epsilon": eps, "num_layers": n_layers,
            "k_w": k_w, "k_r": k_r,
            "bpb_undef": undef.get("bpb"), "mmlu_undef": undef.get("mmlu"),
            "math500_undef": undef.get("math500"),
            "bpb_def": defended.get("bpb"), "mmlu_def": defended.get("mmlu"),
            "math500_def": defended.get("math500"),
            "bpb_loss": bpb_loss, "mmlu_loss": mmlu_loss, "math500_loss": m500_loss,
            "utility_loss": utility_loss,
        }
        results.append(row)
        print(f"  utility_loss={utility_loss:.4f}")

    # Save CSV
    ts = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
    csv_path = os.path.join(args.output_dir, f"qwen_multiconfig_{ts}.csv")
    with open(csv_path, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(results[0].keys()))
        w.writeheader()
        w.writerows(results)

    print("\n" + "=" * 72)
    print(f"{'label':<10} {'ε':>6} {'L':>4} {'kw':>4} {'kr':>4} {'utility_loss':>14} {'mmlu_def':>10} {'math500_def':>12}")
    print("-" * 72)
    for r in results:
        print(f"{r['label']:<10} {r['epsilon']:>6} {r['num_layers']:>4} {r['k_w']:>4} {r['k_r']:>4} "
              f"{r['utility_loss']:>14.4f} {(r['mmlu_def'] or 0):>10.4f} {(r['math500_def'] or 0):>12.4f}")
    print("=" * 72)
    print(f"\n[multi] saved → {csv_path}")


if __name__ == "__main__":
    main()
