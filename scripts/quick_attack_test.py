#!/usr/bin/env python3
"""
quick_attack_test.py — Local Mac smoke test for Arditi abliteration + PCA-8 attack.

Correct threat model:
  1. Extract refusal direction from *undefended* model.
  2. Filter to harmful prompts the base model actually refuses (score > 0).
  3. Undefended baseline: apply Arditi abliteration on original direction
     → model should STOP refusing (attack works on undefended model).
  4. Apply APRS defense (weight edits).
  5. Defended baseline: abliterate using direction re-extracted from defended
     model's activations → model should STILL refuse (defense holds).
  6. PCA-8 variant: same but with top-8 principal components.

Usage:
    python scripts/quick_attack_test.py [--model llama|qwen|gemma|all] [--n 20]
"""

import argparse
import functools
import math
import os
import sys
import tempfile
import torch

REPO_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, REPO_DIR)
sys.path.insert(0, os.path.join(REPO_DIR, "refusal_direction"))

from pipeline.model_utils.model_factory import construct_model_base
from pipeline.submodules.generate_directions import generate_directions
from pipeline.submodules.select_direction import select_direction, get_refusal_scores
from pipeline.submodules.evaluate_loss import evaluate_loss
from pipeline.utils.hook_utils import (
    get_direction_ablation_input_pre_hook,
    get_direction_ablation_output_hook,
)
from apply_obfuscation import apply_obfuscation
from obfuscation_config import ObfuscationConfig
from run_obfuscation_pipeline import load_mlabonne_datasets, filter_data
from attacks.evaluate_abliteration import evaluate_abliteration_resistance, extract_refusal_direction
from attacks.evaluate_adaptive_attack import pca_multi_direction_attack
from benchmarks.evaluate_lm_harness import run_lm_harness

SYSTEM_PROMPT = "You are a helpful assistant."

CONFIGS = {
    "llama": ("meta-llama/Meta-Llama-3-8B-Instruct", 0.025, 15),
    "qwen":  ("Qwen/Qwen3-8B",                        0.05,  25),
    "gemma": ("google/gemma-2-9b-it",                 0.01,  42),
}

_QWEN3_TEMPLATE = (
    "<|im_start|>user\n{instruction}<|im_end|>\n"
    "<|im_start|>assistant\n<think>\n\n</think>\n\n"
)

_SNAPSHOT_KEYS = frozenset(
    ["o_proj", "down_proj", "q_proj", "k_proj", "v_proj",
     "gate_proj", "up_proj", "lm_head"]
)


def _save(model):
    return {n: p.data.clone() for n, p in model.named_parameters()
            if any(k in n for k in _SNAPSHOT_KEYS)}


def _restore(model, snap):
    with torch.no_grad():
        for n, p in model.named_parameters():
            if n in snap:
                p.data.copy_(snap[n])


def _refusal_rate(scores):
    return (scores > 0).float().mean().item()


def _build_arditi_hooks(block_modules, attn_modules, mlp_modules, direction):
    """Build forward hooks that project out `direction` at every layer."""
    num_layers = len(block_modules)
    fwd_pre_hooks = [
        (block_modules[ell], get_direction_ablation_input_pre_hook(direction=direction))
        for ell in range(num_layers)
    ]
    fwd_hooks = [
        (attn_modules[ell], get_direction_ablation_output_hook(direction=direction))
        for ell in range(num_layers)
    ] + [
        (mlp_modules[ell], get_direction_ablation_output_hook(direction=direction))
        for ell in range(num_layers)
    ]
    return fwd_pre_hooks, fwd_hooks


def _measure_utility(model_base, fwd_pre_hooks, fwd_hooks,
                     n_batches_bpb=32, n_samples_mmlu=200, batch_size=8):
    """Return (pile_bpb, mmlu_acc) under the given hooks."""
    bpb_result = evaluate_loss(
        model_base,
        fwd_pre_hooks=fwd_pre_hooks,
        fwd_hooks=fwd_hooks,
        batch_size=batch_size,
        n_batches=n_batches_bpb,
        dataset_labels=["pile"],
    )
    # server's evaluate_loss returns ce_loss (nats); convert to bits-per-byte
    pile_bpb = bpb_result["pile"].get("bpb") or \
               bpb_result["pile"]["ce_loss"] / math.log(2)

    lm_result = run_lm_harness(
        model=model_base.model,
        tokenizer=model_base.tokenizer,
        tasks=["mmlu"],
        n_samples=n_samples_mmlu,
        batch_size=batch_size,
        seed=42,
        fwd_pre_hooks=fwd_pre_hooks,
        fwd_hooks=fwd_hooks,
    )
    mmlu_acc = lm_result.get("mmlu", {}).get("acc")
    return pile_bpb, mmlu_acc


def run(model_key: str, n_prompts: int):
    model_id, epsilon, num_layers = CONFIGS[model_key]
    model_tag = os.path.basename(model_id).lower()

    print(f"\n{'='*60}")
    print(f" {model_id}  ε={epsilon}  layers={num_layers}")
    print(f"{'='*60}")

    model_base = construct_model_base(model_id)
    _is_qwen3 = "qwen3" in model_id.lower()
    _is_gemma = "gemma" in model_id.lower()

    if _is_qwen3:
        _tok = model_base.tokenizer
        _orig_apply = _tok.apply_chat_template
        def _no_think_apply(messages, **kwargs):
            kwargs.setdefault("enable_thinking", False)
            return _orig_apply(messages, **kwargs)
        _tok.apply_chat_template = _no_think_apply

        def _qwen3_tokenize(instructions, outputs=None, system=None):
            prompts = [_QWEN3_TEMPLATE.format(instruction=i) for i in instructions]
            if outputs is not None:
                prompts = [p + o for p, o in zip(prompts, outputs)]
            return _tok(prompts, padding=True, truncation=False, return_tensors="pt")
        model_base.tokenize_instructions_fn = functools.partial(
            _qwen3_tokenize, system=SYSTEM_PROMPT
        )
    elif not _is_gemma:
        model_base.tokenize_instructions_fn = functools.partial(
            model_base.tokenize_instructions_fn, system=SYSTEM_PROMPT
        )

    harmful_train, harmless_train, harmful_val, harmless_val = load_mlabonne_datasets(
        n_train=400, n_val=100
    )

    # Keep only examples the base model responds to correctly
    print("\nFiltering examples with base model …")
    harmful_train, harmless_train, harmful_val, harmless_val = filter_data(
        model_base, harmful_train, harmless_train, harmful_val, harmless_val
    )
    print(f"  harmful_train={len(harmful_train)}  harmless_train={len(harmless_train)}")
    print(f"  harmful_val={len(harmful_val)}  harmless_val={len(harmless_val)}")

    # Direction extraction from undefended model
    print("\nExtracting refusal direction from undefended model …")
    with tempfile.TemporaryDirectory() as _tmp:
        mean_diffs = generate_directions(
            model_base, harmful_train, harmless_train, artifact_dir=_tmp,
        )
        pos, layer, direction = select_direction(
            model_base, harmful_val, harmless_val, mean_diffs, artifact_dir=_tmp,
        )
    print(f"Direction: pos={pos}, layer={layer}, ||r||={direction.norm():.4f}")

    harmful_test  = harmful_val[:n_prompts]
    harmless_test = harmless_val[:n_prompts]

    # ----------------------------------------------------------------
    # Phase 1: Undefended model + abliteration (should stop refusing)
    # ----------------------------------------------------------------
    print("\n" + "─"*50)
    print(" Phase 1: UNDEFENDED model")
    print("─"*50)

    baseline_scores_undefended = get_refusal_scores(
        model_base.model, harmful_test,
        model_base.tokenize_instructions_fn, model_base.refusal_toks,
        batch_size=4,
    )
    baseline_mean_undefended = baseline_scores_undefended.mean().item()
    baseline_rate_undefended = _refusal_rate(baseline_scores_undefended)
    print(f"  Baseline refusal: score={baseline_mean_undefended:.4f}  "
          f"rate={baseline_rate_undefended:.2%}")

    print("  Measuring undefended utility (no attack) …")
    undef_bpb_base, undef_mmlu_base = _measure_utility(model_base, [], [])

    undefended_arditi = evaluate_abliteration_resistance(
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
        batch_size=4,
    )

    # Build Arditi hooks from the direction the attacker extracted
    undef_arditi_dir = undefended_arditi["defended_direction"]
    undef_arditi_pre, undef_arditi_post = _build_arditi_hooks(
        model_base.model_block_modules,
        model_base.model_attn_modules,
        model_base.model_mlp_modules,
        undef_arditi_dir,
    )
    print("  Measuring undefended utility under Arditi abliteration …")
    undef_bpb_arditi, undef_mmlu_arditi = _measure_utility(
        model_base, undef_arditi_pre, undef_arditi_post,
    )

    undefended_pca = pca_multi_direction_attack(
        model=model_base.model,
        tokenizer=model_base.tokenizer,
        tokenize_fn=model_base.tokenize_instructions_fn,
        block_modules=model_base.model_block_modules,
        attn_modules=model_base.model_attn_modules,
        mlp_modules=model_base.model_mlp_modules,
        harmful_prompts=harmful_test,
        benign_prompts=harmless_test,
        refusal_toks=model_base.refusal_toks,
        top_k=8,
        batch_size=4,
    )

    print(f"  Arditi post-abliteration: score={undefended_arditi['arditi_refusal_score']:.4f}")
    print(f"  PCA-8  post-abliteration: score={undefended_pca['post_attack_refusal_score']:.4f}")

    # ----------------------------------------------------------------
    # Phase 2: APRS defense + abliteration (should still refuse)
    # ----------------------------------------------------------------
    cfg = ObfuscationConfig(
        epsilon=epsilon,
        num_pertinent_layers=num_layers,
        num_calibration_prompts=64,
        seed=42,
        projection_mode="full",
        per_layer_direction=True,
        writer_output_directions=True,
    )
    snap = _save(model_base.model)
    try:
        print("\n" + "─"*50)
        print(" Applying APRS defense …")
        print("─"*50)
        obf = apply_obfuscation(
            model=model_base.model,
            tokenize_fn=model_base.tokenize_instructions_fn,
            harmful_prompts=harmful_train,
            harmless_prompts=harmless_train,
            mean_diffs=mean_diffs,
            selected_pos=pos, selected_layer=layer,
            direction=direction, cfg=cfg,
            ablation_scores=None,
        )
        pertinent = obf["pertinent_layers"]
        print(f"Pertinent layers ({len(pertinent)}): {sorted(pertinent)}")

        print("\n" + "─"*50)
        print(" Phase 2: DEFENDED model")
        print("─"*50)

        baseline_scores_defended = get_refusal_scores(
            model_base.model, harmful_test,
            model_base.tokenize_instructions_fn, model_base.refusal_toks,
            batch_size=4,
        )
        baseline_mean_defended = baseline_scores_defended.mean().item()
        baseline_rate_defended = _refusal_rate(baseline_scores_defended)
        print(f"  Baseline refusal: score={baseline_mean_defended:.4f}  "
              f"rate={baseline_rate_defended:.2%}")

        print("  Measuring defended utility (no attack) …")
        def_bpb_base, def_mmlu_base = _measure_utility(model_base, [], [])

        defended_arditi = evaluate_abliteration_resistance(
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
            batch_size=4,
            pertinent_layers=list(pertinent),
        )

        def_arditi_dir = defended_arditi["defended_direction"]
        def_arditi_pre, def_arditi_post = _build_arditi_hooks(
            model_base.model_block_modules,
            model_base.model_attn_modules,
            model_base.model_mlp_modules,
            def_arditi_dir,
        )
        print("  Measuring defended utility under Arditi abliteration …")
        def_bpb_arditi, def_mmlu_arditi = _measure_utility(
            model_base, def_arditi_pre, def_arditi_post,
        )

        defended_pca = pca_multi_direction_attack(
            model=model_base.model,
            tokenizer=model_base.tokenizer,
            tokenize_fn=model_base.tokenize_instructions_fn,
            block_modules=model_base.model_block_modules,
            attn_modules=model_base.model_attn_modules,
            mlp_modules=model_base.model_mlp_modules,
            harmful_prompts=harmful_test,
            benign_prompts=harmless_test,
            refusal_toks=model_base.refusal_toks,
            top_k=8,
            batch_size=4,
        )

        def _f(v):
            return f"{v:.4f}" if v is not None else "N/A"

        print(f"\n{'='*60}")
        print(f" RESULTS — {model_tag}")
        print(f"{'='*60}")
        print(f"  {'':35s} {'ref_score':>9} {'pile_bpb':>9} {'mmlu':>7}")
        print(f"  {'─'*62}")
        print(f"  {'Undefended baseline':35s} "
              f"{baseline_mean_undefended:>9.4f} "
              f"{undef_bpb_base:>9.4f} "
              f"{_f(undef_mmlu_base):>7}")
        print(f"  {'Undefended + Arditi abliteration':35s} "
              f"{undefended_arditi['arditi_refusal_score']:>9.4f} "
              f"{undef_bpb_arditi:>9.4f} "
              f"{_f(undef_mmlu_arditi):>7}")
        print(f"  {'Undefended + PCA-8 abliteration':35s} "
              f"{undefended_pca['post_attack_refusal_score']:>9.4f} "
              f"{'—':>9} {'—':>7}")
        print(f"  {'─'*62}")
        print(f"  {'Defended baseline':35s} "
              f"{baseline_mean_defended:>9.4f} "
              f"{def_bpb_base:>9.4f} "
              f"{_f(def_mmlu_base):>7}")
        print(f"  {'Defended + Arditi abliteration':35s} "
              f"{defended_arditi['arditi_refusal_score']:>9.4f} "
              f"{def_bpb_arditi:>9.4f} "
              f"{_f(def_mmlu_arditi):>7}")
        print(f"  {'Defended + PCA-8 abliteration':35s} "
              f"{defended_pca['post_attack_refusal_score']:>9.4f} "
              f"{'—':>9} {'—':>7}")
        print(f"  avg_cos_sim (defended vs original dir): "
              f"{defended_arditi['mean_cos_sim']:.4f}")

        return {
            "model":               model_tag,
            "undef_baseline":      baseline_mean_undefended,
            "undef_arditi":        undefended_arditi["arditi_refusal_score"],
            "undef_pca8":          undefended_pca["post_attack_refusal_score"],
            "undef_bpb_base":      undef_bpb_base,
            "undef_bpb_arditi":    undef_bpb_arditi,
            "undef_mmlu_base":     undef_mmlu_base,
            "undef_mmlu_arditi":   undef_mmlu_arditi,
            "def_baseline":        baseline_mean_defended,
            "def_arditi":          defended_arditi["arditi_refusal_score"],
            "def_pca8":            defended_pca["post_attack_refusal_score"],
            "def_bpb_base":        def_bpb_base,
            "def_bpb_arditi":      def_bpb_arditi,
            "def_mmlu_base":       def_mmlu_base,
            "def_mmlu_arditi":     def_mmlu_arditi,
            "avg_cos_sim":         defended_arditi["mean_cos_sim"],
        }

    finally:
        _restore(model_base.model, snap)
        del model_base
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
        elif hasattr(torch.mps, "empty_cache"):
            torch.mps.empty_cache()


def main():
    pa = argparse.ArgumentParser()
    pa.add_argument("--model", choices=["llama", "qwen", "gemma", "all"],
                    default="all")
    pa.add_argument("--n", type=int, default=20,
                    help="Harmful/harmless prompts for attack eval")
    args = pa.parse_args()

    keys = list(CONFIGS.keys()) if args.model == "all" else [args.model]
    results = []
    for k in keys:
        try:
            r = run(k, args.n)
            if r:
                results.append(r)
        except Exception as e:
            print(f"[ERROR] {k}: {e}")
            import traceback; traceback.print_exc()

    if results:
        def _f(v):
            return f"{v:.4f}" if v is not None else "N/A"

        print(f"\n{'='*60}")
        print(" SUMMARY — Refusal scores")
        print(f"{'='*60}")
        hdr = f"{'Model':<30} {'undef_base':>10} {'undef_ard':>10} {'undef_pca':>10} {'def_base':>9} {'def_ard':>8} {'def_pca':>8} {'cos_sim':>8}"
        print(hdr)
        print("─" * len(hdr))
        for r in results:
            print(
                f"{r['model']:<30} "
                f"{r['undef_baseline']:>10.4f} "
                f"{r['undef_arditi']:>10.4f} "
                f"{r['undef_pca8']:>10.4f} "
                f"{r['def_baseline']:>9.4f} "
                f"{r['def_arditi']:>8.4f} "
                f"{r['def_pca8']:>8.4f} "
                f"{r['avg_cos_sim']:>8.4f}"
            )

        print(f"\n{'='*60}")
        print(" SUMMARY — Pile BPB (lower = better)")
        print(f"{'='*60}")
        hdr2 = f"{'Model':<30} {'undef_base':>10} {'undef+ard':>10} {'Δundef':>7} {'def_base':>9} {'def+ard':>8} {'Δdef':>6}"
        print(hdr2)
        print("─" * len(hdr2))
        for r in results:
            d_undef = r["undef_bpb_arditi"] - r["undef_bpb_base"]
            d_def   = r["def_bpb_arditi"]   - r["def_bpb_base"]
            print(
                f"{r['model']:<30} "
                f"{r['undef_bpb_base']:>10.4f} "
                f"{r['undef_bpb_arditi']:>10.4f} "
                f"{d_undef:>+7.4f} "
                f"{r['def_bpb_base']:>9.4f} "
                f"{r['def_bpb_arditi']:>8.4f} "
                f"{d_def:>+6.4f}"
            )

        print(f"\n{'='*60}")
        print(" SUMMARY — MMLU accuracy (higher = better)")
        print(f"{'='*60}")
        hdr3 = f"{'Model':<30} {'undef_base':>10} {'undef+ard':>10} {'Δundef':>7} {'def_base':>9} {'def+ard':>8} {'Δdef':>6}"
        print(hdr3)
        print("─" * len(hdr3))
        for r in results:
            ub, ua = r["undef_mmlu_base"], r["undef_mmlu_arditi"]
            db, da = r["def_mmlu_base"],   r["def_mmlu_arditi"]
            d_undef = (ua - ub) if (ua is not None and ub is not None) else None
            d_def   = (da - db) if (da is not None and db is not None) else None
            print(
                f"{r['model']:<30} "
                f"{_f(ub):>10} "
                f"{_f(ua):>10} "
                f"{(_f(d_undef) if d_undef is not None else 'N/A'):>7} "
                f"{_f(db):>9} "
                f"{_f(da):>8} "
                f"{(_f(d_def) if d_def is not None else 'N/A'):>6}"
            )


if __name__ == "__main__":
    main()
