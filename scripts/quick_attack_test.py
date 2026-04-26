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
from apply_obfuscation import apply_obfuscation
from obfuscation_config import ObfuscationConfig
from run_obfuscation_pipeline import load_mlabonne_datasets, filter_data
from attacks.evaluate_abliteration import evaluate_abliteration_resistance
from attacks.evaluate_adaptive_attack import pca_multi_direction_attack

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

        print(f"\n{'='*60}")
        print(f" RESULTS — {model_tag}")
        print(f"{'='*60}")
        print(f"  {'':35s} {'score':>8}  {'rate':>6}")
        print(f"  {'─'*51}")
        print(f"  {'Undefended baseline':35s} "
              f"{baseline_mean_undefended:>8.4f}  "
              f"{baseline_rate_undefended:>6.2%}")
        print(f"  {'Undefended + Arditi abliteration':35s} "
              f"{undefended_arditi['arditi_refusal_score']:>8.4f}  "
              f"(↓ should drop)")
        print(f"  {'Undefended + PCA-8 abliteration':35s} "
              f"{undefended_pca['post_attack_refusal_score']:>8.4f}  "
              f"(↓ should drop)")
        print(f"  {'─'*51}")
        print(f"  {'Defended baseline':35s} "
              f"{baseline_mean_defended:>8.4f}  "
              f"{baseline_rate_defended:>6.2%}")
        print(f"  {'Defended + Arditi abliteration':35s} "
              f"{defended_arditi['arditi_refusal_score']:>8.4f}  "
              f"(↑ should stay high)")
        print(f"  {'Defended + PCA-8 abliteration':35s} "
              f"{defended_pca['post_attack_refusal_score']:>8.4f}  "
              f"(↑ should stay high)")
        print(f"  avg_cos_sim (defended vs original dir): "
              f"{defended_arditi['mean_cos_sim']:.4f}  (↓ should be low)")

        return {
            "model": model_tag,
            "undefended_baseline":    baseline_mean_undefended,
            "undefended_arditi":      undefended_arditi["arditi_refusal_score"],
            "undefended_pca8":        undefended_pca["post_attack_refusal_score"],
            "defended_baseline":      baseline_mean_defended,
            "defended_arditi":        defended_arditi["arditi_refusal_score"],
            "defended_pca8":          defended_pca["post_attack_refusal_score"],
            "avg_cos_sim":            defended_arditi["mean_cos_sim"],
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
        print(f"\n{'='*60}")
        print(" SUMMARY")
        print(f"{'='*60}")
        hdr = f"{'Model':<35} {'undef_base':>10} {'undef_ard':>10} {'undef_pca':>10} {'def_base':>9} {'def_ard':>8} {'def_pca':>8} {'cos_sim':>8}"
        print(hdr)
        print("─" * len(hdr))
        for r in results:
            print(
                f"{r['model']:<35} "
                f"{r['undefended_baseline']:>10.4f} "
                f"{r['undefended_arditi']:>10.4f} "
                f"{r['undefended_pca8']:>10.4f} "
                f"{r['defended_baseline']:>9.4f} "
                f"{r['defended_arditi']:>8.4f} "
                f"{r['defended_pca8']:>8.4f} "
                f"{r['avg_cos_sim']:>8.4f}"
            )


if __name__ == "__main__":
    main()
