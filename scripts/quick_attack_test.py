#!/usr/bin/env python3
"""
quick_attack_test.py — Local Mac smoke test for Arditi abliteration + PCA-8 attack.

Extracts the refusal direction on-the-fly, applies the ideal APRS config,
then measures:
  1. Arditi et al. difference-in-means abliteration (single best direction)
  2. PCA-8 multi-direction abliteration (top-8 principal components)

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
from pipeline.submodules.select_direction import select_direction
from apply_obfuscation import apply_obfuscation
from obfuscation_config import ObfuscationConfig
from run_obfuscation_pipeline import load_mlabonne_datasets
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
        n_train=128, n_val=50
    )
    # skip filter_data — saves ~2GB peak memory, fine for a quick attack test

    # Direction extraction
    print("\nExtracting refusal direction …")
    with tempfile.TemporaryDirectory() as _tmp:
        mean_diffs = generate_directions(
            model_base, harmful_train, harmless_train, artifact_dir=_tmp,
        )
        pos, layer, direction = select_direction(
            model_base, harmful_val, harmless_val, mean_diffs, artifact_dir=_tmp,
        )
    print(f"Direction: pos={pos}, layer={layer}, ||r||={direction.norm():.4f}")

    # Apply defense
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

        harmful_test  = harmful_val[:n_prompts]
        harmless_test = harmless_val[:n_prompts]

        # Attack 1: Arditi abliteration
        print("\n--- Arditi abliteration ---")
        abl_result = evaluate_abliteration_resistance(
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

        # Attack 2: PCA-8
        print("\n--- PCA-8 attack ---")
        pca_result = pca_multi_direction_attack(
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

        print(f"\n{'─'*50}")
        print(f" RESULTS — {model_tag}")
        print(f"{'─'*50}")
        print(f"  avg_cos_sim (defended vs original): "
              f"{abl_result['mean_cos_sim']:.4f}")
        print(f"  Baseline refusal score (defended):  "
              f"{abl_result['baseline_refusal_score']:.4f}")
        print(f"  Post Arditi abliteration:           "
              f"{abl_result['post_abliteration_refusal_score']:.4f}")
        print(f"  Post PCA-8 abliteration:            "
              f"{pca_result['post_attack_refusal_score']:.4f}")

        return {
            "model": model_tag,
            "avg_cos_sim": abl_result["mean_cos_sim"],
            "baseline_refusal": abl_result["baseline_refusal_score"],
            "arditi_post_refusal": abl_result["post_abliteration_refusal_score"],
            "pca8_post_refusal": pca_result["post_attack_refusal_score"],
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
        print(f"{'Model':<35} {'cos_sim':>8} {'baseline':>9} {'arditi':>8} {'pca-8':>8}")
        print(f"{'─'*35} {'─'*8} {'─'*9} {'─'*8} {'─'*8}")
        for r in results:
            print(f"{r['model']:<35} {r['avg_cos_sim']:>8.4f} "
                  f"{r['baseline_refusal']:>9.4f} "
                  f"{r['arditi_post_refusal']:>8.4f} "
                  f"{r['pca8_post_refusal']:>8.4f}")


if __name__ == "__main__":
    main()
