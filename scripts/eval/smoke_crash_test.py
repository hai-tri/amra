#!/usr/bin/env python3
"""
smoke_crash_test.py — Fast crash test for all defenses and attacks.

Loads one model, applies each defense, runs each attack for 1 step,
and reports pass/fail. No utility evals. Finishes in minutes, not hours.

Usage:
    python scripts/eval/smoke_crash_test.py --model llama
    python scripts/eval/smoke_crash_test.py --model all
"""

import argparse
import os
import sys
import tempfile
import time
import traceback

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
from obfuscation_config import ObfuscationConfig
from run_obfuscation_pipeline import filter_data, load_mlabonne_datasets
from quick_attack_test import (
    CONFIGS, SYSTEM_PROMPT, _QWEN3_TEMPLATE,
    _build_arditi_hooks, _build_pca_hooks, _extract_pca_directions,
    _save, _restore,
)

import functools


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


def _check(name, fn):
    t0 = time.time()
    try:
        fn()
        elapsed = time.time() - t0
        print(f"  [PASS] {name} ({elapsed:.1f}s)")
        return True
    except Exception:
        elapsed = time.time() - t0
        print(f"  [FAIL] {name} ({elapsed:.1f}s)")
        traceback.print_exc()
        return False


@torch.no_grad()
def run_model(model_key):
    model_id, default_eps, _ = CONFIGS[model_key]
    print(f"\n{'='*60}")
    print(f"Model: {model_key} ({model_id})")
    print(f"{'='*60}")

    results = {}

    # Load model
    print("\nLoading model...")
    model_base = construct_model_base(model_id)
    _setup_tokenizer(model_base, model_id)

    harmful_train, harmless_train, harmful_val, harmless_val = load_mlabonne_datasets(
        n_train=16, n_val=8,
    )
    harmful_train, harmless_train, harmful_val, harmless_val = filter_data(
        model_base, harmful_train, harmless_train, harmful_val, harmless_val,
    )
    harmful_test = harmful_val[:4]
    harmless_test = harmless_val[:4]

    # Direction extraction
    print("\nExtracting refusal direction...")
    with tempfile.TemporaryDirectory() as tmp:
        mean_diffs = generate_directions(
            model_base, harmful_train, harmless_train, artifact_dir=tmp,
        )
        pos, layer, direction = select_direction(
            model_base, harmful_val, harmless_val, mean_diffs, artifact_dir=tmp,
        )
    print(f"  direction: pos={pos}, layer={layer}")

    snap = _save(model_base.model)

    def _refusal(pre, post):
        scores = get_refusal_scores(
            model_base.model, harmful_test,
            model_base.tokenize_instructions_fn,
            model_base.refusal_toks,
            fwd_pre_hooks=pre, fwd_hooks=post, batch_size=4,
        )
        return scores.mean().item()

    # ── Defenses ──
    print("\n--- Defenses ---")

    def _test_aprs():
        _restore(model_base.model, snap)
        cfg = ObfuscationConfig(
            epsilon=default_eps, num_pertinent_layers=4,
            num_calibration_prompts=8, num_probe_prompts=8,
            seed=42, projection_mode="full", per_layer_direction=True,
            writer_output_directions=True, num_writer_directions=1,
            num_reader_directions=1, forward_batch_size=4,
        )
        apply_obfuscation(
            model=model_base.model,
            tokenize_fn=model_base.tokenize_instructions_fn,
            harmful_prompts=harmful_train,
            harmless_prompts=harmless_train,
            mean_diffs=mean_diffs,
            selected_pos=pos, selected_layer=layer,
            direction=direction, cfg=cfg,
        )
        _refusal([], [])
        _restore(model_base.model, snap)

    results["aprs"] = _check("APRS (full)", _test_aprs)

    def _test_surgical():
        _restore(model_base.model, snap)
        from defenses.apply_surgical import apply_surgical
        apply_surgical(
            model=model_base.model,
            tokenizer=model_base.tokenizer,
            tokenize_fn=model_base.tokenize_instructions_fn,
            block_modules=model_base.model_block_modules,
            harmful_prompts=harmful_train[:8],
            harmless_prompts=harmless_train[:8],
            batch_size=4,
        )
        _refusal([], [])
        _restore(model_base.model, snap)

    results["surgical"] = _check("Surgical", _test_surgical)

    def _test_cast():
        _restore(model_base.model, snap)
        from defenses.apply_cast import apply_cast
        apply_cast(
            model=model_base.model,
            tokenizer=model_base.tokenizer,
            tokenize_fn=model_base.tokenize_instructions_fn,
            block_modules=model_base.model_block_modules,
            harmful_prompts=harmful_train[:8],
            harmless_prompts=harmless_train[:8],
            batch_size=4,
        )
        _refusal([], [])
        _restore(model_base.model, snap)

    results["cast"] = _check("CAST", _test_cast)

    def _test_circuit_breakers():
        _restore(model_base.model, snap)
        from defenses.apply_circuit_breakers import apply_circuit_breakers
        with torch.enable_grad():
            apply_circuit_breakers(
                model=model_base.model,
                tokenizer=model_base.tokenizer,
                harmful_prompts=harmful_train[:8],
                harmless_prompts=harmless_train[:8],
                batch_size=1, max_steps=2,
            )
        _refusal([], [])
        _restore(model_base.model, snap)

    results["circuit_breakers"] = _check("Circuit Breakers", _test_circuit_breakers)

    def _test_alphasteer():
        _restore(model_base.model, snap)
        from defenses.apply_alphasteer import apply_alphasteer
        n_layers = len(model_base.model_block_modules)
        target = [n_layers // 2, n_layers // 2 + 1, n_layers // 2 + 2]
        apply_alphasteer(
            model=model_base.model,
            tokenizer=model_base.tokenizer,
            tokenize_fn=model_base.tokenize_instructions_fn,
            block_modules=model_base.model_block_modules,
            harmful_prompts=harmful_train[:8],
            harmless_prompts=harmless_train[:8],
            refusal_direction=direction,
            target_layers=target,
            batch_size=4,
        )
        _refusal([], [])
        _restore(model_base.model, snap)

    results["alphasteer"] = _check("AlphaSteer", _test_alphasteer)

    # ── Attacks ──
    print("\n--- Attacks ---")
    _restore(model_base.model, snap)

    def _test_arditi():
        from attacks.evaluate_abliteration import evaluate_abliteration_resistance
        evaluate_abliteration_resistance(
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

    results["arditi"] = _check("Arditi abliteration", _test_arditi)

    def _test_pca():
        dirs, _ = _extract_pca_directions(
            model_base.model, model_base.tokenizer,
            model_base.tokenize_instructions_fn,
            model_base.model_block_modules,
            harmful_test, harmless_test, top_k=8, batch_size=4,
        )
        pre, post = _build_pca_hooks(
            model_base.model_block_modules,
            model_base.model_attn_modules,
            model_base.model_mlp_modules, dirs,
        )
        _refusal(pre, post)

    results["pca8"] = _check("PCA-8", _test_pca)

    def _test_leace():
        from attacks.evaluate_leace_attack import leace_attack
        orig_n = model_base.model.config.num_hidden_layers
        model_base.model.config.num_hidden_layers = min(4, orig_n)
        try:
            leace_attack(
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
        finally:
            model_base.model.config.num_hidden_layers = orig_n

    results["leace"] = _check("LEACE", _test_leace)

    def _test_nonlinear_probe():
        from attacks.evaluate_nonlinear_probe import nonlinear_probe_attack
        orig_n = model_base.model.config.num_hidden_layers
        model_base.model.config.num_hidden_layers = min(4, orig_n)
        try:
            nonlinear_probe_attack(
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
                nlprobe_epochs=5,
            )
        finally:
            model_base.model.config.num_hidden_layers = orig_n

    results["nonlinear_probe"] = _check("Nonlinear Probe", _test_nonlinear_probe)

    def _test_gcg():
        from attacks.evaluate_gcg import evaluate_gcg
        with torch.enable_grad():
            evaluate_gcg(
                model=model_base.model,
                tokenizer=model_base.tokenizer,
                tokenize_fn=model_base.tokenize_instructions_fn,
                harmful_prompts=harmful_test,
                refusal_toks=model_base.refusal_toks,
                n_behaviors=1, num_steps=2, topk=16, batch_size=4,
            )

    results["gcg"] = _check("GCG (1 behavior, 2 steps)", _test_gcg)

    def _test_autodan():
        from attacks.evaluate_autodan import evaluate_autodan
        evaluate_autodan(
            model=model_base.model,
            tokenizer=model_base.tokenizer,
            tokenize_fn=model_base.tokenize_instructions_fn,
            harmful_prompts=harmful_test,
            refusal_toks=model_base.refusal_toks,
            n_behaviors=1, num_steps=2, population_size=4,
        )

    results["autodan"] = _check("AutoDAN (1 behavior, 2 steps)", _test_autodan)

    def _test_jailbroken():
        from attacks.evaluate_jailbroken import evaluate_jailbroken
        evaluate_jailbroken(
            model=model_base.model,
            tokenizer=model_base.tokenizer,
            harmful_prompts=harmful_test,
            n_behaviors=1, templates=["roleplay"],
        )

    results["jailbroken"] = _check("Jailbroken", _test_jailbroken)

    def _test_pair():
        from attacks.evaluate_pair import evaluate_pair
        evaluate_pair(
            model=model_base.model,
            tokenizer=model_base.tokenizer,
            tokenize_fn=model_base.tokenize_instructions_fn,
            harmful_prompts=harmful_test,
            refusal_toks=model_base.refusal_toks,
            n_behaviors=1, n_streams=1, n_iterations=1,
        )

    results["pair"] = _check("PAIR (1 behavior, 1 iter)", _test_pair)

    def _test_renellm():
        from attacks.evaluate_renellm import evaluate_renellm
        evaluate_renellm(
            model=model_base.model,
            tokenizer=model_base.tokenizer,
            harmful_prompts=harmful_test,
            n_rewrite_strategies=1, n_scenario_attempts=1,
        )

    results["renellm"] = _check("ReNeLLM (1 strategy, 1 attempt)", _test_renellm)

    def _test_softopt():
        from attacks.evaluate_softopt import run_softopt_evaluation, SoftOptConfig
        bench = os.path.join(REPO_DIR, "data", "harmbench_test_std.json")
        with tempfile.TemporaryDirectory() as tmp:
            with torch.enable_grad():
                run_softopt_evaluation(
                    model=model_base.model,
                    tokenizer=model_base.tokenizer,
                    benchmark_path=bench,
                    output_dir=tmp,
                    softopt_config=SoftOptConfig(num_steps=2),
                    limit=1,
                )

    results["softopt"] = _check("SoftOpt (1 behavior, 2 steps)", _test_softopt)

    # ── Summary ──
    print(f"\n{'='*60}")
    passed = sum(1 for v in results.values() if v)
    total = len(results)
    print(f"Results: {passed}/{total} passed")
    for name, ok in results.items():
        print(f"  {'PASS' if ok else 'FAIL'}  {name}")
    print(f"{'='*60}")

    del model_base
    if torch.cuda.is_available():
        torch.cuda.empty_cache()

    return results


def main():
    pa = argparse.ArgumentParser()
    pa.add_argument("--model", choices=list(CONFIGS.keys()) + ["all"], default="llama")
    args = pa.parse_args()

    models = list(CONFIGS.keys()) if args.model == "all" else [args.model]
    all_results = {}
    for m in models:
        all_results[m] = run_model(m)

    print(f"\n{'='*60}")
    print("OVERALL SUMMARY")
    print(f"{'='*60}")
    all_pass = True
    for model, results in all_results.items():
        passed = sum(1 for v in results.values() if v)
        total = len(results)
        status = "ALL PASS" if passed == total else f"{total - passed} FAILED"
        print(f"  {model}: {passed}/{total} — {status}")
        if passed != total:
            all_pass = False
    sys.exit(0 if all_pass else 1)


if __name__ == "__main__":
    main()
