#!/usr/bin/env python3
"""
quick_attack_test.py — Attack resistance + utility evaluation at best APRS configs.

Correct threat model:
  1. Extract refusal direction from *undefended* model.
  2. Filter to harmful prompts the base model actually refuses (score > 0).
  3. Measure refusal + utility on all 6 conditions:
       Undefended baseline | + Arditi | + PCA-8
       Defended baseline   | + Arditi | + PCA-8

Metrics per condition: refusal score, Pile BPB, MMLU, MATH500.

Usage:
    python scripts/eval/quick_attack_test.py [--model llama|qwen|gemma|all] [--n 20]
    python scripts/eval/quick_attack_test.py --skip_utility          # ASR only, fast
    python scripts/eval/quick_attack_test.py --bpb_batches 64 --mmlu_n 500 --math500_n 500
"""

import argparse
import csv
import datetime
import functools
import math
import os
import sys
import tempfile
import torch

REPO_DIR = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.insert(0, REPO_DIR)
sys.path.insert(0, os.path.join(REPO_DIR, "refusal_direction"))

from pipeline.model_utils.model_factory import construct_model_base
from pipeline.submodules.generate_directions import generate_directions, get_mean_activations
from pipeline.submodules.select_direction import select_direction, get_refusal_scores
from pipeline.submodules.evaluate_loss import evaluate_loss
from pipeline.utils.hook_utils import (
    add_hooks,
    get_direction_ablation_input_pre_hook,
    get_direction_ablation_output_hook,
)
from apply_obfuscation import apply_obfuscation
from obfuscation_config import ObfuscationConfig
from run_obfuscation_pipeline import load_mlabonne_datasets, filter_data
from attacks.evaluate_abliteration import evaluate_abliteration_resistance
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

CSV_FIELDS = [
    "model", "epsilon", "num_layers", "avg_cos_sim",
    # refusal scores
    "ref_undef_base", "ref_undef_arditi", "ref_undef_pca8",
    "ref_def_base",   "ref_def_arditi",   "ref_def_pca8",
    # pile bpb
    "bpb_undef_base", "bpb_undef_arditi", "bpb_undef_pca8",
    "bpb_def_base",   "bpb_def_arditi",   "bpb_def_pca8",
    # mmlu
    "mmlu_undef_base", "mmlu_undef_arditi", "mmlu_undef_pca8",
    "mmlu_def_base",   "mmlu_def_arditi",   "mmlu_def_pca8",
    # math500
    "math500_undef_base", "math500_undef_arditi", "math500_undef_pca8",
    "math500_def_base",   "math500_def_arditi",   "math500_def_pca8",
]


# ──────────────────────────────────────────────────────────────────────────────
# Helpers
# ──────────────────────────────────────────────────────────────────────────────

def _save(model):
    return {n: p.data.clone() for n, p in model.named_parameters()
            if any(k in n for k in _SNAPSHOT_KEYS)}


def _restore(model, snap):
    with torch.no_grad():
        for n, p in model.named_parameters():
            if n in snap:
                p.data.copy_(snap[n])


def _fmt(v):
    return f"{v:.4f}" if v is not None else "N/A"


def _refusal_rate(scores):
    return (scores > 0).float().mean().item()


def _build_arditi_hooks(block_modules, attn_modules, mlp_modules, direction):
    num_layers = len(block_modules)
    fwd_pre_hooks = [
        (block_modules[ell], get_direction_ablation_input_pre_hook(direction=direction))
        for ell in range(num_layers)
    ]
    fwd_hooks = (
        [(attn_modules[ell], get_direction_ablation_output_hook(direction=direction))
         for ell in range(num_layers)] +
        [(mlp_modules[ell],  get_direction_ablation_output_hook(direction=direction))
         for ell in range(num_layers)]
    )
    return fwd_pre_hooks, fwd_hooks


def _extract_pca_directions(
    model, tokenizer, tokenize_fn, block_modules,
    harmful_prompts, benign_prompts,
    top_k=8, batch_size=4,
    base_fwd_pre_hooks=None, base_fwd_hooks=None,
):
    """Return (top_k_directions, singular_values) from diff-in-means PCA."""
    base_fwd_pre_hooks = base_fwd_pre_hooks or []
    base_fwd_hooks     = base_fwd_hooks     or []
    with add_hooks(base_fwd_pre_hooks, base_fwd_hooks):
        mean_harmful = get_mean_activations(
            model, tokenizer, harmful_prompts, tokenize_fn,
            block_modules, batch_size=batch_size, positions=[-1],
        )
        mean_benign = get_mean_activations(
            model, tokenizer, benign_prompts, tokenize_fn,
            block_modules, batch_size=batch_size, positions=[-1],
        )
    mean_diffs = (mean_harmful - mean_benign).squeeze(0)       # (n_layers, d_model)
    centered   = mean_diffs - mean_diffs.mean(dim=0, keepdim=True)
    _, S, Vh   = torch.linalg.svd(centered.float(), full_matrices=False)
    return Vh[:top_k], S[:top_k]


def _build_pca_hooks(block_modules, attn_modules, mlp_modules, top_k_directions):
    num_layers = len(block_modules)

    def _pre_hook(module, inp):
        x = inp[0] if isinstance(inp, tuple) else inp
        for d in top_k_directions:
            d = d.to(x) / (d.to(x).norm() + 1e-8)
            x = x - (x @ d).unsqueeze(-1) * d
        return (x, *inp[1:]) if isinstance(inp, tuple) else x

    def _out_hook(module, inp, out):
        x = out[0] if isinstance(out, tuple) else out
        for d in top_k_directions:
            d = d.to(x) / (d.to(x).norm() + 1e-8)
            x = x - (x @ d).unsqueeze(-1) * d
        return (x, *out[1:]) if isinstance(out, tuple) else x

    fwd_pre_hooks = [(block_modules[ell], _pre_hook) for ell in range(num_layers)]
    fwd_hooks = (
        [(attn_modules[ell], _out_hook) for ell in range(num_layers)] +
        [(mlp_modules[ell],  _out_hook) for ell in range(num_layers)]
    )
    return fwd_pre_hooks, fwd_hooks


def _measure_utility(model_base, fwd_pre_hooks, fwd_hooks,
                     n_batches_bpb=32, mmlu_n=200, math500_n=200,
                     batch_size=8):
    """Return dict with pile_bpb, mmlu_acc, math500_acc under given hooks."""
    bpb_res = evaluate_loss(
        model_base,
        fwd_pre_hooks=fwd_pre_hooks,
        fwd_hooks=fwd_hooks,
        batch_size=batch_size,
        n_batches=n_batches_bpb,
        dataset_labels=["pile"],
    )
    pile = bpb_res["pile"]
    pile_bpb = pile.get("bpb") or pile["ce_loss"] / math.log(2)

    lm_res = run_lm_harness(
        model=model_base.model,
        tokenizer=model_base.tokenizer,
        tasks=["mmlu", "math500"],
        n_samples=max(mmlu_n, math500_n),
        batch_size=batch_size,
        seed=42,
        fwd_pre_hooks=fwd_pre_hooks,
        fwd_hooks=fwd_hooks,
    )
    mmlu_acc    = lm_res.get("mmlu",    {}).get("acc")
    math500_acc = lm_res.get("math500", {}).get("exact_match")

    return {"bpb": pile_bpb, "mmlu": mmlu_acc, "math500": math500_acc}


# ──────────────────────────────────────────────────────────────────────────────
# Main evaluation loop
# ──────────────────────────────────────────────────────────────────────────────

def run(model_key: str, n_prompts: int, skip_utility: bool,
        bpb_batches: int, mmlu_n: int, math500_n: int, batch_size: int):

    model_id, epsilon, num_layers = CONFIGS[model_key]
    model_tag = os.path.basename(model_id).lower()

    print(f"\n{'='*60}")
    print(f" {model_id}  ε={epsilon}  layers={num_layers}")
    print(f"{'='*60}")

    model_base = construct_model_base(model_id)
    _is_qwen3  = "qwen3" in model_id.lower()
    _is_gemma  = "gemma" in model_id.lower()

    if _is_qwen3:
        _tok = model_base.tokenizer
        _orig = _tok.apply_chat_template
        def _no_think(messages, **kw):
            kw.setdefault("enable_thinking", False)
            return _orig(messages, **kw)
        _tok.apply_chat_template = _no_think
        def _qwen3_tok(instructions, outputs=None, system=None):
            prompts = [_QWEN3_TEMPLATE.format(instruction=i) for i in instructions]
            if outputs is not None:
                prompts = [p + o for p, o in zip(prompts, outputs)]
            return _tok(prompts, padding=True, truncation=False, return_tensors="pt")
        model_base.tokenize_instructions_fn = functools.partial(_qwen3_tok, system=SYSTEM_PROMPT)
    elif not _is_gemma:
        model_base.tokenize_instructions_fn = functools.partial(
            model_base.tokenize_instructions_fn, system=SYSTEM_PROMPT)

    # ── Data ─────────────────────────────────────────────────────────────────
    harmful_train, harmless_train, harmful_val, harmless_val = load_mlabonne_datasets(
        n_train=400, n_val=100)
    print("\nFiltering with base model …")
    harmful_train, harmless_train, harmful_val, harmless_val = filter_data(
        model_base, harmful_train, harmless_train, harmful_val, harmless_val)
    print(f"  harmful_train={len(harmful_train)}  harmless_train={len(harmless_train)}")
    print(f"  harmful_val={len(harmful_val)}    harmless_val={len(harmless_val)}")

    # ── Direction extraction (undefended) ────────────────────────────────────
    print("\nExtracting refusal direction from undefended model …")
    with tempfile.TemporaryDirectory() as _tmp:
        mean_diffs_train = generate_directions(
            model_base, harmful_train, harmless_train, artifact_dir=_tmp)
        pos, layer, direction = select_direction(
            model_base, harmful_val, harmless_val, mean_diffs_train, artifact_dir=_tmp)
    print(f"Direction: pos={pos}, layer={layer}, ||r||={direction.norm():.4f}")

    harmful_test  = harmful_val[:n_prompts]
    harmless_test = harmless_val[:n_prompts]

    # ── Helper: refusal score under hooks ────────────────────────────────────
    def _refusal(pre, post):
        s = get_refusal_scores(
            model_base.model, harmful_test,
            model_base.tokenize_instructions_fn, model_base.refusal_toks,
            fwd_pre_hooks=pre, fwd_hooks=post, batch_size=4,
        )
        return s.mean().item()

    # ── Phase 1: UNDEFENDED ──────────────────────────────────────────────────
    print("\n" + "─"*50)
    print(" Phase 1: UNDEFENDED model")
    print("─"*50)

    ref_undef_base = _refusal([], [])
    print(f"  Baseline refusal score: {ref_undef_base:.4f}")

    # Arditi direction from undefended model
    undef_arditi_res = evaluate_abliteration_resistance(
        model=model_base.model, tokenizer=model_base.tokenizer,
        tokenize_fn=model_base.tokenize_instructions_fn,
        block_modules=model_base.model_block_modules,
        attn_modules=model_base.model_attn_modules,
        mlp_modules=model_base.model_mlp_modules,
        harmful_prompts=harmful_test, benign_prompts=harmless_test,
        original_direction=direction, refusal_toks=model_base.refusal_toks,
        batch_size=4,
    )
    ref_undef_arditi = undef_arditi_res["arditi_refusal_score"]
    ua_pre, ua_post = _build_arditi_hooks(
        model_base.model_block_modules, model_base.model_attn_modules,
        model_base.model_mlp_modules, undef_arditi_res["defended_direction"])

    # PCA-8 direction from undefended model
    print("  Extracting PCA-8 directions (undefended) …")
    undef_pca_dirs, undef_pca_svals = _extract_pca_directions(
        model_base.model, model_base.tokenizer,
        model_base.tokenize_instructions_fn, model_base.model_block_modules,
        harmful_test, harmless_test, top_k=8, batch_size=4,
    )
    print(f"  Top-8 singular values: {[f'{s:.1f}' for s in undef_pca_svals.tolist()]}")
    up_pre, up_post = _build_pca_hooks(
        model_base.model_block_modules, model_base.model_attn_modules,
        model_base.model_mlp_modules, undef_pca_dirs)
    ref_undef_pca8 = _refusal(up_pre, up_post)
    print(f"  Arditi post-abliteration: {ref_undef_arditi:.4f}")
    print(f"  PCA-8  post-abliteration: {ref_undef_pca8:.4f}")

    util_undef_base   = util_undef_arditi = util_undef_pca8 = None
    if not skip_utility:
        print("  Measuring utility — undefended baseline …")
        util_undef_base   = _measure_utility(model_base, [], [],
                                             bpb_batches, mmlu_n, math500_n, batch_size)
        print("  Measuring utility — undefended + Arditi …")
        util_undef_arditi = _measure_utility(model_base, ua_pre, ua_post,
                                             bpb_batches, mmlu_n, math500_n, batch_size)
        print("  Measuring utility — undefended + PCA-8 …")
        util_undef_pca8   = _measure_utility(model_base, up_pre, up_post,
                                             bpb_batches, mmlu_n, math500_n, batch_size)

    # ── Phase 2: DEFENDED ────────────────────────────────────────────────────
    cfg = ObfuscationConfig(
        epsilon=epsilon, num_pertinent_layers=num_layers,
        num_calibration_prompts=64, seed=42,
        projection_mode="full", per_layer_direction=True,
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
            harmful_prompts=harmful_train, harmless_prompts=harmless_train,
            mean_diffs=mean_diffs_train,
            selected_pos=pos, selected_layer=layer,
            direction=direction, cfg=cfg, ablation_scores=None,
        )
        pertinent = obf["pertinent_layers"]
        print(f"Pertinent layers ({len(pertinent)}): {sorted(pertinent)}")

        print("\n" + "─"*50)
        print(" Phase 2: DEFENDED model")
        print("─"*50)

        ref_def_base = _refusal([], [])
        print(f"  Baseline refusal score: {ref_def_base:.4f}")

        # Arditi direction from defended model
        def_arditi_res = evaluate_abliteration_resistance(
            model=model_base.model, tokenizer=model_base.tokenizer,
            tokenize_fn=model_base.tokenize_instructions_fn,
            block_modules=model_base.model_block_modules,
            attn_modules=model_base.model_attn_modules,
            mlp_modules=model_base.model_mlp_modules,
            harmful_prompts=harmful_test, benign_prompts=harmless_test,
            original_direction=direction, refusal_toks=model_base.refusal_toks,
            batch_size=4, pertinent_layers=list(pertinent),
        )
        ref_def_arditi = def_arditi_res["arditi_refusal_score"]
        avg_cos_sim    = def_arditi_res["mean_cos_sim"]
        da_pre, da_post = _build_arditi_hooks(
            model_base.model_block_modules, model_base.model_attn_modules,
            model_base.model_mlp_modules, def_arditi_res["defended_direction"])

        # PCA-8 direction from defended model
        print("  Extracting PCA-8 directions (defended) …")
        def_pca_dirs, def_pca_svals = _extract_pca_directions(
            model_base.model, model_base.tokenizer,
            model_base.tokenize_instructions_fn, model_base.model_block_modules,
            harmful_test, harmless_test, top_k=8, batch_size=4,
        )
        print(f"  Top-8 singular values: {[f'{s:.1f}' for s in def_pca_svals.tolist()]}")
        dp_pre, dp_post = _build_pca_hooks(
            model_base.model_block_modules, model_base.model_attn_modules,
            model_base.model_mlp_modules, def_pca_dirs)
        ref_def_pca8 = _refusal(dp_pre, dp_post)
        print(f"  Arditi post-abliteration: {ref_def_arditi:.4f}")
        print(f"  PCA-8  post-abliteration: {ref_def_pca8:.4f}")
        print(f"  avg_cos_sim: {avg_cos_sim:.4f}")

        util_def_base   = util_def_arditi = util_def_pca8 = None
        if not skip_utility:
            print("  Measuring utility — defended baseline …")
            util_def_base   = _measure_utility(model_base, [], [],
                                               bpb_batches, mmlu_n, math500_n, batch_size)
            print("  Measuring utility — defended + Arditi …")
            util_def_arditi = _measure_utility(model_base, da_pre, da_post,
                                               bpb_batches, mmlu_n, math500_n, batch_size)
            print("  Measuring utility — defended + PCA-8 …")
            util_def_pca8   = _measure_utility(model_base, dp_pre, dp_post,
                                               bpb_batches, mmlu_n, math500_n, batch_size)

        # ── Results table ────────────────────────────────────────────────────
        def _u(d, k):
            return d[k] if d is not None else None

        print(f"\n{'='*60}")
        print(f" RESULTS — {model_tag}")
        print(f"{'='*60}")
        print(f"  {'Condition':<35} {'ref_score':>9} {'pile_bpb':>9} {'mmlu':>7} {'math500':>8}")
        print(f"  {'─'*70}")
        rows_display = [
            ("Undefended baseline",       ref_undef_base,  util_undef_base),
            ("Undefended + Arditi",        ref_undef_arditi, util_undef_arditi),
            ("Undefended + PCA-8",         ref_undef_pca8,  util_undef_pca8),
            ("Defended baseline",          ref_def_base,    util_def_base),
            ("Defended + Arditi",          ref_def_arditi,  util_def_arditi),
            ("Defended + PCA-8",           ref_def_pca8,    util_def_pca8),
        ]
        for label, ref, util in rows_display:
            bpb = _fmt(_u(util, "bpb"))   if util else "—"
            mmlu = _fmt(_u(util, "mmlu")) if util else "—"
            m5   = _fmt(_u(util, "math500")) if util else "—"
            print(f"  {label:<35} {ref:>9.4f} {bpb:>9} {mmlu:>7} {m5:>8}")
        print(f"  avg_cos_sim (defended vs original): {avg_cos_sim:.4f}")

        return {
            "model": model_tag, "epsilon": epsilon, "num_layers": num_layers,
            "avg_cos_sim": avg_cos_sim,
            "ref_undef_base":   ref_undef_base,
            "ref_undef_arditi": ref_undef_arditi,
            "ref_undef_pca8":   ref_undef_pca8,
            "ref_def_base":     ref_def_base,
            "ref_def_arditi":   ref_def_arditi,
            "ref_def_pca8":     ref_def_pca8,
            "bpb_undef_base":   _u(util_undef_base,   "bpb"),
            "bpb_undef_arditi": _u(util_undef_arditi, "bpb"),
            "bpb_undef_pca8":   _u(util_undef_pca8,   "bpb"),
            "bpb_def_base":     _u(util_def_base,     "bpb"),
            "bpb_def_arditi":   _u(util_def_arditi,   "bpb"),
            "bpb_def_pca8":     _u(util_def_pca8,     "bpb"),
            "mmlu_undef_base":   _u(util_undef_base,   "mmlu"),
            "mmlu_undef_arditi": _u(util_undef_arditi, "mmlu"),
            "mmlu_undef_pca8":   _u(util_undef_pca8,   "mmlu"),
            "mmlu_def_base":     _u(util_def_base,     "mmlu"),
            "mmlu_def_arditi":   _u(util_def_arditi,   "mmlu"),
            "mmlu_def_pca8":     _u(util_def_pca8,     "mmlu"),
            "math500_undef_base":   _u(util_undef_base,   "math500"),
            "math500_undef_arditi": _u(util_undef_arditi, "math500"),
            "math500_undef_pca8":   _u(util_undef_pca8,   "math500"),
            "math500_def_base":     _u(util_def_base,     "math500"),
            "math500_def_arditi":   _u(util_def_arditi,   "math500"),
            "math500_def_pca8":     _u(util_def_pca8,     "math500"),
        }

    finally:
        _restore(model_base.model, snap)
        del model_base
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
        elif hasattr(torch.mps, "empty_cache"):
            torch.mps.empty_cache()


# ──────────────────────────────────────────────────────────────────────────────
# Entry point
# ──────────────────────────────────────────────────────────────────────────────

def main():
    pa = argparse.ArgumentParser()
    pa.add_argument("--model", choices=["llama", "qwen", "gemma", "all"], default="all")
    pa.add_argument("--n", type=int, default=20,
                    help="Harmful/harmless prompts for attack eval (default: 20)")
    pa.add_argument("--skip_utility", action="store_true",
                    help="Skip BPB/MMLU/MATH500 utility probes")
    pa.add_argument("--bpb_batches", type=int, default=32,
                    help="Pile BPB n_batches (default: 32)")
    pa.add_argument("--mmlu_n", type=int, default=200,
                    help="MMLU examples (default: 200)")
    pa.add_argument("--math500_n", type=int, default=200,
                    help="MATH500 examples (default: 200)")
    pa.add_argument("--batch_size", type=int, default=8,
                    help="Batch size for utility probes (default: 8)")
    pa.add_argument("--output_dir", default=os.path.join(REPO_DIR, "results", "attack_utility"),
                    help="Directory for CSV output")
    args = pa.parse_args()

    keys = list(CONFIGS.keys()) if args.model == "all" else [args.model]
    results = []
    for k in keys:
        try:
            r = run(k, args.n,
                    skip_utility=args.skip_utility,
                    bpb_batches=args.bpb_batches,
                    mmlu_n=args.mmlu_n,
                    math500_n=args.math500_n,
                    batch_size=args.batch_size)
            if r:
                results.append(r)
        except Exception as e:
            print(f"[ERROR] {k}: {e}")
            import traceback; traceback.print_exc()

    if not results:
        return

    # ── Summary tables ───────────────────────────────────────────────────────
    conditions = [
        ("undef_base",   "Undef baseline"),
        ("undef_arditi", "Undef+Arditi"),
        ("undef_pca8",   "Undef+PCA-8"),
        ("def_base",     "Def baseline"),
        ("def_arditi",   "Def+Arditi"),
        ("def_pca8",     "Def+PCA-8"),
    ]

    for metric, label, better in [
        ("ref",     "Refusal score",         None),
        ("bpb",     "Pile BPB",              "lower"),
        ("mmlu",    "MMLU accuracy",         "higher"),
        ("math500", "MATH500 exact match",   "higher"),
    ]:
        print(f"\n{'='*60}")
        print(f" {label}" + (f" ({better} = better)" if better else ""))
        print(f"{'='*60}")
        cond_keys = [c[0] for c in conditions]
        cond_labels = [c[1] for c in conditions]
        hdr = f"{'Model':<28}" + "".join(f"{l:>12}" for l in cond_labels)
        print(hdr)
        print("─" * len(hdr))
        for r in results:
            row = f"{r['model']:<28}"
            for ck in cond_keys:
                v = r.get(f"{metric}_{ck}")
                row += f"{_fmt(v):>12}"
            print(row)

    # ── CSV output ───────────────────────────────────────────────────────────
    if not args.skip_utility:
        os.makedirs(args.output_dir, exist_ok=True)
        ts = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
        csv_path = os.path.join(args.output_dir, f"attack_utility_{ts}.csv")
        with open(csv_path, "w", newline="") as f:
            w = csv.DictWriter(f, fieldnames=CSV_FIELDS, extrasaction="ignore")
            w.writeheader()
            w.writerows(results)
        print(f"\nSaved → {csv_path}")


if __name__ == "__main__":
    main()
