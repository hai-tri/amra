#!/usr/bin/env python3
"""
sweep_epsilon.py — Single-process ε hyperparameter sweep for APRS.

Loads each model once, then iterates over all ε values in memory,
applying and reverting the obfuscation weight update between runs.
Saves ~2–3 min of model-load overhead per ε run (26 reloads eliminated).

Usage:
    python scripts/sweep_epsilon.py [--output_dir DIR] [--seed N]
                                    [--num_calibration_prompts N]
                                    [--ce_loss_n_batches N]
                                    [--lm_harness_n N]

Key metrics per row:
    epsilon               — ε value used
    num_pertinent_layers  — number of layers selected by ablation scoring
    avg_cos_sim           — mean |cos| between original and defended refusal
                            directions over pertinent layers (lower = better)
    max_cos_sim           — max |cos| across all layers
    pile_bpb_pre          — undefended bits-per-byte on Pile
    alpaca_bpb_pre        — undefended bits-per-byte on Alpaca
    pile_bpb              — defended bits-per-byte on Pile
    alpaca_bpb            — defended bits-per-byte on Alpaca
    gsm8k_pre             — undefended GSM8k exact-match (if --lm_harness_n > 0)
    gsm8k                 — defended GSM8k exact-match (if --lm_harness_n > 0)
"""

import argparse
import csv
import functools
import json
import os
import sys

import torch

REPO_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, REPO_DIR)
sys.path.insert(0, os.path.join(REPO_DIR, "refusal_direction"))

from pipeline.model_utils.model_factory import construct_model_base
from pipeline.submodules.generate_directions import generate_directions
from pipeline.submodules.select_direction import select_direction, get_refusal_scores
from pipeline.submodules.evaluate_loss import evaluate_loss

from apply_obfuscation import apply_obfuscation
from obfuscation_config import ObfuscationConfig
from run_obfuscation_pipeline import load_mlabonne_datasets, filter_data
from attacks.evaluate_abliteration import evaluate_abliteration_resistance

SYSTEM_PROMPT = "You are a helpful assistant."

MODELS = [
    "meta-llama/Meta-Llama-3-8B-Instruct",
    "Qwen/Qwen3-8B",
    "google/gemma-2-9b-it",
]

EPSILONS = [0.01, 0.025, 0.05, 0.1, 0.15, 0.2, 0.3, 0.5]

FIELDNAMES = [
    "model", "epsilon", "num_pertinent_layers", "pertinent_layers",
    "avg_cos_sim", "max_cos_sim",
    "pile_bpb_pre", "alpaca_bpb_pre",
    "pile_bpb", "alpaca_bpb",
    "gsm8k_pre", "gsm8k",
]


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--output_dir", default=os.path.join(os.path.expanduser("~"), "aprs_sweep"))
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--num_calibration_prompts", type=int, default=128)
    p.add_argument("--ce_loss_n_batches", type=int, default=64)
    p.add_argument("--ce_loss_batch_size", type=int, default=4)
    p.add_argument("--lm_harness_n", type=int, default=0,
                   help="GSM8k samples per run. 0 = skip lm-harness entirely.")
    p.add_argument("--models", nargs="+", default=MODELS)
    p.add_argument("--epsilons", nargs="+", type=float, default=EPSILONS)
    return p.parse_args()


# ---------------------------------------------------------------------------
# Weight snapshot helpers
# ---------------------------------------------------------------------------

_SNAPSHOT_KEYS = frozenset(
    ["o_proj", "down_proj", "q_proj", "k_proj", "v_proj",
     "gate_proj", "up_proj", "lm_head"]
)


def _save_snapshot(model: torch.nn.Module) -> dict:
    return {
        name: param.data.clone()
        for name, param in model.named_parameters()
        if any(k in name for k in _SNAPSHOT_KEYS)
    }


def _restore_snapshot(model: torch.nn.Module, snapshot: dict) -> None:
    with torch.no_grad():
        for name, param in model.named_parameters():
            if name in snapshot:
                param.data.copy_(snapshot[name])


# ---------------------------------------------------------------------------
# CSV helpers
# ---------------------------------------------------------------------------

def _load_completed_epsilons(csv_path: str) -> set:
    if not os.path.exists(csv_path):
        return set()
    completed = set()
    with open(csv_path, newline="") as f:
        for row in csv.DictReader(f):
            try:
                completed.add(float(row["epsilon"]))
            except (KeyError, ValueError):
                pass
    return completed


def _append_row(csv_path: str, row: dict) -> None:
    file_exists = os.path.isfile(csv_path)
    with open(csv_path, "a", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=FIELDNAMES, extrasaction="ignore")
        if not file_exists:
            writer.writeheader()
        writer.writerow(row)


# ---------------------------------------------------------------------------
# Per-model sweep
# ---------------------------------------------------------------------------

def sweep_model(model_id: str, args, output_dir: str) -> None:
    model_tag = os.path.basename(model_id).lower()
    csv_path = os.path.join(output_dir, f"sweep_{model_tag}.csv")

    completed = _load_completed_epsilons(csv_path)
    remaining = [e for e in args.epsilons if e not in completed]
    if not remaining:
        print(f"[{model_tag}] All ε already complete — skipping.")
        return

    print(f"\n{'='*60}")
    print(f" Model : {model_id}")
    print(f" ε remaining : {remaining}")
    print(f" CSV : {csv_path}")
    print(f"{'='*60}\n")

    # ------------------------------------------------------------------
    # Load model + datasets
    # ------------------------------------------------------------------
    model_base = construct_model_base(model_id)
    _is_gemma = "gemma" in model_id.lower()
    _is_qwen3 = "qwen3" in model_id.lower()

    if _is_qwen3:
        # Qwen3 generates <think> tokens before responding, so refusal tokens
        # never appear at the expected output position during direction selection.
        # Inject an empty <think></think> block to disable thinking mode.
        _QWEN3_TEMPLATE = (
            "<|im_start|>user\n{instruction}<|im_end|>\n"
            "<|im_start|>assistant\n<think>\n\n</think>\n\n"
        )
        _QWEN3_TEMPLATE_SYS = (
            "<|im_start|>system\n{system_prompt}<|im_end|>\n"
            "<|im_start|>user\n{instruction}<|im_end|>\n"
            "<|im_start|>assistant\n<think>\n\n</think>\n\n"
        )
        _tok = model_base.tokenizer

        def _qwen3_tokenize(instructions, outputs=None, system=None):
            tmpl = _QWEN3_TEMPLATE_SYS if system else _QWEN3_TEMPLATE
            if outputs is not None:
                prompts = [
                    tmpl.format(instruction=i, system_prompt=system or "") + o
                    for i, o in zip(instructions, outputs)
                ]
            else:
                prompts = [
                    tmpl.format(instruction=i, system_prompt=system or "")
                    for i in instructions
                ]
            return _tok(prompts, padding=True, truncation=False, return_tensors="pt")

        model_base.tokenize_instructions_fn = functools.partial(
            _qwen3_tokenize, system=SYSTEM_PROMPT
        )
        print("[config] Qwen3 thinking mode disabled via empty <think></think> block.")
    elif not _is_gemma:
        model_base.tokenize_instructions_fn = functools.partial(
            model_base.tokenize_instructions_fn, system=SYSTEM_PROMPT
        )
        print(f"[config] System prompt injected.")
    else:
        print(f"[config] System prompt: (none — Gemma does not support system prompts)")

    harmful_train, harmless_train, harmful_val, harmless_val = load_mlabonne_datasets(
        n_train=400, n_val=100,
    )
    harmful_train, harmless_train, harmful_val, harmless_val = filter_data(
        model_base, harmful_train, harmless_train, harmful_val, harmless_val,
    )

    # ------------------------------------------------------------------
    # Direction extraction (cached per model, shared across all ε)
    # ------------------------------------------------------------------
    artifact_dir = os.path.join(output_dir, "artifacts", model_tag)
    gen_dir = os.path.join(artifact_dir, "generate_directions")
    mean_diffs_path = os.path.join(gen_dir, "mean_diffs.pt")
    direction_path = os.path.join(artifact_dir, "direction.pt")
    meta_path = os.path.join(artifact_dir, "direction_metadata.json")
    ablation_path = os.path.join(artifact_dir, "select_direction", "direction_evaluations.json")

    if (os.path.exists(direction_path) and os.path.exists(mean_diffs_path)
            and os.path.exists(meta_path)):
        print("Loading cached direction artifacts …")
        mean_diffs = torch.load(mean_diffs_path, map_location="cpu")
        direction = torch.load(direction_path, map_location="cpu")
        with open(meta_path) as f:
            meta = json.load(f)
        pos, layer = meta["pos"], meta["layer"]
    else:
        print("Extracting refusal direction (Stage 1) …")
        os.makedirs(gen_dir, exist_ok=True)
        mean_diffs = generate_directions(
            model_base, harmful_train, harmless_train, artifact_dir=gen_dir,
        )
        torch.save(mean_diffs, mean_diffs_path)

        print("Selecting best direction (Stage 2) …")
        sel_dir = os.path.join(artifact_dir, "select_direction")
        os.makedirs(sel_dir, exist_ok=True)
        pos, layer, direction = select_direction(
            model_base, harmful_val, harmless_val,
            mean_diffs, artifact_dir=sel_dir,
        )
        with open(meta_path, "w") as f:
            json.dump({"pos": pos, "layer": layer}, f)
        torch.save(direction, direction_path)

    ablation_scores = None
    if os.path.exists(ablation_path):
        with open(ablation_path) as f:
            ablation_scores = json.load(f)

    original_direction = direction.clone()

    # ------------------------------------------------------------------
    # Undefended BPB (computed once, reused across all ε rows)
    # ------------------------------------------------------------------
    print("\nComputing undefended BPB …")
    undefended_loss = evaluate_loss(
        model_base,
        fwd_pre_hooks=[], fwd_hooks=[],
        batch_size=args.ce_loss_batch_size,
        n_batches=args.ce_loss_n_batches,
        dataset_labels=["pile", "alpaca"],
        completions_file_path=None,
    )
    pile_bpb_pre = undefended_loss["pile"]["bpb"]
    alpaca_bpb_pre = undefended_loss["alpaca"]["bpb"]
    print(f"  Pile BPB (undefended):   {pile_bpb_pre:.4f}")
    print(f"  Alpaca BPB (undefended): {alpaca_bpb_pre:.4f}")

    # ------------------------------------------------------------------
    # Undefended GSM8k (once, if requested)
    # ------------------------------------------------------------------
    gsm8k_pre = ""
    if args.lm_harness_n > 0:
        try:
            from benchmarks.evaluate_lm_harness import run_lm_harness
            print(f"\nRunning undefended GSM8k (n={args.lm_harness_n}) …")
            lm_pre = run_lm_harness(
                model=model_base.model,
                tokenizer=model_base.tokenizer,
                fwd_pre_hooks=[], fwd_hooks=[],
                tasks=["gsm8k"],
                n_samples=args.lm_harness_n,
                output_dir=os.path.join(artifact_dir, "lm_harness_pre"),
                batch_size=4,
                seed=args.seed,
            )
            gsm8k_pre = f"{lm_pre['gsm8k']['exact_match']:.4f}" if lm_pre else ""
        except Exception as e:
            print(f"[WARN] undefended lm-harness failed: {e}")

    # ------------------------------------------------------------------
    # ε sweep — apply → measure → restore
    # ------------------------------------------------------------------
    for eps in remaining:
        print(f"\n{'─'*60}")
        print(f" ε = {eps}")
        print(f"{'─'*60}")

        obf_cfg = ObfuscationConfig(
            epsilon=eps,
            num_calibration_prompts=args.num_calibration_prompts,
            seed=args.seed,
            projection_mode="full",
            per_layer_direction=True,
            writer_output_directions=True,
        )

        # Save weights before any modification
        snapshot = _save_snapshot(model_base.model)

        try:
            # Apply defense (in-place weight update)
            obf_result = apply_obfuscation(
                model=model_base.model,
                tokenize_fn=model_base.tokenize_instructions_fn,
                harmful_prompts=harmful_train,
                harmless_prompts=harmless_train,
                mean_diffs=mean_diffs,
                selected_pos=pos,
                selected_layer=layer,
                direction=direction,
                cfg=obf_cfg,
                ablation_scores=ablation_scores,
            )
            pertinent_layers = obf_result["pertinent_layers"]

            # avg_cos_sim: cosine similarity between original and defended directions
            print(f"\n[ε={eps}] Evaluating abliteration resistance …")
            abl_result = evaluate_abliteration_resistance(
                model=model_base.model,
                tokenizer=model_base.tokenizer,
                tokenize_fn=model_base.tokenize_instructions_fn,
                block_modules=model_base.model_block_modules,
                attn_modules=model_base.model_attn_modules,
                mlp_modules=model_base.model_mlp_modules,
                harmful_prompts=harmful_val,
                benign_prompts=harmless_val,
                original_direction=original_direction,
                refusal_toks=model_base.refusal_toks,
                pertinent_layers=pertinent_layers,
            )
            avg_cos_sim = abl_result["mean_cos_sim"]
            max_cos_sim = abl_result["max_cos_sim"]

            # BPB on defended model
            print(f"[ε={eps}] Computing defended BPB …")
            defended_loss = evaluate_loss(
                model_base,
                fwd_pre_hooks=[], fwd_hooks=[],
                batch_size=args.ce_loss_batch_size,
                n_batches=args.ce_loss_n_batches,
                dataset_labels=["pile", "alpaca"],
                completions_file_path=None,
            )
            pile_bpb = defended_loss["pile"]["bpb"]
            alpaca_bpb = defended_loss["alpaca"]["bpb"]

            # GSM8k on defended model (optional)
            gsm8k = ""
            if args.lm_harness_n > 0:
                try:
                    from benchmarks.evaluate_lm_harness import run_lm_harness
                    print(f"[ε={eps}] Running defended GSM8k …")
                    lm_def = run_lm_harness(
                        model=model_base.model,
                        tokenizer=model_base.tokenizer,
                        fwd_pre_hooks=[], fwd_hooks=[],
                        tasks=["gsm8k"],
                        n_samples=args.lm_harness_n,
                        output_dir=os.path.join(
                            artifact_dir, f"lm_harness_eps{str(eps).replace('.', '_')}"
                        ),
                        batch_size=4,
                        seed=args.seed,
                    )
                    gsm8k = f"{lm_def['gsm8k']['exact_match']:.4f}" if lm_def else ""
                except Exception as e:
                    print(f"[WARN] defended lm-harness failed: {e}")

            row = {
                "model":                model_id,
                "epsilon":              eps,
                "num_pertinent_layers": len(pertinent_layers),
                "pertinent_layers":     str(sorted(pertinent_layers)),
                "avg_cos_sim":          f"{avg_cos_sim:.4f}",
                "max_cos_sim":          f"{max_cos_sim:.4f}",
                "pile_bpb_pre":         f"{pile_bpb_pre:.4f}",
                "alpaca_bpb_pre":       f"{alpaca_bpb_pre:.4f}",
                "pile_bpb":             f"{pile_bpb:.4f}",
                "alpaca_bpb":           f"{alpaca_bpb:.4f}",
                "gsm8k_pre":            gsm8k_pre,
                "gsm8k":                gsm8k,
            }
            _append_row(csv_path, row)
            print(f"\n[ε={eps}] Done → avg_cos_sim={avg_cos_sim:.4f}, "
                  f"pile_bpb={pile_bpb:.4f} (pre={pile_bpb_pre:.4f})")

        except Exception as e:
            print(f"[WARN] ε={eps} failed: {e}")

        finally:
            # Always restore weights before the next ε
            _restore_snapshot(model_base.model, snapshot)
            print(f"[ε={eps}] Weights restored.")

    # Free GPU memory before loading next model
    del model_base
    torch.cuda.empty_cache()


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    args = parse_args()
    os.makedirs(args.output_dir, exist_ok=True)

    print("================================================================")
    print(f" APRS ε Sweep (single-pass)")
    print(f" Models  : {args.models}")
    print(f" Epsilons: {args.epsilons}")
    print(f" Output  : {args.output_dir}")
    print("================================================================")

    for model_id in args.models:
        try:
            sweep_model(model_id, args, args.output_dir)
        except Exception as e:
            print(f"\n[ERROR] {model_id} failed: {e}")
            import traceback; traceback.print_exc()
            print("[continuing to next model]")

    # Aggregate all per-model CSVs into one summary
    model_tags = [os.path.basename(m).lower() for m in args.models]
    all_rows, all_keys = [], []
    for tag in model_tags:
        p = os.path.join(args.output_dir, f"sweep_{tag}.csv")
        if not os.path.exists(p):
            continue
        with open(p, newline="") as f:
            rows = list(csv.DictReader(f))
        if not rows:
            continue
        for k in rows[0]:
            if k not in all_keys:
                all_keys.append(k)
        all_rows.extend(rows)

    if all_rows:
        out = os.path.join(args.output_dir, "sweep_results.csv")
        with open(out, "w", newline="") as f:
            w = csv.DictWriter(f, fieldnames=all_keys, extrasaction="ignore", restval="")
            w.writeheader()
            w.writerows(all_rows)
        print(f"\nAggregated {len(all_rows)} rows → {out}")
    else:
        print("\n[WARN] No result CSVs found.")

    print("\n================================================================")
    print(" Sweep complete.")
    print("================================================================")


if __name__ == "__main__":
    main()
