#!/usr/bin/env python3
"""
sweep_layers.py — Sweep num_pertinent_layers for Llama-3-8B at fixed ε=0.025.

Runs num_pertinent_layers ∈ {2, 5, 10, 15} in a single model-load pass,
measuring avg_cos_sim and BPB at each setting.
"""

import csv, functools, json, os, sys
import torch

REPO_DIR = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.insert(0, REPO_DIR)
sys.path.insert(0, os.path.join(REPO_DIR, "refusal_direction"))

from pipeline.model_utils.model_factory import construct_model_base
from pipeline.submodules.generate_directions import generate_directions
from pipeline.submodules.select_direction import select_direction
from pipeline.submodules.evaluate_loss import evaluate_loss
from apply_obfuscation import apply_obfuscation
from obfuscation_config import ObfuscationConfig
from run_obfuscation_pipeline import load_mlabonne_datasets, filter_data
from attacks.evaluate_abliteration import evaluate_abliteration_resistance

MODEL_ID   = "meta-llama/Meta-Llama-3-8B-Instruct"
EPSILON    = 0.025
N_LAYERS   = [2, 5, 10, 15]
SYSTEM_PROMPT = "You are a helpful assistant."

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

def main():
    import argparse
    pa = argparse.ArgumentParser()
    pa.add_argument("--output_dir", default=os.path.join(os.path.expanduser("~"), "aprs_sweep"))
    args = pa.parse_args()
    os.makedirs(args.output_dir, exist_ok=True)
    csv_path = os.path.join(args.output_dir, "sweep_llama_layers.csv")

    # Load model
    model_base = construct_model_base(MODEL_ID)
    model_base.tokenize_instructions_fn = functools.partial(
        model_base.tokenize_instructions_fn, system=SYSTEM_PROMPT
    )

    harmful_train, harmless_train, harmful_val, harmless_val = load_mlabonne_datasets(
        n_train=400, n_val=100
    )
    harmful_train, harmless_train, harmful_val, harmless_val = filter_data(
        model_base, harmful_train, harmless_train, harmful_val, harmless_val
    )

    # Direction (reuse cached artifacts from epsilon sweep)
    artifact_dir = os.path.join(args.output_dir, "artifacts", "meta-llama-3-8b-instruct")
    mean_diffs = torch.load(os.path.join(artifact_dir, "generate_directions", "mean_diffs.pt"), map_location="cpu")
    direction   = torch.load(os.path.join(artifact_dir, "direction.pt"), map_location="cpu")
    with open(os.path.join(artifact_dir, "direction_metadata.json")) as f:
        meta = json.load(f)
    pos, layer = meta["pos"], meta["layer"]
    ablation_path = os.path.join(artifact_dir, "select_direction", "direction_evaluations.json")
    with open(ablation_path) as f:
        ablation_scores = json.load(f)
    original_direction = direction.clone()
    print(f"Direction loaded from cache (pos={pos}, layer={layer})")

    # Undefended BPB once
    print("\nComputing undefended BPB …")
    pre = evaluate_loss(model_base, fwd_pre_hooks=[], fwd_hooks=[],
                        batch_size=4, n_batches=64,
                        dataset_labels=["pile", "alpaca"], completions_file_path=None)
    pile_pre   = pre["pile"]["bpb"]
    alpaca_pre = pre["alpaca"]["bpb"]
    print(f"  Pile {pile_pre:.4f}  Alpaca {alpaca_pre:.4f}")

    fieldnames = ["epsilon", "forced_num_layers", "actual_num_layers", "pertinent_layers",
                  "avg_cos_sim", "max_cos_sim",
                  "pile_bpb_pre", "alpaca_bpb_pre", "pile_bpb", "alpaca_bpb"]

    file_exists = os.path.isfile(csv_path)
    fout = open(csv_path, "a", newline="")
    writer = csv.DictWriter(fout, fieldnames=fieldnames)
    if not file_exists:
        writer.writeheader()

    for n in N_LAYERS:
        print(f"\n{'─'*50}\n num_pertinent_layers = {n}, ε = {EPSILON}\n{'─'*50}")
        cfg = ObfuscationConfig(
            epsilon=EPSILON,
            num_pertinent_layers=n,
            num_calibration_prompts=128,
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
                ablation_scores=ablation_scores,
            )
            pertinent = obf["pertinent_layers"]

            abl = evaluate_abliteration_resistance(
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
                pertinent_layers=pertinent,
            )

            loss = evaluate_loss(model_base, fwd_pre_hooks=[], fwd_hooks=[],
                                 batch_size=4, n_batches=64,
                                 dataset_labels=["pile", "alpaca"],
                                 completions_file_path=None)

            row = {
                "epsilon": EPSILON,
                "forced_num_layers": n,
                "actual_num_layers": len(pertinent),
                "pertinent_layers": str(sorted(pertinent)),
                "avg_cos_sim":   f"{abl['mean_cos_sim']:.4f}",
                "max_cos_sim":   f"{abl['max_cos_sim']:.4f}",
                "pile_bpb_pre":  f"{pile_pre:.4f}",
                "alpaca_bpb_pre":f"{alpaca_pre:.4f}",
                "pile_bpb":      f"{loss['pile']['bpb']:.4f}",
                "alpaca_bpb":    f"{loss['alpaca']['bpb']:.4f}",
            }
            writer.writerow(row)
            fout.flush()
            print(f"  avg_cos_sim={abl['mean_cos_sim']:.4f}  "
                  f"pile_bpb={loss['pile']['bpb']:.4f} (Δ{loss['pile']['bpb']-pile_pre:+.4f})")
        except Exception as e:
            print(f"[WARN] n={n} failed: {e}")
        finally:
            _restore(model_base.model, snap)

    fout.close()
    print(f"\nDone → {csv_path}")

if __name__ == "__main__":
    main()
