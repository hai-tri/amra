#!/usr/bin/env python3
"""
sweep_layers_best_eps.py — For each model's best ε (lowest avg_cos_sim from the
epsilon sweep), sweep over num_pertinent_layers to see how cos_sim and BPB
trade off with layer count.

Layer grids are model-specific because auto-selected counts differ dramatically:
  Llama  auto=2  → sweep [2, 5, 10, 15, 20]
  Qwen   auto=7  → sweep [7, 10, 15, 20, 25]
  Gemma  auto=33 → sweep [10, 20, 33, 38, 42]  (42 = all layers)
"""

import csv, functools, json, os, sys
import torch

REPO_DIR = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.insert(0, REPO_DIR)
sys.path.insert(0, os.path.join(REPO_DIR, "refusal_direction"))

from pipeline.model_utils.model_factory import construct_model_base
from pipeline.submodules.evaluate_loss import evaluate_loss
from apply_obfuscation import apply_obfuscation
from obfuscation_config import ObfuscationConfig
from run_obfuscation_pipeline import load_mlabonne_datasets, filter_data
from attacks.evaluate_abliteration import evaluate_abliteration_resistance

SYSTEM_PROMPT = "You are a helpful assistant."

# (model_id, epsilon, layer_grid)
# Epsilon chosen as the highest value where BPB degradation is still acceptable.
# Llama at ε=0.025 already covered by sweep_layers.py — skipped here.
CONFIGS = [
    (
        "Qwen/Qwen3-8B",
        0.05,           # Δpile=+0.009, near-zero utility cost
        [7, 10, 15, 20, 25],
    ),
    (
        "google/gemma-2-9b-it",
        0.01,           # Δpile=+0.035, only clean ε before BPB climbs
        [10, 20, 33, 38, 42],
    ),
]

FIELDNAMES = [
    "model", "epsilon", "forced_num_layers", "actual_num_layers", "pertinent_layers",
    "avg_cos_sim", "max_cos_sim",
    "pile_bpb_pre", "alpaca_bpb_pre", "pile_bpb", "alpaca_bpb",
]

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


def sweep_model(model_id, epsilon, layer_grid, artifact_base, output_dir):
    model_tag = os.path.basename(model_id).lower()
    csv_path = os.path.join(output_dir, f"sweep_layers_besteps_{model_tag}.csv")

    # Check already-completed (forced_num_layers, epsilon) pairs
    completed = set()
    if os.path.exists(csv_path):
        with open(csv_path, newline="") as f:
            for row in csv.DictReader(f):
                try:
                    completed.add(int(row["forced_num_layers"]))
                except (KeyError, ValueError):
                    pass
    remaining = [n for n in layer_grid if n not in completed]
    if not remaining:
        print(f"[{model_tag}] All layers already complete — skipping.")
        return

    print(f"\n{'='*60}")
    print(f" Model   : {model_id}")
    print(f" ε       : {epsilon}")
    print(f" Layers  : {remaining}")
    print(f"{'='*60}\n")

    # Load model
    model_base = construct_model_base(model_id)
    _is_gemma = "gemma" in model_id.lower()
    _is_qwen3 = "qwen3" in model_id.lower()

    if _is_qwen3:
        _QWEN3_TEMPLATE = (
            "<|im_start|>user\n{instruction}<|im_end|>\n"
            "<|im_start|>assistant\n<think>\n\n</think>\n\n"
        )
        _tok = model_base.tokenizer
        def _qwen3_tokenize(instructions, outputs=None, system=None):
            prompts = [_QWEN3_TEMPLATE.format(instruction=i) for i in instructions]
            if outputs is not None:
                prompts = [p + o for p, o in zip(prompts, outputs)]
            return _tok(prompts, padding=True, truncation=False, return_tensors="pt")
        model_base.tokenize_instructions_fn = functools.partial(_qwen3_tokenize, system=SYSTEM_PROMPT)
        print("[config] Qwen3 thinking mode disabled.")
    elif not _is_gemma:
        model_base.tokenize_instructions_fn = functools.partial(
            model_base.tokenize_instructions_fn, system=SYSTEM_PROMPT
        )

    harmful_train, harmless_train, harmful_val, harmless_val = load_mlabonne_datasets(
        n_train=400, n_val=100
    )
    harmful_train, harmless_train, harmful_val, harmless_val = filter_data(
        model_base, harmful_train, harmless_train, harmful_val, harmless_val
    )

    # Load cached direction artifacts
    artifact_dir = os.path.join(artifact_base, model_tag)
    mean_diffs = torch.load(
        os.path.join(artifact_dir, "generate_directions", "mean_diffs.pt"), map_location="cpu"
    )
    direction = torch.load(os.path.join(artifact_dir, "direction.pt"), map_location="cpu")
    with open(os.path.join(artifact_dir, "direction_metadata.json")) as f:
        meta = json.load(f)
    pos, layer = meta["pos"], meta["layer"]
    with open(os.path.join(artifact_dir, "select_direction", "direction_evaluations.json")) as f:
        ablation_scores = json.load(f)
    original_direction = direction.clone()
    print(f"Direction loaded (pos={pos}, layer={layer})")

    # Undefended BPB once
    print("Computing undefended BPB …")
    pre = evaluate_loss(model_base, fwd_pre_hooks=[], fwd_hooks=[],
                        batch_size=4, n_batches=64,
                        dataset_labels=["pile", "alpaca"], completions_file_path=None)
    pile_pre = pre["pile"]["bpb"]
    alpaca_pre = pre["alpaca"]["bpb"]
    print(f"  Pile {pile_pre:.4f}  Alpaca {alpaca_pre:.4f}")

    file_exists = os.path.isfile(csv_path)
    with open(csv_path, "a", newline="") as fout:
        writer = csv.DictWriter(fout, fieldnames=FIELDNAMES)
        if not file_exists:
            writer.writeheader()

        for n in remaining:
            print(f"\n{'─'*50}\n forced_layers={n}, ε={epsilon}\n{'─'*50}")
            cfg = ObfuscationConfig(
                epsilon=epsilon,
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
                    "model":              model_id,
                    "epsilon":            epsilon,
                    "forced_num_layers":  n,
                    "actual_num_layers":  len(pertinent),
                    "pertinent_layers":   str(sorted(pertinent)),
                    "avg_cos_sim":        f"{abl['mean_cos_sim']:.4f}",
                    "max_cos_sim":        f"{abl['max_cos_sim']:.4f}",
                    "pile_bpb_pre":       f"{pile_pre:.4f}",
                    "alpaca_bpb_pre":     f"{alpaca_pre:.4f}",
                    "pile_bpb":           f"{loss['pile']['bpb']:.4f}",
                    "alpaca_bpb":         f"{loss['alpaca']['bpb']:.4f}",
                }
                writer.writerow(row)
                fout.flush()
                print(f"  avg_cos_sim={abl['mean_cos_sim']:.4f}  "
                      f"pile_bpb={loss['pile']['bpb']:.4f} (Δ{loss['pile']['bpb']-pile_pre:+.4f})")

            except Exception as e:
                print(f"[WARN] n={n} failed: {e}")
                import traceback; traceback.print_exc()
            finally:
                _restore(model_base.model, snap)

    del model_base
    torch.cuda.empty_cache()


def main():
    import argparse
    pa = argparse.ArgumentParser()
    pa.add_argument("--output_dir", default=os.path.join(os.path.expanduser("~"), "aprs_sweep"))
    args = pa.parse_args()
    os.makedirs(args.output_dir, exist_ok=True)

    artifact_base = os.path.join(args.output_dir, "artifacts")

    for model_id, epsilon, layer_grid in CONFIGS:
        try:
            sweep_model(model_id, epsilon, layer_grid, artifact_base, args.output_dir)
        except Exception as e:
            print(f"\n[ERROR] {model_id}: {e}")
            import traceback; traceback.print_exc()
            print("[continuing to next model]")

    # Aggregate
    all_rows, all_keys = [], []
    for model_id, _, _ in CONFIGS:
        tag = os.path.basename(model_id).lower()
        p = os.path.join(args.output_dir, f"sweep_layers_besteps_{tag}.csv")
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
        out = os.path.join(args.output_dir, "sweep_layers_besteps_results.csv")
        with open(out, "w", newline="") as f:
            w = csv.DictWriter(f, fieldnames=all_keys, extrasaction="ignore", restval="")
            w.writeheader()
            w.writerows(all_rows)
        print(f"\nAggregated {len(all_rows)} rows → {out}")

    print("\nDone.")


if __name__ == "__main__":
    main()
