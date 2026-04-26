#!/usr/bin/env python3
"""
run_utility_benchmarks.py — GSM8k / MATH500 / MMLU on the three best APRS configs.

Best configs identified from the epsilon + layer sweeps:
  Llama-3-8B-Instruct  ε=0.025  forced_layers=10
  Qwen3-8B             ε=0.05   forced_layers=7 (auto)
  Gemma-2-9B-it        ε=0.01   forced_layers=10

Direction artifacts are loaded from the existing sweep artifact cache
(~/aprs_sweep/artifacts/<model_tag>/) so no re-extraction is needed.

Results are written to ~/aprs_utility/utility_results.csv.
"""

import csv, functools, json, os, sys
import torch

REPO_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, REPO_DIR)
sys.path.insert(0, os.path.join(REPO_DIR, "refusal_direction"))

from pipeline.model_utils.model_factory import construct_model_base
from pipeline.submodules.evaluate_loss import evaluate_loss
from apply_obfuscation import apply_obfuscation
from obfuscation_config import ObfuscationConfig
from run_obfuscation_pipeline import load_mlabonne_datasets, filter_data
from benchmarks.evaluate_lm_harness import run_lm_harness

SYSTEM_PROMPT = "You are a helpful assistant."

# (model_id, epsilon, num_pertinent_layers or None for auto)
CONFIGS = [
    ("meta-llama/Meta-Llama-3-8B-Instruct", 0.025, 10),
    ("Qwen/Qwen3-8B",                        0.05,  None),  # auto = 7
    ("google/gemma-2-9b-it",                 0.01,  10),
]

LM_TASKS = ["gsm8k", "math500", "mmlu"]
LM_N_SAMPLES = 500
LM_BATCH_SIZE = "auto"  # overridden per-model where needed (see run_model)

FIELDNAMES = [
    "model", "epsilon", "forced_num_layers", "actual_num_layers",
    "pile_bpb_pre", "alpaca_bpb_pre", "pile_bpb", "alpaca_bpb",
    "gsm8k_pre", "gsm8k", "math500_pre", "math500", "mmlu_pre", "mmlu",
]

_SNAPSHOT_KEYS = frozenset(
    ["o_proj", "down_proj", "q_proj", "k_proj", "v_proj",
     "gate_proj", "up_proj", "lm_head"]
)

_QWEN3_TEMPLATE = (
    "<|im_start|>user\n{instruction}<|im_end|>\n"
    "<|im_start|>assistant\n<think>\n\n</think>\n\n"
)


def _save(model):
    return {n: p.data.clone() for n, p in model.named_parameters()
            if any(k in n for k in _SNAPSHOT_KEYS)}


def _restore(model, snap):
    with torch.no_grad():
        for n, p in model.named_parameters():
            if n in snap:
                p.data.copy_(snap[n])


def _extract_score(lm_result, task):
    r = lm_result.get(task, {})
    if task == "gsm8k":
        # flexible-extract tolerates chain-of-thought / format changes from defense
        return r.get("exact_match_flexible", r.get("exact_match"))
    if task == "math500":
        return r.get("exact_match")
    if task == "mmlu":
        return r.get("acc")
    return None


def _fmt(v):
    if v is None:
        return ""
    return f"{float(v):.4f}"


def run_model(model_id, epsilon, num_pertinent_layers, artifact_base, output_dir):
    model_tag = os.path.basename(model_id).lower()
    csv_path = os.path.join(output_dir, f"utility_{model_tag}.csv")

    if os.path.exists(csv_path):
        print(f"[{model_tag}] Already complete — skipping.")
        return

    print(f"\n{'='*60}")
    print(f" Model   : {model_id}")
    print(f" ε       : {epsilon}   layers: {num_pertinent_layers or 'auto'}")
    print(f"{'='*60}\n")

    model_base = construct_model_base(model_id)
    _is_qwen3  = "qwen3" in model_id.lower()
    _is_gemma  = "gemma" in model_id.lower()

    if _is_qwen3:
        _tok = model_base.tokenizer
        # Patch the tokenizer's chat_template to disable thinking mode so that
        # lm-eval (which calls apply_chat_template internally) doesn't emit
        # <think>...</think> tokens that break exact-match scoring.
        _no_think_template = (
            "{%- for message in messages %}"
            "{%- if message.role == 'user' %}"
            "<|im_start|>user\n{{ message.content }}<|im_end|>\n"
            "<|im_start|>assistant\n<think>\n\n</think>\n\n"
            "{%- endif %}"
            "{%- endfor %}"
        )
        _tok.chat_template = _no_think_template

        # Monkey-patch apply_chat_template so lm-eval always passes
        # enable_thinking=False, preventing thinking token generation.
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
        print("[config] Qwen3 thinking mode disabled (tokenizer + lm-eval template).")
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

    artifact_dir = os.path.join(artifact_base, model_tag)
    mean_diffs = torch.load(
        os.path.join(artifact_dir, "generate_directions", "mean_diffs.pt"),
        map_location="cpu",
    )
    direction = torch.load(
        os.path.join(artifact_dir, "direction.pt"), map_location="cpu"
    )
    with open(os.path.join(artifact_dir, "direction_metadata.json")) as f:
        meta = json.load(f)
    pos, layer = meta["pos"], meta["layer"]
    with open(os.path.join(artifact_dir, "select_direction", "direction_evaluations.json")) as f:
        ablation_scores = json.load(f)
    print(f"Direction loaded (pos={pos}, layer={layer})")

    # Pre-defense BPB
    print("Computing pre-defense BPB …")
    pre_bpb = evaluate_loss(model_base, fwd_pre_hooks=[], fwd_hooks=[],
                            batch_size=4, n_batches=64,
                            dataset_labels=["pile", "alpaca"],
                            completions_file_path=None)
    pile_pre   = pre_bpb["pile"]["bpb"]
    alpaca_pre = pre_bpb["alpaca"]["bpb"]
    print(f"  Pile {pile_pre:.4f}  Alpaca {alpaca_pre:.4f}")

    # Qwen3 with thinking disabled still generates longer outputs than Llama/Gemma;
    # use a fixed batch size to avoid auto-detection probing in thinking mode.
    lm_batch_size = 8 if _is_qwen3 else LM_BATCH_SIZE

    # Pre-defense lm-harness
    print("Running pre-defense lm-harness …")
    pre_lm = run_lm_harness(
        model=model_base.model,
        tokenizer=model_base.tokenizer,
        tasks=LM_TASKS,
        n_samples=LM_N_SAMPLES,
        batch_size=lm_batch_size,
        seed=42,
    )

    # Apply defense
    cfg = ObfuscationConfig(
        epsilon=epsilon,
        num_pertinent_layers=num_pertinent_layers,
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
        print(f"  Pertinent layers ({len(pertinent)}): {sorted(pertinent)}")

        # Post-defense BPB
        post_bpb = evaluate_loss(model_base, fwd_pre_hooks=[], fwd_hooks=[],
                                 batch_size=4, n_batches=64,
                                 dataset_labels=["pile", "alpaca"],
                                 completions_file_path=None)

        # Post-defense lm-harness
        print("Running post-defense lm-harness …")
        post_lm = run_lm_harness(
            model=model_base.model,
            tokenizer=model_base.tokenizer,
            tasks=LM_TASKS,
            n_samples=LM_N_SAMPLES,
            batch_size=lm_batch_size,
            seed=42,
        )

        row = {
            "model":              model_id,
            "epsilon":            epsilon,
            "forced_num_layers":  num_pertinent_layers or "auto",
            "actual_num_layers":  len(pertinent),
            "pile_bpb_pre":       _fmt(pile_pre),
            "alpaca_bpb_pre":     _fmt(alpaca_pre),
            "pile_bpb":           _fmt(post_bpb["pile"]["bpb"]),
            "alpaca_bpb":         _fmt(post_bpb["alpaca"]["bpb"]),
            "gsm8k_pre":          _fmt(_extract_score(pre_lm, "gsm8k")),
            "gsm8k":              _fmt(_extract_score(post_lm, "gsm8k")),
            "math500_pre":        _fmt(_extract_score(pre_lm, "math500")),
            "math500":            _fmt(_extract_score(post_lm, "math500")),
            "mmlu_pre":           _fmt(_extract_score(pre_lm, "mmlu")),
            "mmlu":               _fmt(_extract_score(post_lm, "mmlu")),
        }

        with open(csv_path, "w", newline="") as f:
            w = csv.DictWriter(f, fieldnames=FIELDNAMES)
            w.writeheader()
            w.writerow(row)
        print(f"\nSaved → {csv_path}")
        for k in ["gsm8k", "math500", "mmlu"]:
            print(f"  {k}: pre={row[k+'_pre']}  post={row[k]}")

    except Exception as e:
        print(f"[ERROR] {model_id}: {e}")
        import traceback; traceback.print_exc()
    finally:
        _restore(model_base.model, snap)

    del model_base
    torch.cuda.empty_cache()


def main():
    import argparse
    pa = argparse.ArgumentParser()
    pa.add_argument("--output_dir",   default=os.path.join(os.path.expanduser("~"), "aprs_utility"))
    pa.add_argument("--artifact_base", default=os.path.join(os.path.expanduser("~"), "aprs_sweep", "artifacts"))
    args = pa.parse_args()
    os.makedirs(args.output_dir, exist_ok=True)

    for model_id, epsilon, n_layers in CONFIGS:
        try:
            run_model(model_id, epsilon, n_layers, args.artifact_base, args.output_dir)
        except Exception as e:
            print(f"\n[ERROR] {model_id}: {e}")
            import traceback; traceback.print_exc()
            print("[continuing to next model]")

    # Aggregate
    all_rows, all_keys = [], []
    for model_id, _, _ in CONFIGS:
        tag = os.path.basename(model_id).lower()
        p = os.path.join(args.output_dir, f"utility_{tag}.csv")
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
        out = os.path.join(args.output_dir, "utility_results.csv")
        with open(out, "w", newline="") as f:
            w = csv.DictWriter(f, fieldnames=all_keys, extrasaction="ignore", restval="")
            w.writeheader()
            w.writerows(all_rows)
        print(f"\nAggregated {len(all_rows)} rows → {out}")

    print("\nDone.")


if __name__ == "__main__":
    main()
