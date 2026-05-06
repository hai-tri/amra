#!/usr/bin/env python3
"""
Utility-only evaluator for APRS and defense baselines.

Runs one row per (model, defense) and writes a flat CSV with:
  - Pile BPB
  - Alpaca BPB
  - GSM8K exact match
  - MATH500 exact match
  - MMLU accuracy
  - optional AlpacaEval win rates

Each defense row loads a fresh model process-locally so weight-editing defenses
cannot leak into subsequent rows. Hook-based defenses pass their hooks into the
utility evaluators.
"""

from __future__ import annotations

import argparse
import csv
import datetime as _dt
import functools
import json
import math
import os
import sys
import tempfile
import time
import traceback
from pathlib import Path
from typing import Any

os.environ.setdefault("MPLCONFIGDIR", "/tmp/mplconfig")
os.environ.setdefault("FONTCONFIG_PATH", "/tmp/fontconfig")
Path(os.environ["MPLCONFIGDIR"]).mkdir(parents=True, exist_ok=True)
Path(os.environ["FONTCONFIG_PATH"]).mkdir(parents=True, exist_ok=True)

import torch

torch.backends.cuda.matmul.allow_tf32 = True
torch.backends.cudnn.allow_tf32 = True
torch.backends.cudnn.benchmark = True
torch.set_float32_matmul_precision("high")

REPO_DIR = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO_DIR))
sys.path.insert(0, str(REPO_DIR / "refusal_direction"))

from apply_obfuscation import apply_obfuscation
from benchmarks.evaluate_lm_harness import run_lm_harness
from defenses.apply_cast import apply_cast
from defenses.apply_circuit_breakers import apply_circuit_breakers
from defenses.apply_surgical import apply_surgical
from obfuscation_config import ObfuscationConfig
from pipeline.model_utils.model_factory import construct_model_base
from pipeline.submodules.evaluate_loss import evaluate_loss
from pipeline.submodules.generate_directions import generate_directions
from pipeline.submodules.select_direction import select_direction
from run_obfuscation_pipeline import filter_data, load_mlabonne_datasets


SYSTEM_PROMPT = "You are a helpful assistant."

QWEN3_TEMPLATE = (
    "<|im_start|>user\n{instruction}<|im_end|>\n"
    "<|im_start|>assistant\n<think>\n\n</think>\n\n"
)

MODEL_ALIASES = {
    "llama": "meta-llama/Meta-Llama-3-8B-Instruct",
    "qwen": "Qwen/Qwen3-8B",
    "gemma": "google/gemma-2-9b-it",
}

APRS_OPTIMA = {
    "llama": {"epsilon": 0.025, "num_layers": 20, "k_w": 1, "k_r": 8},
    "qwen": {"epsilon": 0.025, "num_layers": 20, "k_w": 1, "k_r": 8},
    "gemma": {"epsilon": 0.025, "num_layers": 30, "k_w": 4, "k_r": 16},
}

DEFENSES = ("none", "aprs", "surgical", "cast", "circuit_breakers", "alphasteer")

CSV_FIELDS = [
    "timestamp",
    "model",
    "model_key",
    "defense",
    "epsilon",
    "num_layers",
    "k_w",
    "k_r",
    "pile_bpb",
    "alpaca_bpb",
    "gsm8k_exact_match",
    "math500_exact_match",
    "mmlu_acc",
    "alpacaeval_win_rate",
    "alpacaeval_lc_win_rate",
    "alpacaeval_n",
    "alpacaeval_annotator",
    "lm_harness_n",
    "ce_loss_n_batches",
    "batch_size",
    "wall_minutes",
    "status",
    "error",
]


def _model_key(model_path: str) -> str:
    lower = model_path.lower()
    if "llama" in lower:
        return "llama"
    if "qwen" in lower:
        return "qwen"
    if "gemma" in lower:
        return "gemma"
    return Path(model_path).name.lower().replace("/", "_")


def _setup_tokenizer(model_base, model_path: str):
    lower = model_path.lower()
    if "qwen3" in lower:
        tok = model_base.tokenizer
        no_think_template = (
            "{%- for message in messages %}"
            "{%- if message.role == 'user' %}"
            "<|im_start|>user\n{{ message.content }}<|im_end|>\n"
            "<|im_start|>assistant\n<think>\n\n</think>\n\n"
            "{%- endif %}"
            "{%- endfor %}"
        )
        tok.chat_template = no_think_template
        orig_apply = tok.apply_chat_template

        def no_think_apply(messages, **kwargs):
            kwargs.setdefault("enable_thinking", False)
            return orig_apply(messages, **kwargs)

        tok.apply_chat_template = no_think_apply

        def qwen3_tokenize(instructions, outputs=None, system=None):
            prompts = [QWEN3_TEMPLATE.format(instruction=i) for i in instructions]
            if outputs is not None:
                prompts = [p + o for p, o in zip(prompts, outputs)]
            return tok(prompts, padding=True, truncation=False, return_tensors="pt")

        model_base.tokenize_instructions_fn = functools.partial(
            qwen3_tokenize,
            system=SYSTEM_PROMPT,
        )
        return
    if "gemma" in lower:
        return
    model_base.tokenize_instructions_fn = functools.partial(
        model_base.tokenize_instructions_fn,
        system=SYSTEM_PROMPT,
    )


def _load_and_filter_data(model_base, args):
    harmful_train, harmless_train, harmful_val, harmless_val = load_mlabonne_datasets(
        n_train=args.n_train,
        n_val=args.n_val,
        seed=args.seed,
    )
    if args.no_filter_data:
        return harmful_train, harmless_train, harmful_val, harmless_val
    return filter_data(model_base, harmful_train, harmless_train, harmful_val, harmless_val)


def _extract_direction(model_base, harmful_train, harmless_train, harmful_val, harmless_val):
    with tempfile.TemporaryDirectory() as tmp:
        mean_diffs = generate_directions(
            model_base,
            harmful_train,
            harmless_train,
            artifact_dir=tmp,
        )
        pos, layer, direction = select_direction(
            model_base,
            harmful_val,
            harmless_val,
            mean_diffs,
            artifact_dir=tmp,
        )
        scores_path = Path(tmp) / "direction_evaluations.json"
        ablation_scores = None
        if scores_path.exists():
            with scores_path.open() as f:
                ablation_scores = json.load(f)
    return mean_diffs, pos, layer, direction, ablation_scores


def _aprs_params_for(model_key: str, args) -> dict[str, Any]:
    params = dict(APRS_OPTIMA.get(model_key, APRS_OPTIMA["llama"]))
    if args.epsilon is not None:
        params["epsilon"] = args.epsilon
    if args.num_layers is not None:
        params["num_layers"] = args.num_layers
    if args.k_w is not None:
        params["k_w"] = args.k_w
    if args.k_r is not None:
        params["k_r"] = args.k_r
    return params


def _apply_defense(
    defense: str,
    model_base,
    model_key: str,
    harmful_train,
    harmless_train,
    harmful_val,
    harmless_val,
    artifact_dir: Path,
    args,
):
    fwd_pre_hooks: list = []
    fwd_hooks: list = []
    row_meta = {"epsilon": "", "num_layers": "", "k_w": "", "k_r": ""}

    if defense == "none":
        return fwd_pre_hooks, fwd_hooks, row_meta

    if defense in {"aprs", "alphasteer"}:
        mean_diffs, pos, layer, direction, ablation_scores = _extract_direction(
            model_base, harmful_train, harmless_train, harmful_val, harmless_val,
        )
    else:
        mean_diffs = pos = layer = direction = ablation_scores = None

    if defense == "aprs":
        params = _aprs_params_for(model_key, args)
        cfg = ObfuscationConfig(
            epsilon=params["epsilon"],
            num_pertinent_layers=params["num_layers"],
            num_calibration_prompts=args.num_calibration_prompts,
            seed=args.seed,
            projection_mode="full",
            per_layer_direction=True,
            writer_output_directions=True,
            num_writer_directions=params["k_w"],
            num_reader_directions=params["k_r"],
            forward_batch_size=args.forward_batch_size,
        )
        apply_obfuscation(
            model=model_base.model,
            tokenize_fn=model_base.tokenize_instructions_fn,
            harmful_prompts=harmful_train,
            harmless_prompts=harmless_train,
            mean_diffs=mean_diffs,
            selected_pos=pos,
            selected_layer=layer,
            direction=direction,
            cfg=cfg,
            ablation_scores=ablation_scores,
            writer_only=False,
        )
        row_meta.update({
            "epsilon": params["epsilon"],
            "num_layers": params["num_layers"],
            "k_w": params["k_w"],
            "k_r": params["k_r"],
        })
        return fwd_pre_hooks, fwd_hooks, row_meta

    if defense == "surgical":
        result = apply_surgical(
            model=model_base.model,
            tokenizer=model_base.tokenizer,
            tokenize_fn=model_base.tokenize_instructions_fn,
            block_modules=model_base.model_block_modules,
            harmful_prompts=harmful_train,
            harmless_prompts=harmless_train,
            ablation_coeff=args.surgical_ablation_coeff,
            actadd_coeff=args.surgical_actadd_coeff,
            apply_all_layers=True,
            artifact_dir=str(artifact_dir),
        )
        return result["fwd_pre_hooks"], result["fwd_hooks"], row_meta

    if defense == "cast":
        result = apply_cast(
            model=model_base.model,
            tokenizer=model_base.tokenizer,
            tokenize_fn=model_base.tokenize_instructions_fn,
            block_modules=model_base.model_block_modules,
            harmful_prompts=harmful_train,
            harmless_prompts=harmless_train,
            behavior_strength=args.cast_strength,
            condition_threshold=args.cast_threshold,
            preserve_norm=True,
            artifact_dir=str(artifact_dir),
        )
        return result["fwd_pre_hooks"], result["fwd_hooks"], row_meta

    if defense == "circuit_breakers":
        result = apply_circuit_breakers(
            model=model_base.model,
            tokenizer=model_base.tokenizer,
            harmful_prompts=harmful_train,
            harmless_prompts=harmless_train,
            lora_rank=args.cb_lora_rank,
            max_steps=args.cb_steps,
            lr=args.cb_lr,
            batch_size=args.cb_batch_size,
            cb_coeff_max=args.cb_coeff_max,
            retain_coeff_max=args.cb_retain_coeff_max,
            merge_weights=True,
            seed=args.seed,
            artifact_dir=str(artifact_dir),
        )
        model_base.model = result["model"]
        return fwd_pre_hooks, fwd_hooks, row_meta

    if defense == "alphasteer":
        from defenses.apply_alphasteer import apply_alphasteer

        result = apply_alphasteer(
            model=model_base.model,
            tokenizer=model_base.tokenizer,
            tokenize_fn=model_base.tokenize_instructions_fn,
            block_modules=model_base.model_block_modules,
            harmful_prompts=harmful_train,
            harmless_prompts=harmless_train,
            refusal_direction=direction,
            mean_diffs=mean_diffs,
            strength=args.alphasteer_strength,
            null_ratio=args.alphasteer_null_ratio,
            lambda_reg=args.alphasteer_lambda,
            batch_size=args.forward_batch_size,
            artifact_dir=str(artifact_dir),
        )
        return result["fwd_pre_hooks"], result["fwd_hooks"], row_meta

    raise ValueError(f"unknown defense: {defense}")


def _measure_utility(model_base, fwd_pre_hooks, fwd_hooks, artifact_dir: Path, args) -> dict[str, Any]:
    out = {
        "pile_bpb": "",
        "alpaca_bpb": "",
        "gsm8k_exact_match": "",
        "math500_exact_match": "",
        "mmlu_acc": "",
        "alpacaeval_win_rate": "",
        "alpacaeval_lc_win_rate": "",
        "alpacaeval_n": "",
        "alpacaeval_annotator": "",
    }

    labels = []
    if "pile" in args.bpb_datasets:
        labels.append("pile")
    if "alpaca" in args.bpb_datasets:
        labels.append("alpaca")
    if labels and args.ce_loss_n_batches > 0:
        loss_result = evaluate_loss(
            model_base,
            fwd_pre_hooks=fwd_pre_hooks,
            fwd_hooks=fwd_hooks,
            batch_size=args.ce_loss_batch_size,
            n_batches=args.ce_loss_n_batches,
            max_seq_length=args.ce_loss_max_seq_length,
            dataset_labels=labels,
        )
        if "pile" in loss_result:
            out["pile_bpb"] = loss_result["pile"]["bpb"]
        if "alpaca" in loss_result:
            out["alpaca_bpb"] = loss_result["alpaca"]["bpb"]

    tasks = [x.strip() for x in args.lm_harness_tasks.split(",") if x.strip()]
    if tasks and args.lm_harness_n > 0:
        lm_result = run_lm_harness(
            model=model_base.model,
            tokenizer=model_base.tokenizer,
            fwd_pre_hooks=fwd_pre_hooks,
            fwd_hooks=fwd_hooks,
            tasks=tasks,
            n_samples=args.lm_harness_n,
            output_dir=str(artifact_dir / "lm_harness"),
            batch_size=args.lm_harness_batch_size,
            seed=args.seed,
        )
        if "gsm8k" in lm_result:
            out["gsm8k_exact_match"] = lm_result["gsm8k"].get("exact_match", "")
        if "math500" in lm_result:
            out["math500_exact_match"] = lm_result["math500"].get("exact_match", "")
        if "mmlu" in lm_result:
            out["mmlu_acc"] = lm_result["mmlu"].get("acc", "")

    if args.alpacaeval_n > 0:
        from benchmarks.evaluate_alpacaeval import evaluate_alpacaeval

        alpaca_result = evaluate_alpacaeval(
            model=model_base.model,
            tokenizer=model_base.tokenizer,
            tokenize_fn=model_base.tokenize_instructions_fn,
            fwd_pre_hooks=fwd_pre_hooks,
            fwd_hooks=fwd_hooks,
            n_samples=args.alpacaeval_n,
            max_new_tokens=args.alpacaeval_max_new_tokens,
            batch_size=args.alpacaeval_batch_size,
            seed=args.seed,
            run_judge=not args.alpacaeval_skip_judge,
            annotators_config=args.alpacaeval_annotator,
            generator_name=f"{args.generator_prefix}",
            artifact_dir=str(artifact_dir / "alpacaeval"),
        )
        out["alpacaeval_win_rate"] = alpaca_result.get("win_rate")
        out["alpacaeval_lc_win_rate"] = alpaca_result.get("length_controlled_win_rate")
        out["alpacaeval_n"] = alpaca_result.get("n_samples", "")
        out["alpacaeval_annotator"] = alpaca_result.get("annotator") or ""

    return out


def _fmt_value(value):
    if value is None:
        return ""
    if isinstance(value, float):
        if math.isnan(value):
            return ""
        return f"{value:.6g}"
    return value


def _append_row(csv_path: Path, row: dict[str, Any]):
    csv_path.parent.mkdir(parents=True, exist_ok=True)
    exists = csv_path.exists()
    with csv_path.open("a", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=CSV_FIELDS)
        if not exists:
            writer.writeheader()
        writer.writerow({k: _fmt_value(row.get(k, "")) for k in CSV_FIELDS})


def run_one(model_path: str, defense: str, args) -> dict[str, Any]:
    model_key = _model_key(model_path)
    tag = f"{model_key}_{defense}"
    artifact_dir = Path(args.output_dir) / "artifacts" / tag
    artifact_dir.mkdir(parents=True, exist_ok=True)

    row = {
        "timestamp": _dt.datetime.now().isoformat(timespec="seconds"),
        "model": model_path,
        "model_key": model_key,
        "defense": defense,
        "lm_harness_n": args.lm_harness_n,
        "ce_loss_n_batches": args.ce_loss_n_batches,
        "batch_size": args.lm_harness_batch_size,
        "status": "ok",
        "error": "",
    }
    t0 = time.time()
    try:
        print("=" * 80)
        print(f"[utility] model={model_path} defense={defense}")
        print("=" * 80)
        model_base = construct_model_base(model_path)
        _setup_tokenizer(model_base, model_path)
        harmful_train, harmless_train, harmful_val, harmless_val = _load_and_filter_data(
            model_base,
            args,
        )
        fwd_pre_hooks, fwd_hooks, meta = _apply_defense(
            defense,
            model_base,
            model_key,
            harmful_train,
            harmless_train,
            harmful_val,
            harmless_val,
            artifact_dir,
            args,
        )
        row.update(meta)
        row.update(_measure_utility(model_base, fwd_pre_hooks, fwd_hooks, artifact_dir, args))
    except Exception as exc:
        row["status"] = "error"
        row["error"] = f"{type(exc).__name__}: {exc}"
        traceback.print_exc()
    finally:
        row["wall_minutes"] = (time.time() - t0) / 60
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
    return row


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--models", default="llama",
                        help="Comma-separated model aliases or HF paths. Aliases: llama,qwen,gemma")
    parser.add_argument("--defenses", default="none,aprs,surgical,cast,circuit_breakers,alphasteer",
                        help=f"Comma-separated defenses from: {','.join(DEFENSES)}")
    parser.add_argument("--output_dir", default=str(REPO_DIR / "results" / "defense_utility"))
    parser.add_argument("--save_csv", default=None)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--n_train", type=int, default=400)
    parser.add_argument("--n_val", type=int, default=100)
    parser.add_argument("--no_filter_data", action="store_true")

    parser.add_argument("--bpb_datasets", default="pile,alpaca",
                        help="Comma-separated BPB datasets: pile,alpaca. Empty disables BPB.")
    parser.add_argument("--ce_loss_n_batches", type=int, default=64)
    parser.add_argument("--ce_loss_batch_size", type=int, default=16)
    parser.add_argument("--ce_loss_max_seq_length", type=int, default=256)
    parser.add_argument("--lm_harness_tasks", default="gsm8k,math500,mmlu")
    parser.add_argument("--lm_harness_n", type=int, default=250)
    parser.add_argument("--lm_harness_batch_size", type=int, default=16)

    parser.add_argument("--alpacaeval_n", type=int, default=0,
                        help="0 disables AlpacaEval. Use 805 for full eval.")
    parser.add_argument("--alpacaeval_max_new_tokens", type=int, default=512)
    parser.add_argument("--alpacaeval_batch_size", type=int, default=32)
    parser.add_argument("--alpacaeval_skip_judge", action="store_true")
    parser.add_argument("--alpacaeval_annotator", default="alpaca_eval_gpt4o_fn")
    parser.add_argument("--generator_prefix", default="aprs_utility")

    parser.add_argument("--num_calibration_prompts", type=int, default=64)
    parser.add_argument("--forward_batch_size", type=int, default=16)
    parser.add_argument("--epsilon", type=float, default=None,
                        help="Override APRS epsilon for all APRS rows.")
    parser.add_argument("--num_layers", type=int, default=None,
                        help="Override APRS layer count for all APRS rows.")
    parser.add_argument("--k_w", type=int, default=None,
                        help="Override APRS writer rank for all APRS rows.")
    parser.add_argument("--k_r", type=int, default=None,
                        help="Override APRS reader rank for all APRS rows.")

    parser.add_argument("--surgical_ablation_coeff", type=float, default=1.0)
    parser.add_argument("--surgical_actadd_coeff", type=float, default=0.0)
    parser.add_argument("--cast_strength", type=float, default=1.5)
    parser.add_argument("--cast_threshold", type=float, default=0.02)
    parser.add_argument("--cb_lora_rank", type=int, default=16)
    parser.add_argument("--cb_steps", type=int, default=150)
    parser.add_argument("--cb_lr", type=float, default=1e-4)
    parser.add_argument("--cb_batch_size", type=int, default=4)
    parser.add_argument("--cb_coeff_max", type=float, default=4.0)
    parser.add_argument("--cb_retain_coeff_max", type=float, default=1.0)
    parser.add_argument("--alphasteer_strength", type=float, default=0.4)
    parser.add_argument("--alphasteer_null_ratio", type=float, default=0.5)
    parser.add_argument("--alphasteer_lambda", type=float, default=10.0)
    return parser.parse_args()


def main():
    args = parse_args()
    models = []
    for item in args.models.split(","):
        item = item.strip()
        if not item:
            continue
        models.append(MODEL_ALIASES.get(item, item))
    defenses = [d.strip() for d in args.defenses.split(",") if d.strip()]
    unknown = [d for d in defenses if d not in DEFENSES]
    if unknown:
        raise SystemExit(f"unknown defenses: {unknown}")

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    csv_path = Path(args.save_csv) if args.save_csv else (
        output_dir / f"defense_utility_{_dt.datetime.now().strftime('%Y%m%d_%H%M%S')}.csv"
    )

    print(f"[utility] models={models}")
    print(f"[utility] defenses={defenses}")
    print(f"[utility] csv={csv_path}")

    for model_path in models:
        for defense in defenses:
            row = run_one(model_path, defense, args)
            _append_row(csv_path, row)
            print(f"[utility] appended {row['model_key']} / {defense}: {row['status']}")

    print(f"[utility] done -> {csv_path}")


if __name__ == "__main__":
    main()
