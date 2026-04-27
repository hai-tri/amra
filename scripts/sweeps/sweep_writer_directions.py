#!/usr/bin/env python3
"""
Sweep APRS rank-k writer-output directions.

This is intentionally a thin wrapper around ``scripts/eval/quick_attack_test.py``
so the sweep reports the same contemporary-weight Arditi/PCA-8 attack metrics
used in the quick paper diagnostics.
"""

import argparse
import csv
import datetime
import os
import sys

REPO_DIR = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.insert(0, REPO_DIR)
sys.path.insert(0, os.path.join(REPO_DIR, "scripts", "eval"))

from quick_attack_test import CSV_FIELDS, CONFIGS, run_config


def _parse_int_list(text: str):
    return [int(x.strip()) for x in text.split(",") if x.strip()]


def main():
    pa = argparse.ArgumentParser()
    pa.add_argument("--model", choices=["llama", "qwen", "gemma", "all"], default="llama")
    pa.add_argument("--ks", default="1,2,4,8",
                    help="Comma-separated num_writer_directions values")
    pa.add_argument("--n", type=int, default=20,
                    help="Harmful/harmless prompts for attack eval")
    pa.add_argument("--skip_utility", action="store_true",
                    help="Only run refusal attacks")
    pa.add_argument("--bpb_batches", type=int, default=32)
    pa.add_argument("--mmlu_n", type=int, default=200)
    pa.add_argument("--math500_n", type=int, default=200)
    pa.add_argument("--batch_size", type=int, default=8)
    pa.add_argument("--output_dir",
                    default=os.path.join(REPO_DIR, "results", "writer_direction_sweep"))
    args = pa.parse_args()

    keys = list(CONFIGS.keys()) if args.model == "all" else [args.model]
    ks = _parse_int_list(args.ks)
    if not ks:
        raise ValueError("--ks must contain at least one integer")

    os.makedirs(args.output_dir, exist_ok=True)
    ts = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
    csv_path = os.path.join(args.output_dir, f"writer_direction_sweep_{ts}.csv")

    results = []
    with open(csv_path, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=CSV_FIELDS, extrasaction="ignore")
        writer.writeheader()

        for model_key in keys:
            for k in ks:
                try:
                    row = run_config(
                        model_key=model_key,
                        n_prompts=args.n,
                        skip_utility=args.skip_utility,
                        bpb_batches=args.bpb_batches,
                        mmlu_n=args.mmlu_n,
                        math500_n=args.math500_n,
                        batch_size=args.batch_size,
                        num_writer_directions=k,
                    )
                    if row:
                        writer.writerow(row)
                        f.flush()
                        results.append(row)
                except Exception as exc:
                    print(f"[ERROR] model={model_key} k={k}: {exc}")
                    import traceback
                    traceback.print_exc()

    print(f"\nSaved → {csv_path}")
    if results:
        print("\nSummary: defended refusal scores")
        print(f"{'model':<24} {'k':>3} {'base':>9} {'arditi':>9} {'pca8':>9} {'cos':>9}")
        print("-" * 67)
        for r in results:
            print(
                f"{r['model']:<24} {r['num_writer_directions']:>3} "
                f"{r['ref_def_base']:>9.4f} {r['ref_def_arditi']:>9.4f} "
                f"{r['ref_def_pca8']:>9.4f} {r['avg_cos_sim']:>9.4f}"
            )


if __name__ == "__main__":
    main()
