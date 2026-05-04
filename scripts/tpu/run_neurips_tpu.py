#!/usr/bin/env python3
"""
Launch the full APRS NeurIPS TPU pipeline on TRC v6e.

Default grid:
  3 models × {undefended, per-model APRS optimum, surgical, cast,
              circuit_breakers, alphasteer} = 18 rows × full attack/utility suite.

The launcher is intentionally a subprocess orchestrator. Each row gets a fresh
Python process, which keeps XLA compile state and any baked weight edits from
leaking across defenses. Use --parallel_cells 4 on a v6e-16 allocation to keep
four independent cells busy. By default, worker i gets TPU_VISIBLE_CHIPS
``4*i..4*i+3``; override --chips_per_cell or pre-set TPU_VISIBLE_CHIPS if your
TRC setup binds cells externally.
"""

from __future__ import annotations

import argparse
import json
import os
import queue
import subprocess
import sys
import threading
import time
from dataclasses import dataclass
from pathlib import Path
from typing import List

REPO = Path(__file__).resolve().parents[2]


DEFAULT_MODELS = [
    "meta-llama/Meta-Llama-3-8B-Instruct",
    "Qwen/Qwen3-8B",
    "google/gemma-2-9b-it",
]


@dataclass(frozen=True)
class DefenseRow:
    name: str
    args: tuple[str, ...]
    # Empty tuple = applies to all models. Otherwise, only the listed model
    # paths get this row. Used for per-model APRS optima where ε / L / k_w / k_r
    # differ across architectures.
    only_models: tuple[str, ...] = ()


def _aprs_row(name: str, *, model: str, eps: float, layers: int,
              k_w: int, k_r: int) -> DefenseRow:
    args = (
        "--defense_type", "obfuscation",
        "--projection_mode", "full",
        "--epsilon", str(eps),
        "--num_pertinent_layers", str(layers),
        "--per_layer_direction",
        "--writer_output_directions",
        "--num_writer_directions", str(k_w),
        "--num_reader_directions", str(k_r),
    )
    return DefenseRow(name, args, only_models=(model,))


# Per-model APRS optima (full projection, rank-k writer + reader). Baselines are
# the undefended row. Other defense families (surgical / CAST / CB / AlphaSteer)
# are intentionally omitted from the TPU launch — re-run via existing GPU
# artifacts when needed.
DEFENSES = [
    DefenseRow("undefended", ("--defense_type", "none")),
    _aprs_row(
        "aprs_llama_optimal",
        model="meta-llama/Meta-Llama-3-8B-Instruct",
        eps=0.025, layers=20, k_w=1, k_r=8,
    ),
    _aprs_row(
        "aprs_gemma_optimal",
        model="google/gemma-2-9b-it",
        eps=0.025, layers=30, k_w=4, k_r=16,
    ),
    _aprs_row(
        "aprs_qwen_optimal",
        model="Qwen/Qwen3-8B",
        eps=0.5, layers=20, k_w=1, k_r=8,
    ),
    DefenseRow("surgical", ("--defense_type", "surgical")),
    DefenseRow("cast", ("--defense_type", "cast")),
    DefenseRow("circuit_breakers", ("--defense_type", "circuit_breakers")),
    DefenseRow("alphasteer", ("--defense_type", "alphasteer")),
]


FULL_SUITE_ARGS = (
    "--llamaguard",
    "--gcg", "--gcg_n_behaviors", "25", "--gcg_steps", "500",
    "--autodan", "--autodan_n_behaviors", "25",
    "--pair", "--pair_n_behaviors", "25",
    "--renellm",
    "--softopt", "--softopt_limit", "25",
    "--harmbench_n", "100",
    "--ce_loss_n_batches", "256",
    "--lm_harness_tasks", "gsm8k,math500,mmlu",
    "--lm_harness_n", "500",
    "--alpacaeval_n", "805",
    "--skip_heretic",
    "--pca_top_k", "8",
    "--tpu_native_utility",
)


def _model_key(model: str) -> str:
    lower = model.lower()
    if "llama" in lower:
        return "llama3_8b"
    if "qwen" in lower:
        return "qwen3_8b"
    if "gemma" in lower:
        return "gemma2_9b"
    return Path(model).name.lower().replace("/", "_")


def _build_jobs(args) -> list[dict]:
    models = [m.strip() for m in args.models.split(",") if m.strip()]
    defenses = DEFENSES
    if args.only_defense:
        keep = {x.strip() for x in args.only_defense.split(",")}
        defenses = [d for d in defenses if d.name in keep]

    jobs = []
    for model in models:
        for defense in defenses:
            if defense.only_models and model not in defense.only_models:
                continue
            tag = f"{_model_key(model)}_{defense.name}"
            if args.resume and (Path(args.output_dir) / "logs" / f"{tag}.log").exists():
                continue
            if len(jobs) % args.num_shards != args.shard_index:
                jobs.append({"skip_shard": True})
                continue
            artifact_subdir = f"tpu_{defense.name}"
            cmd = [
                sys.executable,
                str(REPO / "run_obfuscation_pipeline.py"),
                "--model_path", model,
                "--artifact_subdir", artifact_subdir,
                "--save_csv", args.save_csv,
                "--xla_buckets", args.xla_buckets,
                *defense.args,
                *FULL_SUITE_ARGS,
            ]
            # Defense rows reuse the undef row's baseline utility numbers.
            # The undefended row itself must compute them (its Stage 9 IS the
            # baseline), so the flag is gated on defense.name.
            if defense.name != "undefended":
                cmd.append("--skip_undef_utility")
            if args.skip_direction_extraction:
                cmd.append("--skip_direction_extraction")
            if args.extra_args:
                cmd.extend(args.extra_args.split())
            jobs.append({
                "tag": tag,
                "model": model,
                "defense": defense.name,
                "cmd": cmd,
            })
    return [j for j in jobs if not j.get("skip_shard")]


def _run_job(job: dict, args, worker_idx: int) -> int:
    out_dir = Path(args.output_dir)
    log_dir = out_dir / "logs"
    log_dir.mkdir(parents=True, exist_ok=True)
    log_path = log_dir / f"{job['tag']}.log"
    env = os.environ.copy()
    env.setdefault("PJRT_DEVICE", "TPU")
    if args.chips_per_cell > 0 and "TPU_VISIBLE_CHIPS" not in env:
        first_chip = worker_idx * args.chips_per_cell
        chips = range(first_chip, first_chip + args.chips_per_cell)
        env["TPU_VISIBLE_CHIPS"] = ",".join(str(c) for c in chips)
    env["APRS_TPU_CELL_INDEX"] = str(worker_idx)
    env["APRS_TPU_PARALLEL_CELLS"] = str(args.parallel_cells)

    print(f"[cell {worker_idx}] start {job['tag']} -> {log_path}", flush=True)
    t0 = time.time()
    with log_path.open("w") as log:
        proc = subprocess.run(
            job["cmd"],
            cwd=str(REPO),
            env=env,
            stdout=log,
            stderr=subprocess.STDOUT,
            text=True,
        )
    dt = (time.time() - t0) / 3600
    print(
        f"[cell {worker_idx}] done {job['tag']} rc={proc.returncode} "
        f"wall={dt:.2f}h",
        flush=True,
    )

    if proc.returncode == 0 and args.parity_after_each:
        parity_path = out_dir / "parity" / f"{job['tag']}.json"
        parity_cmd = [
            sys.executable,
            str(REPO / "scripts" / "tpu" / "parity_gates.py"),
            "--csv", args.save_csv,
            "--output", str(parity_path),
            "--abs_tolerance", str(args.parity_abs_tolerance),
            "--rel_tolerance", str(args.parity_rel_tolerance),
        ]
        subprocess.run(parity_cmd, cwd=str(REPO), check=False)
    return proc.returncode


def _worker(q: queue.Queue, args, worker_idx: int, failures: list):
    while True:
        try:
            job = q.get_nowait()
        except queue.Empty:
            return
        try:
            rc = _run_job(job, args, worker_idx)
            if rc != 0:
                failures.append((job["tag"], rc))
                if args.fail_fast:
                    while True:
                        try:
                            q.get_nowait()
                        except queue.Empty:
                            break
        finally:
            q.task_done()


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--models", default=",".join(DEFAULT_MODELS))
    parser.add_argument("--output_dir", default="results/tpu_neurips")
    parser.add_argument("--save_csv", default="results/tpu_neurips/all_results.csv")
    parser.add_argument("--parallel_cells", type=int, default=4)
    parser.add_argument("--chips_per_cell", type=int, default=4,
                        help="TPU chips assigned to each parallel worker; 4 maps v6e-16 to four cells")
    parser.add_argument("--num_shards", type=int, default=1)
    parser.add_argument("--shard_index", type=int, default=0)
    parser.add_argument("--xla_buckets", default="512,1024,2048,4096")
    parser.add_argument("--only_defense", default=None)
    parser.add_argument("--extra_args", default="")
    parser.add_argument("--skip_direction_extraction", action="store_true")
    parser.add_argument("--resume", action="store_true")
    parser.add_argument("--dry_run", action="store_true")
    parser.add_argument("--fail_fast", action="store_true")
    parser.add_argument("--parity_after_each", action="store_true")
    parser.add_argument("--parity_abs_tolerance", type=float, default=0.10)
    parser.add_argument("--parity_rel_tolerance", type=float, default=0.25)
    args = parser.parse_args()

    if not (0 <= args.shard_index < args.num_shards):
        raise SystemExit("--shard_index must be in [0, --num_shards)")

    Path(args.output_dir).mkdir(parents=True, exist_ok=True)
    Path(args.save_csv).parent.mkdir(parents=True, exist_ok=True)

    jobs = _build_jobs(args)
    manifest = {
        "models": [m.strip() for m in args.models.split(",") if m.strip()],
        "defenses": [d.name for d in DEFENSES],
        "n_jobs": len(jobs),
        "parallel_cells": args.parallel_cells,
        "chips_per_cell": args.chips_per_cell,
        "xla_buckets": args.xla_buckets,
        "full_suite_args": list(FULL_SUITE_ARGS),
    }
    manifest_path = Path(args.output_dir) / "manifest.json"
    with manifest_path.open("w") as f:
        json.dump(manifest, f, indent=2)
    print(json.dumps(manifest, indent=2))

    if args.dry_run:
        for job in jobs:
            print(" ".join(job["cmd"]))
        return

    q: queue.Queue = queue.Queue()
    for job in jobs:
        q.put(job)
    failures: list = []
    threads: List[threading.Thread] = []
    for idx in range(min(args.parallel_cells, len(jobs))):
        t = threading.Thread(target=_worker, args=(q, args, idx, failures), daemon=True)
        t.start()
        threads.append(t)
    for t in threads:
        t.join()

    if failures:
        print(f"[FAIL] {len(failures)} job(s) failed: {failures}")
        raise SystemExit(1)
    print("[PASS] TPU NeurIPS grid completed")


if __name__ == "__main__":
    main()
