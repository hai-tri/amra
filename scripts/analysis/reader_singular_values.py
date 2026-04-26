"""
Reader-matrix singular value analysis.

For every layer of a target model, compute the singular value spectrum of
each reader projection (Q, K, V, gate, up) and report:
    * number of *exactly* zero singular values,
    * number of singular values below a relative tolerance (σ_i / σ_max < tol),
    * smallest nonzero singular value,
    * effective rank (= count above tol).

This quantifies how much null space exists in the reader weights — the
"free" rank available for compensating rank-1 updates that do not disturb
the model's normal-activation behaviour.

Usage
-----
    python scripts/reader_singular_values.py \
        --model_path meta-llama/Meta-Llama-3-8B-Instruct \
        --out_csv results/reader_svals.csv \
        --out_fig results/reader_svals.png \
        --tol 1e-6
"""

import argparse
import csv
import os
import sys

import matplotlib.pyplot as plt
import numpy as np
import torch

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)
sys.path.insert(0, os.path.join(ROOT, "refusal_direction"))

from obfuscation_utils import ModelComponents
from pipeline.model_utils.model_factory import construct_model_base


READER_PROJS = [
    ("q_proj", "attn"),
    ("k_proj", "attn"),
    ("v_proj", "attn"),
    ("gate_proj", "mlp"),
    ("up_proj", "mlp"),
]


def analyse_matrix(W: torch.Tensor, tol_rel: float, tol_abs: float):
    W_f = W.detach().float().cpu()
    m, n = W_f.shape
    s = torch.linalg.svdvals(W_f).numpy()
    # Full-SVD convention: pad spectrum with structural zeros to length max(m, n).
    full_len = max(m, n)
    s_full = np.concatenate([s, np.zeros(full_len - s.size, dtype=s.dtype)])
    spectrum = s_full.tolist()
    sigma_max = float(s.max()) if s.size else 0.0
    sigma_min = float(s_full.min()) if s_full.size else 0.0
    threshold = max(tol_rel * sigma_max, tol_abs)

    n_exact_zero = int((s_full == 0).sum())
    n_below_tol = int((s_full < threshold).sum())
    eff_rank = int((s >= threshold).sum())
    max_possible = full_len

    return {
        "rows": m,
        "cols": n,
        "sigma_max": sigma_max,
        "sigma_min": sigma_min,
        "n_exact_zero": n_exact_zero,
        "n_below_tol": n_below_tol,
        "effective_rank": eff_rank,
        "max_rank": max_possible,
        "null_space_dim": max_possible - eff_rank,
        "spectrum": spectrum,
    }


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model_path", default="meta-llama/Meta-Llama-3-8B-Instruct")
    ap.add_argument("--out_csv", default="results/reader_svals.csv")
    ap.add_argument("--out_spectrum_csv", default="results/reader_svals_spectrum.csv",
                    help="Per-matrix singular value spectrum: one row per reader, σ_i across columns.")
    ap.add_argument("--out_fig", default="results/reader_svals.png")
    ap.add_argument("--out_zeros_csv", default="results/reader_svals_zeros.csv",
                    help="Compact table: layer, proj, rows, cols, n_zero singular values.")
    ap.add_argument("--tol_rel", type=float, default=1e-6,
                    help="Relative threshold: σ_i < tol_rel * σ_max counts as zero.")
    ap.add_argument("--tol_abs", type=float, default=0.0,
                    help="Absolute threshold: σ_i < tol_abs counts as zero.")
    args = ap.parse_args()

    os.makedirs(os.path.dirname(args.out_csv) or ".", exist_ok=True)
    os.makedirs(os.path.dirname(args.out_fig) or ".", exist_ok=True)

    print(f"Loading {args.model_path} …")
    model_base = construct_model_base(args.model_path)
    model = model_base.model
    components = ModelComponents(model)
    num_layers = components.num_layers

    rows = []
    for ell in range(num_layers):
        attn_readers = dict(components.get_attn_reader_projs(ell))
        mlp_readers = dict(components.get_mlp_reader_projs(ell))
        by_name = {**attn_readers, **mlp_readers}

        for name, kind in READER_PROJS:
            if name not in by_name:
                continue
            stats = analyse_matrix(by_name[name].weight, args.tol_rel, args.tol_abs)
            stats.update(layer=ell, proj=name, kind=kind)
            rows.append(stats)
            print(
                f"  L{ell:02d} {name:10s} "
                f"{stats['rows']:5d}×{stats['cols']:<5d} "
                f"rank={stats['effective_rank']:5d}/{stats['max_rank']:5d} "
                f"null={stats['null_space_dim']:5d} "
                f"exact0={stats['n_exact_zero']:4d} "
                f"σ_max={stats['sigma_max']:.3g} σ_min={stats['sigma_min']:.3g}"
            )

    # ---- CSV ----
    fieldnames = ["layer", "proj", "kind", "rows", "cols", "max_rank",
                  "effective_rank", "null_space_dim",
                  "n_exact_zero", "n_below_tol",
                  "sigma_max", "sigma_min"]
    with open(args.out_csv, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=fieldnames)
        w.writeheader()
        for r in rows:
            w.writerow({k: r[k] for k in fieldnames})
    print(f"Wrote CSV: {args.out_csv}")

    # ---- Spectrum CSV (one row per matrix, σ_i across columns) ----
    max_len = max(len(r["spectrum"]) for r in rows)
    spectrum_header = ["layer", "proj", "kind"] + [f"sigma_{i+1}" for i in range(max_len)]
    with open(args.out_spectrum_csv, "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(spectrum_header)
        for r in rows:
            pad = max_len - len(r["spectrum"])
            w.writerow([r["layer"], r["proj"], r["kind"]] + r["spectrum"] + [""] * pad)
    print(f"Wrote spectrum CSV: {args.out_spectrum_csv}")

    with open(args.out_zeros_csv, "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["layer", "proj", "kind", "rows", "cols", "n_zero_singular_values"])
        for r in rows:
            w.writerow([r["layer"], r["proj"], r["kind"], r["rows"], r["cols"], r["n_exact_zero"]])
    print(f"Wrote zeros CSV: {args.out_zeros_csv}")

    # ---- Plot: null-space dim per layer per projection ----
    layers = sorted({r["layer"] for r in rows})
    fig, ax = plt.subplots(figsize=(8, 4.5))
    colors = {"q_proj": "#1f77b4", "k_proj": "#ff7f0e", "v_proj": "#2ca02c",
              "gate_proj": "#d62728", "up_proj": "#9467bd"}
    for name, _ in READER_PROJS:
        ys = [r["null_space_dim"] for r in rows if r["proj"] == name]
        if not ys:
            continue
        ax.plot(layers, ys, marker="o", linewidth=1.5,
                color=colors.get(name), label=name)

    ax.set_xlabel("Layer Index")
    ax.set_ylabel(r"Null-space dim ($\sigma_i < \tau \cdot \sigma_{\max}$)")
    ax.set_title("Reader-Matrix Null-Space Dimension per Layer")
    ax.grid(True, alpha=0.3)
    ax.legend(loc="best", ncol=2)
    plt.tight_layout()
    plt.savefig(args.out_fig, dpi=180)
    print(f"Saved figure: {args.out_fig}")

    # ---- Aggregate summary ----
    print("\n=== Summary (across all layers) ===")
    for name, _ in READER_PROJS:
        rs = [r for r in rows if r["proj"] == name]
        if not rs:
            continue
        ns = np.array([r["null_space_dim"] for r in rs])
        ez = np.array([r["n_exact_zero"] for r in rs])
        print(f"  {name:10s} null mean={ns.mean():7.1f} "
              f"min={ns.min():5d} max={ns.max():5d}   "
              f"exact-zero mean={ez.mean():6.1f}")


if __name__ == "__main__":
    main()
