"""
Writer-matrix singular value analysis.

The writers are the projections *immediately downstream of the readers* —
attention `o_proj` (consumes Q/K/V output) and MLP `down_proj` (consumes
gate/up output). Same analysis as reader_singular_values.py, different
matrices.

Outputs:
  * summary CSV: layer, proj, rank, null_dim, σ_max/min
  * spectrum CSV: one row per matrix, σ_i across columns

Usage:
    python3 scripts/writer_singular_values.py \
        --model_path meta-llama/Meta-Llama-3-8B-Instruct \
        --out_csv results/writer_svals.csv \
        --out_spectrum_csv results/writer_svals_spectrum.csv
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


WRITER_PROJS = [
    ("o_proj",    "attn"),
    ("down_proj", "mlp"),
]


def analyse_matrix(W: torch.Tensor, tol_rel: float, tol_abs: float):
    W_f = W.detach().float().cpu()
    m, n = W_f.shape
    s = torch.linalg.svdvals(W_f).numpy()
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
        "rows": m, "cols": n,
        "sigma_max": sigma_max, "sigma_min": sigma_min,
        "n_exact_zero": n_exact_zero, "n_below_tol": n_below_tol,
        "effective_rank": eff_rank, "max_rank": max_possible,
        "null_space_dim": max_possible - eff_rank,
        "spectrum": spectrum,
    }


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model_path", default="meta-llama/Meta-Llama-3-8B-Instruct")
    ap.add_argument("--out_csv", default="results/writer_svals.csv")
    ap.add_argument("--out_spectrum_csv", default="results/writer_svals_spectrum.csv")
    ap.add_argument("--out_fig", default="results/writer_svals.png")
    ap.add_argument("--tol_rel", type=float, default=1e-6)
    ap.add_argument("--tol_abs", type=float, default=0.0)
    args = ap.parse_args()

    for p in (args.out_csv, args.out_spectrum_csv, args.out_fig):
        os.makedirs(os.path.dirname(p) or ".", exist_ok=True)

    print(f"Loading {args.model_path} …")
    model_base = construct_model_base(args.model_path)
    components = ModelComponents(model_base.model)

    rows = []
    for ell in range(components.num_layers):
        by_name = {
            "o_proj":    components.get_attn_output_proj(ell),
            "down_proj": components.get_mlp_output_proj(ell),
        }
        for name, kind in WRITER_PROJS:
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
                f"σ_max={stats['sigma_max']:.3g} σ_min={stats['sigma_min']:.3g}"
            )

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

    max_len = max(len(r["spectrum"]) for r in rows)
    header = ["layer", "proj", "kind"] + [f"sigma_{i+1}" for i in range(max_len)]
    with open(args.out_spectrum_csv, "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(header)
        for r in rows:
            pad = max_len - len(r["spectrum"])
            w.writerow([r["layer"], r["proj"], r["kind"]] + r["spectrum"] + [""] * pad)
    print(f"Wrote spectrum CSV: {args.out_spectrum_csv}")

    layers = sorted({r["layer"] for r in rows})
    fig, ax = plt.subplots(figsize=(8, 4.5))
    colors = {"o_proj": "#1f77b4", "down_proj": "#d62728"}
    for name, _ in WRITER_PROJS:
        ys = [r["null_space_dim"] for r in rows if r["proj"] == name]
        if not ys:
            continue
        ax.plot(layers, ys, marker="o", linewidth=1.5, color=colors.get(name), label=name)
    ax.set_xlabel("Layer Index")
    ax.set_ylabel(r"Null-space dim ($\sigma_i < \tau \cdot \sigma_{\max}$)")
    ax.set_title("Writer-Matrix Null-Space Dimension per Layer")
    ax.grid(True, alpha=0.3)
    ax.legend(loc="best")
    plt.tight_layout()
    plt.savefig(args.out_fig, dpi=180)
    print(f"Saved figure: {args.out_fig}")


if __name__ == "__main__":
    main()
