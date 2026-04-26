"""
Right-null-space overlap between reader matrices across layers.

For Llama-style models only k_proj and v_proj have a nontrivial input (right)
null space (3072-dim for Llama-3-8B). We compute an orthonormal basis N_i for
each such matrix and, for each ordered pair (i, j) with j >= i, report:

  * intersect_dim : # principal angles with cos > cos_tol (default 0.9999)
  * mean_cos2     : ||N_i^T N_j||_F^2 / dim(N_i)
                    — fraction of N_i captured by span(N_j) on average.

Usage:
    python3 scripts/reader_nullspace_overlap.py \
        --model_path meta-llama/Meta-Llama-3-8B-Instruct \
        --out_csv results/reader_nullspace_overlap.csv
"""

import argparse
import csv
import os
import sys

import torch

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)
sys.path.insert(0, os.path.join(ROOT, "refusal_direction"))

from obfuscation_utils import ModelComponents
from pipeline.model_utils.model_factory import construct_model_base


TARGET_READERS = ["k_proj", "v_proj"]  # q/gate/up have trivial right null space


def right_null_basis(W: torch.Tensor) -> torch.Tensor:
    """Return orthonormal basis (n, n - rank) spanning {x : W x = 0}."""
    W_f = W.detach().float().cpu()
    m, n = W_f.shape
    # Full SVD so Vh is (n, n).
    _, s, Vh = torch.linalg.svd(W_f, full_matrices=True)
    # Rank = effective rank under tight tolerance.
    tol = max(m, n) * s.max().item() * 1e-6
    rank = int((s > tol).sum().item())
    # Null space = rows Vh[rank:] transposed → columns of V.
    N = Vh[rank:].T.contiguous()  # (n, n - rank)
    return N


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model_path", default="meta-llama/Meta-Llama-3-8B-Instruct")
    ap.add_argument("--out_csv", default="results/reader_nullspace_overlap.csv")
    ap.add_argument("--cos_tol", type=float, default=0.9999,
                    help="Principal-angle cosines above this count as shared dimensions.")
    ap.add_argument("--device", default="cpu",
                    help="Device for pairwise matmul + svdvals (cpu, mps, cuda). "
                         "Initial full-SVDs for basis extraction always run on CPU.")
    args = ap.parse_args()
    device = torch.device(args.device)

    os.makedirs(os.path.dirname(args.out_csv) or ".", exist_ok=True)

    print(f"Loading {args.model_path} …")
    model_base = construct_model_base(args.model_path)
    components = ModelComponents(model_base.model)

    bases = []  # list of (layer, proj, N)
    for ell in range(components.num_layers):
        attn_readers = dict(components.get_attn_reader_projs(ell))
        for name in TARGET_READERS:
            if name not in attn_readers:
                continue
            N = right_null_basis(attn_readers[name].weight).to(device)
            bases.append((ell, name, N))
            print(f"  L{ell:02d} {name:8s} null basis shape={tuple(N.shape)}")

    print(f"\nComputing pairwise overlap for {len(bases)} bases "
          f"({len(bases)*(len(bases)+1)//2} pairs) …")

    rows = []
    for i in range(len(bases)):
        li, pi, Ni = bases[i]
        for j in range(i, len(bases)):
            lj, pj, Nj = bases[j]
            M = Ni.T @ Nj                       # (r_i, r_j)
            fro2 = float((M * M).sum().item())
            mean_cos2 = fro2 / Ni.shape[1]
            if i == j:
                intersect_dim = Ni.shape[1]
            else:
                # Principal-angle cosines = singular values of M.
                s = torch.linalg.svdvals(M.cpu())
                intersect_dim = int((s > args.cos_tol).sum().item())
            rows.append({
                "layer_i": li, "proj_i": pi, "dim_i": Ni.shape[1],
                "layer_j": lj, "proj_j": pj, "dim_j": Nj.shape[1],
                "intersect_dim": intersect_dim,
                "mean_cos2": mean_cos2,
            })
        print(f"  pair row i={i+1}/{len(bases)} done")

    fieldnames = ["layer_i", "proj_i", "dim_i",
                  "layer_j", "proj_j", "dim_j",
                  "intersect_dim", "mean_cos2"]
    with open(args.out_csv, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=fieldnames)
        w.writeheader()
        w.writerows(rows)
    print(f"\nWrote {len(rows)} pairs → {args.out_csv}")


if __name__ == "__main__":
    main()
