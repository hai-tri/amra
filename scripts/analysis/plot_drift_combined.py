"""Combine l2 / var / cos drift sweeps into one 1x3 figure for NeurIPS."""

import argparse
import os

import matplotlib.pyplot as plt
import numpy as np

plt.rcParams.update({
    "font.size": 15,
    "axes.titlesize": 16,
    "axes.labelsize": 15,
    "xtick.labelsize": 15,
    "ytick.labelsize": 15,
    "legend.fontsize": 15,
})


METRICS = [
    ("l2",  "(a) Residual Stream L2 Differences",
     r"$\|h_{\mathrm{patched}} - h_{\mathrm{clean}}\|_2$"),
    ("var", "(b) Residual Stream Variance Shift",
     r"$\mathrm{Var}(h_{\mathrm{patched}} - h_{\mathrm{clean}})$"),
    ("cos", "(c) Residual Stream Cosine Similarity",
     r"$\cos(h_{\mathrm{patched}},\, h_{\mathrm{clean}})$"),
]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--l2",  default="results/drift_sweep/drift_eps_sweep_l2.npz")
    ap.add_argument("--var", default="results/drift_sweep/drift_eps_sweep_var.npz")
    ap.add_argument("--cos", default="results/drift_sweep/drift_eps_sweep_cos.npz")
    ap.add_argument("--out", default="results/drift_sweep/drift_eps_sweep_combined.png")
    args = ap.parse_args()

    paths = {"l2": args.l2, "var": args.var, "cos": args.cos}
    fig, axes = plt.subplots(1, 3, figsize=(15, 4.2))
    cmap = plt.get_cmap("viridis")

    for ax, (key, title, ylabel) in zip(axes, METRICS):
        data = np.load(paths[key])
        eps_list = data["epsilons"]
        layer = int(data["layer"])
        x = np.arange(len(data[f"mean_{eps_list[0]}"]))
        for i, eps in enumerate(eps_list):
            c = cmap(i / max(1, len(eps_list) - 1))
            m = data[f"mean_{eps}"]; s = data[f"std_{eps}"]
            ax.plot(x, m, marker="o", linewidth=1.6, markersize=4,
                    color=c, label=fr"$\varepsilon = {eps}$")
            ax.fill_between(x, m - s, m + s, alpha=0.12, color=c)
        ax.axvline(layer, color="gray", linestyle="--",
                   alpha=0.5, linewidth=0.8, label="selected layer")
        ax.set_xlabel("Layer Index")
        ax.set_ylabel(ylabel)
        ax.set_title(title)
        ax.grid(True, alpha=0.3)

    axes[0].legend(loc="upper left", fontsize=11, markerscale=1.4,
                   handlelength=2.2, borderpad=0.6, labelspacing=0.5)
    plt.tight_layout()
    plt.savefig(args.out, dpi=180, bbox_inches="tight")
    print(f"Saved: {args.out}")


if __name__ == "__main__":
    main()
