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


def _plot_epsilon_sweep(ax, data, key, ylabel):
    cmap = plt.get_cmap("viridis")
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


def _plot_layer_sweep(ax, data, key, ylabel):
    cmap = plt.get_cmap("viridis")
    layers = data["layers"]
    x = np.arange(len(data[f"mean_{layers[0]}"]))
    for i, sl in enumerate(layers):
        c = cmap(i / max(1, len(layers) - 1))
        m = data[f"mean_{sl}"]; s = data[f"std_{sl}"]
        ax.plot(x, m, marker="o", linewidth=1.6, markersize=4,
                color=c, label=f"patched layer {sl}")
        ax.fill_between(x, m - s, m + s, alpha=0.12, color=c)
    if "random_layers" in data:
        rl = data["random_layers"]
        rk = f"random{len(rl)}"
        if f"mean_{rk}" in data:
            m = data[f"mean_{rk}"]; s = data[f"std_{rk}"]
            rl_str = ",".join(str(r) for r in rl)
            ax.plot(x, m, marker="s", linewidth=2.0, color="black",
                    linestyle="--", label=f"random {len(rl)} (L={rl_str})")
            ax.fill_between(x, m - s, m + s, alpha=0.08, color="black")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--l2",  default="results/drift_sweep/drift_eps_sweep_l2.npz")
    ap.add_argument("--var", default="results/drift_sweep/drift_eps_sweep_var.npz")
    ap.add_argument("--cos", default="results/drift_sweep/drift_eps_sweep_cos.npz")
    ap.add_argument("--out", default="results/drift_sweep/drift_eps_sweep_combined.png")
    ap.add_argument("--title", default=None, help="Override suptitle")
    args = ap.parse_args()

    paths = {"l2": args.l2, "var": args.var, "cos": args.cos}
    fig, axes = plt.subplots(1, 3, figsize=(15, 4.2))

    for ax, (key, title, ylabel) in zip(axes, METRICS):
        data = np.load(paths[key])
        is_layer_sweep = "layers" in data
        if is_layer_sweep:
            _plot_layer_sweep(ax, data, key, ylabel)
        else:
            _plot_epsilon_sweep(ax, data, key, ylabel)
        ax.set_xlabel("Layer Index")
        ax.set_ylabel(ylabel)
        ax.set_title(title)
        ax.grid(True, alpha=0.3)

    axes[0].legend(loc="upper left", fontsize=11, markerscale=1.4,
                   handlelength=2.2, borderpad=0.6, labelspacing=0.5)
    if args.title:
        fig.suptitle(args.title, fontsize=13)
    plt.tight_layout()
    plt.savefig(args.out, dpi=180, bbox_inches="tight")
    print(f"Saved: {args.out}")


if __name__ == "__main__":
    main()
