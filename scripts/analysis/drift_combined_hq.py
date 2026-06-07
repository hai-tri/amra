#!/usr/bin/env python3
"""Re-render the 2-row drift figure at high quality from raw data.

Row 1: epsilon sweep (fixed layer, varying epsilon) — requires model run.
Row 2: layer-group sweep (writer-only, eps=0.025) — loaded from .npz.

Saves PNG (300 DPI) and PDF (vector).
"""

import json
import os
import sys
import tempfile

import matplotlib.pyplot as plt
import numpy as np
import torch

ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.insert(0, ROOT)
sys.path.insert(0, os.path.join(ROOT, "refusal_direction"))
sys.path.insert(0, os.path.join(ROOT, "scripts", "eval"))

from obfuscation_utils import (
    ModelComponents,
    collect_writer_output_refusal_directions,
    generate_random_alias,
)
from pipeline.model_utils.model_factory import construct_model_base
from pipeline.submodules.generate_directions import generate_directions
from pipeline.submodules.select_direction import select_direction
from run_obfuscation_pipeline import filter_data, load_mlabonne_datasets
from pathlib import Path


def _rank_k_writer_update(W, directions, aliases):
    orig_dtype = W.dtype
    W_f = W.float()
    r = directions.float().to(W.device)
    a = aliases.float().to(W.device)
    r = r / (r.norm(dim=-1, keepdim=True) + 1e-8)
    coeff_rows = r @ W_f
    return (W_f + (a - r).T @ coeff_rows).to(orig_dtype)


def patch_single_writer(model, components, layer, writer, directions, epsilon, seed):
    device = next(model.parameters()).device
    generator = torch.Generator(device=device)
    generator.manual_seed(seed)
    k = directions.shape[0]
    aliases = torch.stack([
        generate_random_alias(components.d_model, epsilon, device, generator)
        for _ in range(k)
    ]).to(device)
    if writer == "attn":
        proj = components.get_attn_output_proj(layer)
    else:
        proj = components.get_mlp_output_proj(layer)
    proj.weight.data = _rank_k_writer_update(proj.weight.data, directions, aliases)


def capture_residual_stream(model, tokenize_fn, prompts, components):
    device = next(model.parameters()).device
    num_layers = components.num_layers
    results = []

    def make_hook(buf, idx):
        def hook(module, inp, output):
            x = inp[0] if isinstance(inp, tuple) else inp
            buf[idx] = x[0].detach().float().cpu()
        return hook

    model.eval()
    with torch.no_grad():
        for p_idx, prompt in enumerate(prompts):
            buf = [None] * (num_layers + 1)
            handles = []
            for ell in range(num_layers):
                handles.append(
                    components.get_attn_layernorm(ell).register_forward_hook(
                        make_hook(buf, ell))
                )
            handles.append(
                components.final_norm.register_forward_hook(make_hook(buf, num_layers))
            )
            inputs = tokenize_fn(instructions=[prompt])
            model(input_ids=inputs.input_ids.to(device),
                  attention_mask=inputs.attention_mask.to(device))
            for h in handles:
                h.remove()
            results.append(torch.stack(buf, dim=0))
            print(f"  captured {p_idx + 1}/{len(prompts)}")
    return results


def compute_metric(clean_list, patched_list, metric):
    P = len(clean_list)
    Lp1 = clean_list[0].shape[0]
    out = torch.zeros(P, Lp1)
    for i, (c, p) in enumerate(zip(clean_list, patched_list)):
        diff = (p - c).float()
        if metric == "l2":
            out[i] = diff.norm(dim=-1).mean(dim=-1)
        elif metric == "var":
            out[i] = diff.var(dim=-1, unbiased=False).mean(dim=-1)
        elif metric == "cos":
            cf = c.float(); pf = p.float()
            num = (cf * pf).sum(dim=-1)
            den = cf.norm(dim=-1) * pf.norm(dim=-1) + 1e-12
            out[i] = (num / den).mean(dim=-1)
    return out


def main():
    from quick_attack_test import CONFIGS
    model_id = CONFIGS["llama"][0]
    d = os.path.join(ROOT, "results", "drift_sweep")
    os.makedirs(d, exist_ok=True)

    epsilons = [0.025, 0.1, 0.3, 1.0]
    metrics = ["l2", "var", "cos"]
    seed = 42
    writer = "attn"

    # ── Load model + direction ──
    print(f"Loading {model_id} ...")
    model_base = construct_model_base(model_id)
    components = ModelComponents(model_base.model)

    harmful_train, harmless_train, harmful_val, harmless_val = load_mlabonne_datasets(
        n_train=400, n_val=100)
    print("Filtering data ...")
    harmful_train, harmless_train, harmful_val, harmless_val = filter_data(
        model_base, harmful_train, harmless_train, harmful_val, harmless_val)
    eval_prompts = harmful_val[:16]

    print("Extracting refusal direction ...")
    with tempfile.TemporaryDirectory() as tmp:
        mean_diffs = generate_directions(
            model_base, harmful_train, harmless_train, artifact_dir=tmp)
        pos, layer, direction = select_direction(
            model_base, harmful_val, harmless_val, mean_diffs, artifact_dir=tmp)
    print(f"Selected layer={layer}, pos={pos}")

    # Writer direction for selected layer
    print("Collecting writer directions ...")
    dirs = collect_writer_output_refusal_directions(
        model=model_base.model, components=components,
        harmful_prompts=harmful_train, harmless_prompts=harmless_train,
        tokenize_fn=model_base.tokenize_instructions_fn,
        num_prompts=64, forward_batch_size=16,
    )
    d_vec = dirs[writer][layer]
    writer_dirs = (d_vec / (d_vec.norm() + 1e-8)).unsqueeze(0)

    # Snapshot + clean residuals
    print("Snapshotting clean weights ...")
    clean_state = {k: v.detach().clone().cpu()
                   for k, v in model_base.model.state_dict().items()}
    print("Capturing clean residual streams ...")
    clean_resid = capture_residual_stream(
        model_base.model, model_base.tokenize_instructions_fn,
        eval_prompts, components)

    # ── Row 1: Epsilon sweep ──
    eps_curves = {m: {} for m in metrics}
    for eps in epsilons:
        print(f"\n=== epsilon={eps} ===")
        model_base.model.load_state_dict(clean_state)
        patch_single_writer(model_base.model, components, layer,
                            writer, writer_dirs, eps, seed)
        patched = capture_residual_stream(
            model_base.model, model_base.tokenize_instructions_fn,
            eval_prompts, components)
        for metric in metrics:
            vals = compute_metric(clean_resid, patched, metric)
            eps_curves[metric][eps] = {
                "mean": vals.mean(dim=0).numpy(),
                "std":  vals.std(dim=0).numpy(),
            }

    # Save epsilon npz for future use
    for metric in metrics:
        npz_path = os.path.join(d, f"drift_eps_sweep_{metric}_{writer}.npz")
        np.savez(npz_path, layer=layer, epsilons=np.array(epsilons),
                 **{f"mean_{e}": eps_curves[metric][e]["mean"] for e in epsilons},
                 **{f"std_{e}":  eps_curves[metric][e]["std"]  for e in epsilons})
        print(f"Saved: {npz_path}")

    model_base.model.load_state_dict(clean_state)

    # ── Row 2: Load from npz ──
    group_curves = {m: {} for m in metrics}
    for metric in metrics:
        npz = np.load(os.path.join(d, f"drift_group_sweep_eps0.025_{metric}_attn.npz"))
        for key in ["L9-16", "L17-23", "L24-31", "random6"]:
            group_curves[metric][key] = {
                "mean": npz[f"mean_{key}"],
                "std":  npz[f"std_{key}"],
            }
        random_layers = npz["random_layers"]

    # ── Render 2x3 figure ──
    plt.rcParams.update({
        "font.size": 15, "axes.titlesize": 16, "axes.labelsize": 15,
        "xtick.labelsize": 15, "ytick.labelsize": 15,
    })
    metric_titles = [
        "(a) Residual Stream L2 Differences",
        "(b) Residual Stream Variance Shift",
        "(c) Residual Stream Cosine Similarity",
    ]
    ylabels = [
        r"$\|h_{\mathrm{patched}} - h_{\mathrm{clean}}\|_2$",
        r"$\mathrm{Var}(h_{\mathrm{patched}} - h_{\mathrm{clean}})$",
        r"$\cos(h_{\mathrm{patched}},\, h_{\mathrm{clean}})$",
    ]

    x = np.arange(components.num_layers + 1)
    cmap = plt.get_cmap("viridis")

    fig, axes = plt.subplots(2, 3, figsize=(15, 8.4))

    # Row 1: epsilon sweep
    for col, (metric, title, ylabel) in enumerate(zip(metrics, metric_titles, ylabels)):
        ax = axes[0, col]
        for i, eps in enumerate(epsilons):
            c = cmap(i / max(1, len(epsilons) - 1))
            m = eps_curves[metric][eps]["mean"]
            s = eps_curves[metric][eps]["std"]
            ax.plot(x, m, marker="o", linewidth=1.6, markersize=4,
                    color=c, label=fr"$\varepsilon = {eps}$")
            ax.fill_between(x, m - s, m + s, alpha=0.12, color=c)
        ax.axvline(layer, color="gray", linestyle="--",
                   alpha=0.5, linewidth=0.8, label="selected layer")
        ax.set_xlabel("Layer Index")
        ax.set_ylabel(ylabel)
        ax.set_title(title)
        ax.grid(True, alpha=0.3)

    axes[0, 0].legend(loc="upper left", fontsize=11, markerscale=1.4,
                      handlelength=2.2, borderpad=0.6, labelspacing=0.5)

    # Row 2: layer group sweep
    colors_map = {"L9-16": "#1f77b4", "L17-23": "#ff7f0e",
                  "L24-31": "#2ca02c", "random6": "black"}
    styles_map = {"L9-16": "-", "L17-23": "-", "L24-31": "-", "random6": "--"}
    rl_str = ",".join(str(r) for r in random_layers)
    labels_map = {"L9-16": "Layers 9–16", "L17-23": "Layers 17–23",
                  "L24-31": "Layers 24–31",
                  "random6": f"Random 6 (L={rl_str})"}
    group_keys = ["L9-16", "L17-23", "L24-31", "random6"]

    for col, (metric, ylabel) in enumerate(zip(metrics, ylabels)):
        ax = axes[1, col]
        for key in group_keys:
            m = group_curves[metric][key]["mean"]
            s = group_curves[metric][key]["std"]
            ax.plot(x, m, marker="o" if key != "random6" else "s",
                    linewidth=1.6, markersize=4,
                    color=colors_map[key], linestyle=styles_map[key],
                    label=labels_map[key])
            ax.fill_between(x, m - s, m + s, alpha=0.10, color=colors_map[key])
        ax.set_xlabel("Layer Index")
        ax.set_ylabel(ylabel)
        ax.grid(True, alpha=0.3)

    axes[1, 0].legend(loc="upper left", fontsize=11, markerscale=1.4,
                      handlelength=2.2, borderpad=0.6, labelspacing=0.5)

    plt.tight_layout()

    for fmt, dpi in [("png", 300), ("pdf", None)]:
        out = os.path.join(d, f"drift_combined_stacked.{fmt}")
        if dpi:
            plt.savefig(out, dpi=dpi, bbox_inches="tight")
        else:
            plt.savefig(out, bbox_inches="tight")
        print(f"Saved: {out}")
    plt.close()

    print("Done")


if __name__ == "__main__":
    main()
