#!/usr/bin/env python3
"""Group layer-sweep drift with full APRS (writer + reader + LM-head)."""

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

from apply_obfuscation import apply_obfuscation
from obfuscation_config import ObfuscationConfig
from obfuscation_utils import ModelComponents
from pipeline.model_utils.model_factory import construct_model_base
from pipeline.submodules.generate_directions import generate_directions
from pipeline.submodules.select_direction import select_direction
from run_obfuscation_pipeline import filter_data, load_mlabonne_datasets
from pathlib import Path


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
        ablation_path = Path(tmp) / "direction_evaluations.json"
        ablation_scores = json.load(open(ablation_path)) if ablation_path.exists() else None
    print(f"Selected layer={layer}, pos={pos}")

    rng = np.random.default_rng(42)
    random_layers = sorted(rng.choice(32, size=6, replace=False).tolist())
    print(f"Random 6 layers: {random_layers}")

    print("Snapshotting clean weights ...")
    clean_state = {k: v.detach().clone().cpu()
                   for k, v in model_base.model.state_dict().items()}

    print("Capturing clean residual streams ...")
    clean_resid = capture_residual_stream(
        model_base.model, model_base.tokenize_instructions_fn, eval_prompts, components)

    configs = {
        "L9-16":  list(range(9, 17)),
        "L17-23": list(range(17, 24)),
        "L24-31": list(range(24, 32)),
        "random6": random_layers,
    }

    eps = 0.025
    seed = 42
    metrics_list = ["l2", "var", "cos"]
    curves = {m: {} for m in metrics_list}

    for label, layer_set in configs.items():
        print(f"\n=== {label}: layers {layer_set}, eps={eps}, with readers ===")
        model_base.model.load_state_dict(clean_state)

        cfg = ObfuscationConfig(
            epsilon=eps,
            num_calibration_prompts=64,
            seed=seed,
            projection_mode="full",
            per_layer_direction=True,
            writer_output_directions=True,
            num_writer_directions=1,
            num_reader_directions=1,
            forward_batch_size=16,
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
            explicit_layers=layer_set,
            writer_only=False,
        )

        print("  capturing patched residual streams ...")
        patched = capture_residual_stream(
            model_base.model, model_base.tokenize_instructions_fn,
            eval_prompts, components)
        for metric in metrics_list:
            vals = compute_metric(clean_resid, patched, metric)
            curves[metric][label] = {
                "mean": vals.mean(dim=0).numpy(),
                "std":  vals.std(dim=0).numpy(),
            }

    model_base.model.load_state_dict(clean_state)

    d = os.path.join(ROOT, "results", "drift_sweep")
    os.makedirs(d, exist_ok=True)
    for metric in metrics_list:
        npz_path = os.path.join(d, f"drift_group_sweep_eps0.025_{metric}_attn_with_readers.npz")
        np.savez(npz_path, epsilon=eps, random_layers=np.array(random_layers),
                 **{f"mean_{k}": curves[metric][k]["mean"] for k in configs},
                 **{f"std_{k}": curves[metric][k]["std"] for k in configs})
        print(f"Saved: {npz_path}")

    # Plots
    plt.rcParams.update({
        "font.size": 15, "axes.titlesize": 16, "axes.labelsize": 15,
        "xtick.labelsize": 15, "ytick.labelsize": 15, "legend.fontsize": 15,
    })
    ylabels = [
        r"$\|h_{\mathrm{patched}} - h_{\mathrm{clean}}\|_2$",
        r"$\mathrm{Var}(h_{\mathrm{patched}} - h_{\mathrm{clean}})$",
        r"$\cos(h_{\mathrm{patched}},\, h_{\mathrm{clean}})$",
    ]
    colors_map = {"L9-16": "#1f77b4", "L17-23": "#ff7f0e", "L24-31": "#2ca02c", "random6": "black"}
    styles_map = {"L9-16": "-", "L17-23": "-", "L24-31": "-", "random6": "--"}
    rl_str = ",".join(str(r) for r in random_layers)
    labels_map = {"L9-16": "Layers 9–16", "L17-23": "Layers 17–23",
                  "L24-31": "Layers 24–31",
                  "random6": f"Random 6 (L={rl_str})"}
    titles = ["(a) Residual Stream L2 Differences",
              "(b) Residual Stream Variance Shift",
              "(c) Residual Stream Cosine Similarity"]
    x = np.arange(33)

    for include_titles, suffix in [(True, ""), (False, "_notitle")]:
        fig, axes = plt.subplots(1, 3, figsize=(15, 4.2))
        for ax, metric, ylabel, title in zip(axes, metrics_list, ylabels, titles):
            for key in configs:
                m = curves[metric][key]["mean"]; s = curves[metric][key]["std"]
                ax.plot(x, m, marker="o" if key != "random6" else "s", linewidth=1.6,
                        markersize=4, color=colors_map[key], linestyle=styles_map[key],
                        label=labels_map[key])
                ax.fill_between(x, m - s, m + s, alpha=0.10, color=colors_map[key])
            ax.set_xlabel("Layer Index")
            ax.set_ylabel(ylabel)
            if include_titles:
                ax.set_title(title)
            ax.grid(True, alpha=0.3)
        axes[0].legend(loc="upper left", fontsize=11, markerscale=1.4,
                       handlelength=2.2, borderpad=0.6, labelspacing=0.5)
        plt.tight_layout()
        png = os.path.join(d, f"drift_group_sweep_eps0.025_with_readers_combined{suffix}.png")
        plt.savefig(png, dpi=180, bbox_inches="tight")
        plt.close()
        print(f"Saved: {png}")

    print("Done")


if __name__ == "__main__":
    main()
