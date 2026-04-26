"""
Writer-only residual-stream drift across an epsilon sweep.

Single-layer patch at the given layer, no reader patches, varying epsilon.
Plots all curves on one axes to show how drift scales with the alias norm.
"""

import argparse
import copy
import json
import os
import sys

import matplotlib.pyplot as plt
import numpy as np
import torch

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)
sys.path.insert(0, os.path.join(ROOT, "refusal_direction"))

from obfuscation_config import ObfuscationConfig
from obfuscation_utils import (
    ModelComponents,
    collect_calibration_activations,
    generate_random_alias,
    rank_one_update,
)
from pipeline.model_utils.model_factory import construct_model_base


def apply_writers_only(model, tokenize_fn, harmful_prompts, harmless_prompts,
                       pertinent_layers, cfg):
    device = next(model.parameters()).device
    components = ModelComponents(model)
    d = components.d_model

    activations = collect_calibration_activations(
        model=model, components=components,
        harmful_prompts=harmful_prompts, harmless_prompts=harmless_prompts,
        harmless_ratio=0.5, tokenize_fn=tokenize_fn,
        num_prompts=cfg.num_calibration_prompts,
    )
    generator = torch.Generator(device=device)
    generator.manual_seed(cfg.seed)

    attn_noise, mlp_noise = {}, {}
    for ell in sorted(pertinent_layers):
        attn_noise[ell] = generate_random_alias(d, cfg.epsilon, device, generator)
        mlp_noise[ell] = (generate_random_alias(d, cfg.epsilon, device, generator)
                         if cfg.separate_attn_mlp_aliases else attn_noise[ell].clone())

    for ell in sorted(pertinent_layers):
        o_proj = components.get_attn_output_proj(ell)
        x_attn = activations[f"layer_{ell}_attn_o_input"].float()
        o_proj.weight.data = rank_one_update(
            o_proj.weight.data, x_attn, attn_noise[ell].float()
        )
        down_proj = components.get_mlp_output_proj(ell)
        x_mlp = activations[f"layer_{ell}_mlp_down_input"].float()
        down_proj.weight.data = rank_one_update(
            down_proj.weight.data, x_mlp, mlp_noise[ell].float()
        )


def capture_residual_stream(model, tokenize_fn, prompts, components):
    """Returns list of per-prompt tensors, each shape (num_layers+1, T_p, d)."""
    device = next(model.parameters()).device
    num_layers = components.num_layers
    results = []

    def make_hook(buf, idx):
        def hook(module, inp, output):
            x = inp[0] if isinstance(inp, tuple) else inp
            buf[idx] = x[0].detach().float().cpu()  # (T, d)
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
            handles.append(components.final_norm.register_forward_hook(
                make_hook(buf, num_layers)))
            inputs = tokenize_fn(instructions=[prompt])
            model(input_ids=inputs.input_ids.to(device),
                  attention_mask=inputs.attention_mask.to(device))
            for h in handles:
                h.remove()
            results.append(torch.stack(buf, dim=0))  # (L+1, T, d)
            print(f"  captured {p_idx + 1}/{len(prompts)}")
    return results


def compute_metric(clean_list, patched_list, metric):
    """Returns per-prompt (P, L+1) tensor of the requested metric."""
    P = len(clean_list)
    Lp1 = clean_list[0].shape[0]
    out = torch.zeros(P, Lp1)
    for i, (c, p) in enumerate(zip(clean_list, patched_list)):
        diff = (p - c).float()  # (L+1, T, d)
        if metric == "l2":
            # L2 norm of per-token diff, averaged over tokens
            out[i] = diff.norm(dim=-1).mean(dim=-1)
        elif metric == "var":
            # variance across feature dim per token, averaged across tokens
            out[i] = diff.var(dim=-1, unbiased=False).mean(dim=-1)
        elif metric == "cos":
            # cosine similarity per token between patched and clean, averaged
            cf = c.float(); pf = p.float()
            num = (cf * pf).sum(dim=-1)
            den = cf.norm(dim=-1) * pf.norm(dim=-1) + 1e-12
            out[i] = (num / den).mean(dim=-1)
        else:
            raise ValueError(metric)
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model_path", default="meta-llama/Meta-Llama-3-8B-Instruct")
    ap.add_argument("--artifact_dir",
                    default="refusal_direction/pipeline/runs/meta-llama-3-8b-instruct")
    ap.add_argument("--layer", type=int, default=10)
    ap.add_argument("--epsilons", type=float, nargs="+",
                    default=[0.025, 0.1, 0.3, 1.0])
    ap.add_argument("--num_eval_prompts", type=int, default=16)
    ap.add_argument("--num_calibration_prompts", type=int, default=32)
    ap.add_argument("--out", default="results/drift_eps_sweep.png")
    ap.add_argument("--with_readers", action="store_true",
                    help="Apply full obfuscation (writer + reader + LM head).")
    ap.add_argument("--metric", choices=["l2", "var", "cos"], default="l2",
                    help="l2: ||patched-clean||_2 per token; "
                         "var: var over features of diff per token; "
                         "cos: cosine similarity per token.")
    args = ap.parse_args()

    os.makedirs(os.path.dirname(args.out) or ".", exist_ok=True)

    # ---- Datasets ----
    ds_root = os.path.join(ROOT, "refusal_direction", "dataset", "splits")
    with open(os.path.join(ds_root, "harmful_train.json")) as f:
        harmful_train = [e["instruction"] for e in json.load(f)]
    with open(os.path.join(ds_root, "harmless_train.json")) as f:
        harmless_train = [e["instruction"] for e in json.load(f)]
    with open(os.path.join(ds_root, "harmful_val.json")) as f:
        harmful_val = [e["instruction"] for e in json.load(f)]
    eval_prompts = harmful_val[: args.num_eval_prompts]

    # ---- Model (load once) ----
    print(f"Loading model: {args.model_path}")
    model_base = construct_model_base(args.model_path)
    model = model_base.model
    tokenize_fn = model_base.tokenize_instructions_fn
    components = ModelComponents(model)

    # Snapshot clean weights once for restore between epsilons
    print("Snapshotting clean weights …")
    clean_state = {k: v.detach().clone().cpu() for k, v in model.state_dict().items()}

    # ---- Clean residual streams (same for every epsilon) ----
    print("Capturing clean residual streams …")
    clean_resid = capture_residual_stream(model, tokenize_fn, eval_prompts, components)

    # ---- Sweep epsilons ----
    curves = {}
    for eps in args.epsilons:
        print(f"\n=== epsilon = {eps} ===")
        model.load_state_dict(clean_state)  # restore clean weights
        cfg = ObfuscationConfig()
        cfg.projection_mode = "full"
        cfg.epsilon = eps
        cfg.num_calibration_prompts = args.num_calibration_prompts

        if args.with_readers:
            from apply_obfuscation import apply_obfuscation
            # Need mean_diffs / direction / selected_pos for the signature
            import json as _json
            mean_diffs = torch.load(
                os.path.join(args.artifact_dir, "generate_directions", "mean_diffs.pt"),
                map_location="cpu")
            direction = torch.load(
                os.path.join(args.artifact_dir, "direction.pt"), map_location="cpu")
            with open(os.path.join(args.artifact_dir, "direction_metadata.json")) as _f:
                _meta = _json.load(_f)
            apply_obfuscation(
                model=model, tokenize_fn=tokenize_fn,
                harmful_prompts=harmful_train, harmless_prompts=harmless_train,
                mean_diffs=mean_diffs, selected_pos=_meta["pos"],
                selected_layer=_meta["layer"], direction=direction,
                cfg=cfg, explicit_layers=[args.layer],
            )
        else:
            apply_writers_only(
                model=model, tokenize_fn=tokenize_fn,
                harmful_prompts=harmful_train, harmless_prompts=harmless_train,
                pertinent_layers={args.layer}, cfg=cfg,
            )
        patched = capture_residual_stream(model, tokenize_fn, eval_prompts, components)
        vals = compute_metric(clean_resid, patched, args.metric)  # (P, L+1)
        curves[eps] = {
            "mean": vals.mean(dim=0).numpy(),
            "std":  vals.std(dim=0).numpy(),
        }

    model.load_state_dict(clean_state)  # tidy up

    # ---- Plot ----
    num_layers = components.num_layers
    x = np.arange(num_layers + 1)
    fig, ax = plt.subplots(figsize=(8, 4.5))
    cmap = plt.get_cmap("viridis")
    for i, eps in enumerate(args.epsilons):
        c = cmap(i / max(1, len(args.epsilons) - 1))
        m = curves[eps]["mean"]; s = curves[eps]["std"]
        ax.plot(x, m, marker="o", linewidth=1.8, color=c,
                label=fr"$\varepsilon = {eps}$")
        ax.fill_between(x, m - s, m + s, alpha=0.12, color=c)

    ax.axvline(args.layer, color="gray", linestyle="--",
               alpha=0.5, linewidth=0.8, label="selected layer")
    ax.set_xlabel("Layer Index")
    ylabels = {
        "l2":  r"$\|h_{\mathrm{patched}} - h_{\mathrm{clean}}\|_2$",
        "var": r"$\mathrm{Var}(h_{\mathrm{patched}} - h_{\mathrm{clean}})$",
        "cos": r"$\cos(h_{\mathrm{patched}},\, h_{\mathrm{clean}})$",
    }
    titles = {
        "l2":  "Residual Stream L2 Differences",
        "var": "Residual Stream Variance Shift",
        "cos": "Residual Stream Cosine Similarity",
    }
    ax.set_ylabel(ylabels[args.metric])
    ax.set_title(titles[args.metric])
    ax.grid(True, alpha=0.3)
    ax.legend(loc="upper left")
    plt.tight_layout()
    plt.savefig(args.out, dpi=180)
    print(f"Saved: {args.out}")

    np.savez(
        os.path.splitext(args.out)[0] + ".npz",
        layer=args.layer,
        epsilons=np.array(args.epsilons),
        **{f"mean_{eps}": curves[eps]["mean"] for eps in args.epsilons},
        **{f"std_{eps}":  curves[eps]["std"]  for eps in args.epsilons},
    )


if __name__ == "__main__":
    main()
