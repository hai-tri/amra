"""
Writer-only residual-stream drift across layers.

Produces a NeurIPS figure: applying the "full" obfuscation mode to the writer
matrices (W_O, W_down) at pertinent layers WITHOUT the compensating reader
patches or LM-head patch. Measures per-layer L2 norm of (patched - clean)
residual stream at the last prompt token, averaged over calibration prompts.

Usage
-----
    python scripts/plot_writer_only_drift.py \
        --model_path google/gemma-2b-it \
        --artifact_dir refusal_direction/pipeline/runs/gemma-2b-it \
        --out results/writer_only_drift.png

Runs on MacBook MPS or CPU. ~16 prompts is enough for a smooth curve.
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


# ------------------------------------------------------------------
# Writer-only variant of apply_obfuscation (no reader, no LM head)
# ------------------------------------------------------------------

def apply_writers_only(
    model,
    tokenize_fn,
    harmful_prompts,
    harmless_prompts,
    mean_diffs,
    selected_pos,
    direction,
    pertinent_layers,
    cfg: ObfuscationConfig,
):
    device = next(model.parameters()).device
    components = ModelComponents(model)
    d = components.d_model
    num_layers = components.num_layers

    activations = collect_calibration_activations(
        model=model,
        components=components,
        harmful_prompts=harmful_prompts,
        harmless_prompts=harmless_prompts,
        harmless_ratio=0.5,
        tokenize_fn=tokenize_fn,
        num_prompts=cfg.num_calibration_prompts,
    )

    generator = torch.Generator(device=device)
    generator.manual_seed(cfg.seed)

    attn_noise, mlp_noise = {}, {}
    for ell in sorted(pertinent_layers):
        attn_noise[ell] = generate_random_alias(d, cfg.epsilon, device, generator)
        if cfg.separate_attn_mlp_aliases:
            mlp_noise[ell] = generate_random_alias(d, cfg.epsilon, device, generator)
        else:
            mlp_noise[ell] = attn_noise[ell].clone()

    num_writers = 0
    for ell in range(num_layers):
        if ell in pertinent_layers and cfg.patch_writers in ("both", "attn_only"):
            o_proj = components.get_attn_output_proj(ell)
            x_attn = activations[f"layer_{ell}_attn_o_input"].float()
            target = attn_noise[ell].float()
            o_proj.weight.data = rank_one_update(o_proj.weight.data, x_attn, target)
            num_writers += 1

        if ell in pertinent_layers and cfg.patch_writers in ("both", "mlp_only"):
            down_proj = components.get_mlp_output_proj(ell)
            x_mlp = activations[f"layer_{ell}_mlp_down_input"].float()
            target = mlp_noise[ell].float()
            down_proj.weight.data = rank_one_update(down_proj.weight.data, x_mlp, target)
            num_writers += 1

    print(f"[writer-only] writers patched: {num_writers}")


# ------------------------------------------------------------------
# Residual stream capture (last-token)
# ------------------------------------------------------------------

def capture_residual_stream(model, tokenize_fn, prompts, components):
    """
    Returns tensor (num_prompts, num_layers + 1, d) of last-token residual stream:
    one row per layer-entry (attn_layernorm input) plus final_ln_input.
    """
    device = next(model.parameters()).device
    num_layers = components.num_layers
    d = components.d_model

    results = torch.zeros(len(prompts), num_layers + 1, d)

    def make_hook(buffer, idx):
        def hook(module, inp, output):
            x = inp[0] if isinstance(inp, tuple) else inp
            buffer[idx] = x[0, -1, :].detach().float().cpu()
        return hook

    model.eval()
    with torch.no_grad():
        for p_idx, prompt in enumerate(prompts):
            buf = torch.zeros(num_layers + 1, d)
            handles = []
            for ell in range(num_layers):
                handles.append(
                    components.get_attn_layernorm(ell).register_forward_hook(
                        make_hook(buf, ell)
                    )
                )
            handles.append(
                components.final_norm.register_forward_hook(
                    make_hook(buf, num_layers)
                )
            )

            inputs = tokenize_fn(instructions=[prompt])
            model(
                input_ids=inputs.input_ids.to(device),
                attention_mask=inputs.attention_mask.to(device),
            )

            for h in handles:
                h.remove()
            results[p_idx] = buf
            print(f"  captured {p_idx + 1}/{len(prompts)}")

    return results


# ------------------------------------------------------------------
# Main
# ------------------------------------------------------------------

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model_path", default="meta-llama/Meta-Llama-3-8B-Instruct")
    ap.add_argument("--artifact_dir",
                    default="refusal_direction/pipeline/runs/meta-llama-3-8b-instruct")
    ap.add_argument("--out", default="results/writer_only_drift.png")
    ap.add_argument("--num_eval_prompts", type=int, default=16)
    ap.add_argument("--num_calibration_prompts", type=int, default=32)
    ap.add_argument("--epsilon", type=float, default=1.0)
    ap.add_argument("--single_layer", type=int, default=None,
                    help="If set, patch only this one layer (overrides pertinent selection).")
    ap.add_argument("--with_readers", action="store_true",
                    help="Apply full obfuscation (writer + reader + LM head patches).")
    args = ap.parse_args()

    os.makedirs(os.path.dirname(args.out) or ".", exist_ok=True)

    # ---- Load artifacts ----
    direction = torch.load(
        os.path.join(args.artifact_dir, "direction.pt"), map_location="cpu"
    )
    mean_diffs = torch.load(
        os.path.join(args.artifact_dir, "generate_directions", "mean_diffs.pt"),
        map_location="cpu",
    )
    with open(os.path.join(args.artifact_dir, "direction_metadata.json")) as f:
        meta = json.load(f)
    pos = meta["pos"]

    ablation_path = os.path.join(
        args.artifact_dir, "select_direction", "direction_evaluations.json"
    )
    ablation_scores = None
    if os.path.exists(ablation_path):
        with open(ablation_path) as f:
            ablation_scores = json.load(f)

    # ---- Load datasets ----
    ds_root = os.path.join(ROOT, "refusal_direction", "dataset", "splits")
    with open(os.path.join(ds_root, "harmful_train.json")) as f:
        harmful_train = [e["instruction"] for e in json.load(f)]
    with open(os.path.join(ds_root, "harmless_train.json")) as f:
        harmless_train = [e["instruction"] for e in json.load(f)]
    with open(os.path.join(ds_root, "harmful_val.json")) as f:
        harmful_val = [e["instruction"] for e in json.load(f)]

    eval_prompts = harmful_val[: args.num_eval_prompts]

    # ---- Load model ----
    print(f"Loading model: {args.model_path}")
    model_base = construct_model_base(args.model_path)
    model = model_base.model
    tokenize_fn = model_base.tokenize_instructions_fn
    components = ModelComponents(model)

    # ---- Clean residual streams ----
    print("Capturing clean residual streams …")
    clean_resid = capture_residual_stream(model, tokenize_fn, eval_prompts, components)

    # ---- Snapshot weights for restore (cheap: only writer weights change) ----
    # We'll reload from scratch instead to be safe.
    original_state = {
        name: p.detach().clone().cpu()
        for name, p in model.state_dict().items()
    }

    # ---- Pick pertinent layers ----
    from apply_obfuscation import select_pertinent_layers
    cfg = ObfuscationConfig()
    cfg.projection_mode = "full"
    cfg.epsilon = args.epsilon
    cfg.num_calibration_prompts = args.num_calibration_prompts

    if args.single_layer is not None:
        pertinent = {args.single_layer}
        print(f"Single-layer mode: patching only layer {args.single_layer}")
    else:
        pertinent = set(select_pertinent_layers(
            mean_diffs, pos,
            k=cfg.num_pertinent_layers,
            ablation_scores=ablation_scores,
        ))
        print(f"Pertinent layers: {sorted(pertinent)}")

    # ---- Apply patches ----
    if args.with_readers:
        from apply_obfuscation import apply_obfuscation
        apply_obfuscation(
            model=model,
            tokenize_fn=tokenize_fn,
            harmful_prompts=harmful_train,
            harmless_prompts=harmless_train,
            mean_diffs=mean_diffs,
            selected_pos=pos,
            selected_layer=meta["layer"],
            direction=direction,
            cfg=cfg,
            explicit_layers=sorted(pertinent),
        )
    else:
        apply_writers_only(
            model=model,
            tokenize_fn=tokenize_fn,
            harmful_prompts=harmful_train,
            harmless_prompts=harmless_train,
            mean_diffs=mean_diffs,
            selected_pos=pos,
            direction=direction,
            pertinent_layers=pertinent,
            cfg=cfg,
        )

    # ---- Patched residual streams ----
    print("Capturing patched residual streams …")
    patched_resid = capture_residual_stream(model, tokenize_fn, eval_prompts, components)

    # ---- Restore weights (tidy up) ----
    model.load_state_dict(original_state)

    # ---- Compute per-layer L2 drift ----
    diff = (patched_resid - clean_resid).float()  # (P, L+1, d)
    l2_per_prompt = diff.norm(dim=-1)              # (P, L+1)
    l2_mean = l2_per_prompt.mean(dim=0).numpy()    # (L+1,)
    l2_std = l2_per_prompt.std(dim=0).numpy()

    # Clean-stream norm for reference / relative drift
    clean_norm = clean_resid.float().norm(dim=-1).mean(dim=0).numpy()

    # ---- Plot ----
    num_layers = components.num_layers
    x = np.arange(num_layers + 1)
    xtick_label = [str(i) for i in range(num_layers)] + ["final"]

    fig, ax = plt.subplots(figsize=(7, 4))
    ax.plot(x, l2_mean, marker="o", linewidth=2, color="#b5179e",
            label=r"$\|h^{\mathrm{patched}}_\ell - h^{\mathrm{clean}}_\ell\|_2$")
    ax.fill_between(x, l2_mean - l2_std, l2_mean + l2_std, alpha=0.15, color="#b5179e")

    for ell in sorted(pertinent):
        ax.axvline(ell, color="gray", linestyle="--", alpha=0.4, linewidth=0.8)
    ax.axvline(sorted(pertinent)[0], color="gray", linestyle="--",
               alpha=0.4, linewidth=0.8, label="selected layer")

    ax.set_xlabel("Layer Index")
    ax.set_ylabel(r"$\|h^{\mathrm{patched}} - h^{\mathrm{clean}}\|_2$")
    ax.set_title("Residual Stream Differences")
    ax.grid(True, alpha=0.3)
    ax.legend(loc="upper left")
    ax.set_xticks(x[::max(1, len(x) // 12)])
    plt.tight_layout()
    plt.savefig(args.out, dpi=180)
    print(f"Saved: {args.out}")

    # Also dump raw numbers for the paper
    np.savez(
        os.path.splitext(args.out)[0] + ".npz",
        l2_mean=l2_mean, l2_std=l2_std, clean_norm=clean_norm,
        pertinent_layers=np.array(sorted(pertinent)),
    )


if __name__ == "__main__":
    main()
