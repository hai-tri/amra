"""
Single-writer residual-stream drift across an epsilon sweep.

Patches ONE weight matrix (W_O or W_down) at the selected layer using the
current rank-k PCA writer update. Optionally adds rank-k reader and LM-head
correction to measure how the compensated residual stream changes.
Saves per-metric .npz files compatible with plot_drift_combined.py.
"""

import argparse
import json
import os
import sys
import tempfile
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import torch

ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.insert(0, ROOT)
sys.path.insert(0, os.path.join(ROOT, "refusal_direction"))
sys.path.insert(0, os.path.join(ROOT, "scripts", "eval"))

from apply_obfuscation import apply_obfuscation
from obfuscation_config import ObfuscationConfig
from obfuscation_utils import (
    ModelComponents,
    collect_writer_output_refusal_directions,
    collect_writer_output_refusal_subspaces,
    generate_random_alias,
)
from pipeline.model_utils.model_factory import construct_model_base
from pipeline.submodules.generate_directions import generate_directions
from pipeline.submodules.select_direction import select_direction
from run_obfuscation_pipeline import filter_data, load_mlabonne_datasets


def _rank_k_writer_update(W, directions, aliases):
    """W_new = W + (A - R)^T (R W)  —  redirects refusal subspace to aliases."""
    orig_dtype = W.dtype
    W_f = W.float()
    r = directions.float().to(W.device)
    a = aliases.float().to(W.device)
    r = r / (r.norm(dim=-1, keepdim=True) + 1e-8)
    coeff_rows = r @ W_f
    return (W_f + (a - r).T @ coeff_rows).to(orig_dtype)


def patch_single_writer(model, components, layer, writer, directions, epsilon, seed):
    """Apply rank-k writer update to ONE matrix at `layer`."""
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
    """Returns list of per-prompt tensors, each shape (num_layers+1, T_p, d)."""
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
    """Returns per-prompt (P, L+1) tensor of the requested metric."""
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
        else:
            raise ValueError(metric)
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", default="llama",
                    help="Key into CONFIGS (llama, gemma, qwen) or a HF model ID")
    ap.add_argument("--writer", default="attn", choices=["attn", "mlp"],
                    help="Which single writer matrix to patch (W_O or W_down)")
    ap.add_argument("--layer", type=int, default=None,
                    help="Layer index for the single writer matrix. Default: use "
                         "the layer selected by refusal_direction artifacts.")
    ap.add_argument("--epsilons", type=float, nargs="+",
                    default=[0.025, 0.1, 0.3, 1.0])
    ap.add_argument("--k_w", type=int, default=1,
                    help="num_writer_directions for rank-k update")
    ap.add_argument("--k_r", type=int, default=None,
                    help="num_reader_directions when --with_readers is set. "
                         "Default: match --k_w.")
    ap.add_argument("--num_eval_prompts", type=int, default=16)
    ap.add_argument("--num_calibration_prompts", type=int, default=64)
    ap.add_argument("--forward_batch_size", type=int, default=16)
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--metrics", nargs="+", choices=["l2", "var", "cos"],
                    default=["l2", "var", "cos"])
    ap.add_argument("--out_dir", default="results/drift_sweep")
    ap.add_argument("--artifact_dir", default=None,
                    help="Path to pre-computed refusal_direction run artifacts. "
                         "Skips generate_directions + select_direction.")
    ap.add_argument("--with_readers", action="store_true",
                    help="Also apply reader patches (and LM head) after the writer "
                         "update, using rank-k reader correction.")
    args = ap.parse_args()
    reader_rank = args.k_w if args.k_r is None else args.k_r
    if args.k_w < 1:
        raise ValueError("--k_w must be >= 1")
    if reader_rank < 1:
        raise ValueError("--k_r must be >= 1")

    os.makedirs(args.out_dir, exist_ok=True)

    # ── Resolve model ID ────────────────────────────────────────────────────
    try:
        from quick_attack_test import CONFIGS
        model_id = CONFIGS[args.model][0] if args.model in CONFIGS else args.model
    except ImportError:
        model_id = args.model

    print(f"[drift] model={model_id}  writer={args.writer}  k_w={args.k_w}"
          f"  k_r={reader_rank if args.with_readers else 'n/a'}")
    print(f"[drift] epsilons={args.epsilons}")

    # ── Load model ──────────────────────────────────────────────────────────
    model_base = construct_model_base(model_id)

    if "qwen3" in model_id.lower():
        import functools
        tok = model_base.tokenizer
        orig = tok.apply_chat_template
        def _no_think(messages, **kw):
            kw.setdefault("enable_thinking", False)
            return orig(messages, **kw)
        tok.apply_chat_template = _no_think

    components = ModelComponents(model_base.model)

    # ── Data + refusal direction ─────────────────────────────────────────────
    if args.artifact_dir:
        adir = Path(args.artifact_dir)
        print(f"[drift] loading artifacts from {adir}")
        with open(adir / "direction_metadata.json") as f:
            meta = json.load(f)
        pos, layer = meta["pos"], meta["layer"]
        mean_diffs_train = torch.load(
            adir / "generate_directions" / "mean_diffs.pt", map_location="cpu"
        )
        direction = torch.load(adir / "direction.pt", map_location="cpu")
        with open(adir / "select_direction" / "direction_evaluations.json") as f:
            ablation_scores = json.load(f)
        print(f"[drift] layer={layer}  pos={pos}")

        splits_dir = Path(ROOT) / "refusal_direction" / "dataset" / "splits"
        with open(splits_dir / "harmful_train.json") as f:
            harmful_train = [e["instruction"] for e in json.load(f)]
        with open(splits_dir / "harmless_train.json") as f:
            harmless_train = [e["instruction"] for e in json.load(f)]
        with open(splits_dir / "harmful_val.json") as f:
            eval_prompts = [e["instruction"] for e in json.load(f)][: args.num_eval_prompts]
    else:
        harmful_train, harmless_train, harmful_val, harmless_val = load_mlabonne_datasets(
            n_train=400, n_val=100,
        )
        print("[drift] filtering data ...")
        harmful_train, harmless_train, harmful_val, harmless_val = filter_data(
            model_base, harmful_train, harmless_train, harmful_val, harmless_val,
        )
        eval_prompts = harmful_val[: args.num_eval_prompts]

        print("[drift] extracting refusal direction ...")
        with tempfile.TemporaryDirectory() as tmp:
            mean_diffs_train = generate_directions(
                model_base, harmful_train, harmless_train, artifact_dir=tmp,
            )
            pos, layer, direction = select_direction(
                model_base, harmful_val, harmless_val, mean_diffs_train, artifact_dir=tmp,
            )
            ablation_path = Path(tmp) / "direction_evaluations.json"
            if ablation_path.exists():
                with open(ablation_path) as f:
                    ablation_scores = json.load(f)
            else:
                ablation_scores = None
        print(f"[drift] selected layer={layer}  pos={pos}")

    if args.layer is not None:
        if args.layer < 0 or args.layer >= components.num_layers:
            raise ValueError(
                f"--layer must be in [0, {components.num_layers - 1}], got {args.layer}"
            )
        print(f"[drift] overriding selected layer {layer} -> {args.layer}")
        layer = args.layer

    # ── Extract writer-output refusal directions at the selected layer ────────
    print(f"[drift] collecting writer-output refusal directions (layer={layer}) ...")
    if args.k_w > 1:
        subspaces = collect_writer_output_refusal_subspaces(
            model=model_base.model,
            components=components,
            harmful_prompts=harmful_train,
            harmless_prompts=harmless_train,
            tokenize_fn=model_base.tokenize_instructions_fn,
            num_prompts=args.num_calibration_prompts,
            num_directions=args.k_w,
            layers=[layer],
            forward_batch_size=args.forward_batch_size,
        )
        writer_dirs = subspaces[args.writer][layer]   # (k_w, d_model)
    else:
        dirs = collect_writer_output_refusal_directions(
            model=model_base.model,
            components=components,
            harmful_prompts=harmful_train,
            harmless_prompts=harmless_train,
            tokenize_fn=model_base.tokenize_instructions_fn,
            num_prompts=args.num_calibration_prompts,
            forward_batch_size=args.forward_batch_size,
        )
        d_vec = dirs[args.writer][layer]              # (d_model,)
        writer_dirs = (d_vec / (d_vec.norm() + 1e-8)).unsqueeze(0)  # (1, d_model)

    print(f"[drift] writer directions shape: {writer_dirs.shape}")

    # ── Snapshot clean weights ───────────────────────────────────────────────
    print("[drift] snapshotting clean weights ...")
    clean_state = {k: v.detach().clone().cpu()
                   for k, v in model_base.model.state_dict().items()}

    # ── Clean residual streams ───────────────────────────────────────────────
    print("[drift] capturing clean residual streams ...")
    clean_resid = capture_residual_stream(
        model_base.model, model_base.tokenize_instructions_fn,
        eval_prompts, components,
    )

    # ── Sweep epsilons ───────────────────────────────────────────────────────
    curves = {m: {} for m in args.metrics}

    for eps in args.epsilons:
        print(f"\n[drift] === epsilon={eps} ===")
        model_base.model.load_state_dict(clean_state)

        if args.with_readers:
            cfg = ObfuscationConfig(
                epsilon=eps,
                num_calibration_prompts=args.num_calibration_prompts,
                seed=args.seed,
                projection_mode="full",
                per_layer_direction=True,
                writer_output_directions=True,
                num_writer_directions=args.k_w,
                num_reader_directions=reader_rank,
                force_subspace_writer_update=True,
                forward_batch_size=args.forward_batch_size,
                patch_writers="attn_only" if args.writer == "attn" else "mlp_only",
            )
            apply_obfuscation(
                model=model_base.model,
                tokenize_fn=model_base.tokenize_instructions_fn,
                harmful_prompts=harmful_train,
                harmless_prompts=harmless_train,
                mean_diffs=mean_diffs_train,
                selected_pos=pos,
                selected_layer=layer,
                direction=direction,
                cfg=cfg,
                ablation_scores=ablation_scores,
                explicit_layers=[layer],
                writer_only=False,
            )
        else:
            patch_single_writer(
                model_base.model, components, layer,
                args.writer, writer_dirs, eps, args.seed,
            )

        print(f"  capturing patched residual streams ...")
        patched = capture_residual_stream(
            model_base.model, model_base.tokenize_instructions_fn,
            eval_prompts, components,
        )
        for metric in args.metrics:
            vals = compute_metric(clean_resid, patched, metric)
            curves[metric][eps] = {
                "mean": vals.mean(dim=0).numpy(),
                "std":  vals.std(dim=0).numpy(),
            }

    model_base.model.load_state_dict(clean_state)

    # ── Save .npz and .png per metric ────────────────────────────────────────
    metric_labels = {
        "l2":  "L2 Differences",
        "var": "Variance Shift",
        "cos": "Cosine Similarity",
    }
    ylabels = {
        "l2":  r"$\|h_{\mathrm{patched}} - h_{\mathrm{clean}}\|_2$",
        "var": r"$\mathrm{Var}(h_{\mathrm{patched}} - h_{\mathrm{clean}})$",
        "cos": r"$\cos(h_{\mathrm{patched}},\, h_{\mathrm{clean}})$",
    }
    x = np.arange(components.num_layers + 1)
    cmap = plt.get_cmap("viridis")

    suffix = f"_{args.writer}_with_readers" if args.with_readers else f"_{args.writer}"
    for metric in args.metrics:
        npz_path = os.path.join(args.out_dir, f"drift_eps_sweep_{metric}{suffix}.npz")
        np.savez(
            npz_path,
            layer=layer,
            epsilons=np.array(args.epsilons),
            **{f"mean_{eps}": curves[metric][eps]["mean"] for eps in args.epsilons},
            **{f"std_{eps}":  curves[metric][eps]["std"]  for eps in args.epsilons},
        )
        print(f"Saved: {npz_path}")

        fig, ax = plt.subplots(figsize=(8, 4.5))
        for i, eps in enumerate(args.epsilons):
            c = cmap(i / max(1, len(args.epsilons) - 1))
            m = curves[metric][eps]["mean"]
            s = curves[metric][eps]["std"]
            ax.plot(x, m, marker="o", linewidth=1.8, color=c,
                    label=fr"$\varepsilon = {eps}$")
            ax.fill_between(x, m - s, m + s, alpha=0.12, color=c)
        ax.axvline(layer, color="gray", linestyle="--",
                   alpha=0.5, linewidth=0.8, label="selected layer")
        ax.set_xlabel("Layer Index")
        ax.set_ylabel(ylabels[metric])
        ax.set_title(f"Residual Stream {metric_labels[metric]}")
        ax.grid(True, alpha=0.3)
        ax.legend(loc="upper left")
        plt.tight_layout()
        png_path = os.path.join(args.out_dir, f"drift_eps_sweep_{metric}{suffix}.png")
        plt.savefig(png_path, dpi=180)
        plt.close()
        print(f"Saved: {png_path}")

    print("\n[drift] done — run plot_drift_combined.py to render the 1x3 figure.")


if __name__ == "__main__":
    main()
