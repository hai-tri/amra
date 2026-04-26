"""
Adaptive attacks against the obfuscation defense.

Three attack strategies:

1. **PCA multi-direction attack** — remove the top-k principal components of
   the harmful-vs-benign difference, not just the single best direction.

2. **Per-layer adaptive attack** — instead of using a single global direction,
   extract and ablate a *per-layer* refusal direction from the residual stream.

3. **Sublayer probing attack** — hook into sublayer internals (e.g. before
   LayerNorm, inside attention) to look for the refusal signal that the defense
   hid from the residual stream.
"""

import torch
import sys
import os
from typing import Dict, List, Optional

_REFUSAL_DIR = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "refusal_direction")
if _REFUSAL_DIR not in sys.path:
    sys.path.insert(0, _REFUSAL_DIR)

from pipeline.submodules.generate_directions import get_mean_activations
from pipeline.submodules.select_direction import get_refusal_scores
from pipeline.utils.hook_utils import (
    add_hooks,
    get_direction_ablation_input_pre_hook,
    get_direction_ablation_output_hook,
)


# ======================================================================
# Attack 1: PCA multi-direction abliteration
# ======================================================================

def pca_multi_direction_attack(
    model,
    tokenizer,
    tokenize_fn,
    block_modules,
    attn_modules,
    mlp_modules,
    harmful_prompts: List[str],
    benign_prompts: List[str],
    refusal_toks,
    top_k: int = 8,
    batch_size: int = 128,
    base_fwd_pre_hooks: Optional[List] = None,
    base_fwd_hooks: Optional[List] = None,
) -> Dict:
    """
    Extract the top-*k* principal components of the harmful-vs-benign
    activation difference and project all of them out simultaneously.

    This tests whether the refusal signal leaked into multiple directions
    that PCA can recover.

    ``base_fwd_pre_hooks`` / ``base_fwd_hooks`` are the defense's hooks;
    pass them for hook-based defenses so both the probe (mean-activation
    collection) and the attack-time refusal scoring run on the defended
    model.  They're empty no-ops for weight-modifying defenses.
    """
    base_fwd_pre_hooks = base_fwd_pre_hooks or []
    base_fwd_hooks = base_fwd_hooks or []

    num_layers = model.config.num_hidden_layers
    device = next(model.parameters()).device

    with add_hooks(base_fwd_pre_hooks, base_fwd_hooks):
        mean_harmful = get_mean_activations(
            model, tokenizer, harmful_prompts, tokenize_fn,
            block_modules, batch_size=batch_size, positions=[-1],
        )
        mean_benign = get_mean_activations(
            model, tokenizer, benign_prompts, tokenize_fn,
            block_modules, batch_size=batch_size, positions=[-1],
        )
    # mean_diffs: (1, n_layers, d_model) — squeeze position dim
    mean_diffs = (mean_harmful - mean_benign).squeeze(0)  # (n_layers, d_model)

    # PCA over the layer axis: each layer contributes a d_model-dimensional
    # "observation", and we want the top-k principal components.
    mean_diffs_centered = mean_diffs - mean_diffs.mean(dim=0, keepdim=True)
    U, S, Vh = torch.linalg.svd(mean_diffs_centered.float(), full_matrices=False)
    top_k_directions = Vh[:top_k]  # (top_k, d_model)

    print(f"[PCA attack] Top-{top_k} singular values: "
          f"{S[:top_k].tolist()}")

    # Project out all top-k directions from the residual stream
    def multi_direction_ablation_pre_hook(module, inp):
        if isinstance(inp, tuple):
            activation = inp[0]
        else:
            activation = inp
        for i in range(top_k):
            d = top_k_directions[i].to(activation)
            d = d / (d.norm() + 1e-8)
            activation = activation - (activation @ d).unsqueeze(-1) * d
        if isinstance(inp, tuple):
            return (activation, *inp[1:])
        return activation

    def multi_direction_ablation_hook(module, inp, output):
        if isinstance(output, tuple):
            activation = output[0]
        else:
            activation = output
        for i in range(top_k):
            d = top_k_directions[i].to(activation)
            d = d / (d.norm() + 1e-8)
            activation = activation - (activation @ d).unsqueeze(-1) * d
        if isinstance(output, tuple):
            return (activation, *output[1:])
        return activation

    attack_fwd_pre_hooks = [
        (block_modules[ell], multi_direction_ablation_pre_hook)
        for ell in range(num_layers)
    ]
    attack_fwd_hooks = [
        (attn_modules[ell], multi_direction_ablation_hook)
        for ell in range(num_layers)
    ] + [
        (mlp_modules[ell], multi_direction_ablation_hook)
        for ell in range(num_layers)
    ]

    post_scores = get_refusal_scores(
        model, harmful_prompts, tokenize_fn, refusal_toks,
        fwd_pre_hooks=list(base_fwd_pre_hooks) + attack_fwd_pre_hooks,
        fwd_hooks=list(base_fwd_hooks) + attack_fwd_hooks,
        batch_size=batch_size,
    )
    mean_score = post_scores.mean().item()
    print(f"  Post PCA-{top_k} abliteration refusal score: {mean_score:.4f}")

    return {
        "top_k": top_k,
        "singular_values": S[:top_k].tolist(),
        "post_attack_refusal_score": mean_score,
    }


# ======================================================================
# Attack 2: Per-layer adaptive abliteration
# ======================================================================

def per_layer_adaptive_attack(
    model,
    tokenizer,
    tokenize_fn,
    block_modules,
    attn_modules,
    mlp_modules,
    harmful_prompts: List[str],
    benign_prompts: List[str],
    refusal_toks,
    batch_size: int = 128,
    base_fwd_pre_hooks: Optional[List] = None,
    base_fwd_hooks: Optional[List] = None,
) -> Dict:
    """
    Instead of using a single global direction, extract and ablate a
    *per-layer* refusal direction.  At each layer the attacker uses the
    local difference-in-means direction for ablation.

    This is strictly stronger than the standard single-direction attack.
    ``base_fwd_pre_hooks`` / ``base_fwd_hooks`` carry the defense's hooks
    (empty for weight-modifying defenses).
    """
    base_fwd_pre_hooks = base_fwd_pre_hooks or []
    base_fwd_hooks = base_fwd_hooks or []

    num_layers = model.config.num_hidden_layers

    with add_hooks(base_fwd_pre_hooks, base_fwd_hooks):
        mean_harmful = get_mean_activations(
            model, tokenizer, harmful_prompts, tokenize_fn,
            block_modules, batch_size=batch_size, positions=[-1],
        )
        mean_benign = get_mean_activations(
            model, tokenizer, benign_prompts, tokenize_fn,
            block_modules, batch_size=batch_size, positions=[-1],
        )
    mean_diffs = (mean_harmful - mean_benign).squeeze(0)  # (n_layers, d_model)

    per_layer_dirs = []
    magnitudes = []
    for ell in range(num_layers):
        d = mean_diffs[ell].float()
        mag = d.norm().item()
        magnitudes.append(mag)
        per_layer_dirs.append(d / (d.norm() + 1e-8))

    # Build attack hooks: each layer uses its own local direction
    attack_fwd_pre_hooks = [
        (block_modules[ell],
         get_direction_ablation_input_pre_hook(direction=per_layer_dirs[ell]))
        for ell in range(num_layers)
    ]
    attack_fwd_hooks = [
        (attn_modules[ell],
         get_direction_ablation_output_hook(direction=per_layer_dirs[ell]))
        for ell in range(num_layers)
    ] + [
        (mlp_modules[ell],
         get_direction_ablation_output_hook(direction=per_layer_dirs[ell]))
        for ell in range(num_layers)
    ]

    post_scores = get_refusal_scores(
        model, harmful_prompts, tokenize_fn, refusal_toks,
        fwd_pre_hooks=list(base_fwd_pre_hooks) + attack_fwd_pre_hooks,
        fwd_hooks=list(base_fwd_hooks) + attack_fwd_hooks,
        batch_size=batch_size,
    )
    mean_score = post_scores.mean().item()
    print(f"[per-layer attack] Post per-layer abliteration refusal score: "
          f"{mean_score:.4f}")

    return {
        "per_layer_magnitudes": magnitudes,
        "post_attack_refusal_score": mean_score,
    }


# ======================================================================
# Attack 3: Sublayer probing
# ======================================================================

def sublayer_probing_attack(
    model,
    tokenizer,
    tokenize_fn,
    harmful_prompts: List[str],
    benign_prompts: List[str],
    original_direction: torch.Tensor,
    batch_size: int = 128,
    base_fwd_pre_hooks: Optional[List] = None,
    base_fwd_hooks: Optional[List] = None,
) -> Dict:
    """
    Probe *inside* sublayer boundaries (attention Q/K/V inputs, MLP gate/up
    inputs) to see whether the refusal signal is detectable before the
    defense's reader patches neutralise it.

    Returns cosine-similarity diagnostics at each sublayer probe point.
    ``base_fwd_pre_hooks`` / ``base_fwd_hooks`` carry the defense's hooks
    (empty for weight-modifying defenses).
    """
    from obfuscation_utils import ModelComponents

    base_fwd_pre_hooks = base_fwd_pre_hooks or []
    base_fwd_hooks = base_fwd_hooks or []

    components = ModelComponents(model)
    num_layers = components.num_layers
    device = next(model.parameters()).device

    # Collect mean activations at sublayer boundaries for harmful vs benign
    probe_points = {}  # key -> accumulated (harmful, benign, count)

    def _make_accum_hook(key: str, harmful: bool):
        def hook_fn(module, inp, output):
            x = inp[0] if isinstance(inp, tuple) else inp
            vec = x[0, -1, :].detach().float().cpu()
            if key not in probe_points:
                probe_points[key] = {"harmful": [], "benign": []}
            side = "harmful" if harmful else "benign"
            probe_points[key][side].append(vec)
        return hook_fn

    for is_harmful, prompts in [(True, harmful_prompts), (False, benign_prompts)]:
        with add_hooks(base_fwd_pre_hooks, base_fwd_hooks):
            hooks = []
            try:
                for ell in range(num_layers):
                    hooks.append(
                        components.get_attn_layernorm(ell).register_forward_hook(
                            _make_accum_hook(f"attn_ln_{ell}", is_harmful)
                        )
                    )
                    hooks.append(
                        components.get_mlp_layernorm(ell).register_forward_hook(
                            _make_accum_hook(f"mlp_ln_{ell}", is_harmful)
                        )
                    )
                with torch.no_grad():
                    for prompt in prompts[:batch_size]:
                        inputs = tokenize_fn(instructions=[prompt])
                        model(
                            input_ids=inputs.input_ids.to(device),
                            attention_mask=inputs.attention_mask.to(device),
                        )
            finally:
                for h in hooks:
                    h.remove()

    # Compute cosine similarities at each probe point
    original_direction = original_direction.float().cpu()
    original_direction = original_direction / (original_direction.norm() + 1e-8)

    cos_sims = {}
    for key, data in probe_points.items():
        mean_h = torch.stack(data["harmful"]).mean(dim=0)
        mean_b = torch.stack(data["benign"]).mean(dim=0)
        diff = mean_h - mean_b
        diff_normed = diff / (diff.norm() + 1e-8)
        cs = (diff_normed @ original_direction).item()
        cos_sims[key] = cs

    # Report
    attn_sims = [cos_sims[f"attn_ln_{ell}"] for ell in range(num_layers)]
    mlp_sims = [cos_sims[f"mlp_ln_{ell}"] for ell in range(num_layers)]

    print(f"[sublayer probing] Cosine sim with original direction:")
    print(f"  Attn LN inputs — max={max(abs(s) for s in attn_sims):.4f}, "
          f"mean={sum(abs(s) for s in attn_sims)/len(attn_sims):.4f}")
    print(f"  MLP  LN inputs — max={max(abs(s) for s in mlp_sims):.4f}, "
          f"mean={sum(abs(s) for s in mlp_sims)/len(mlp_sims):.4f}")

    return {
        "attn_ln_cos_sims": attn_sims,
        "mlp_ln_cos_sims": mlp_sims,
    }


# ======================================================================
# Convenience: run all adaptive attacks
# ======================================================================

def run_all_adaptive_attacks(
    model,
    tokenizer,
    tokenize_fn,
    block_modules,
    attn_modules,
    mlp_modules,
    harmful_prompts: List[str],
    benign_prompts: List[str],
    original_direction: torch.Tensor,
    refusal_toks,
    pca_top_k: int = 8,
    batch_size: int = 128,
    base_fwd_pre_hooks: Optional[List] = None,
    base_fwd_hooks: Optional[List] = None,
) -> Dict:
    base_fwd_pre_hooks = base_fwd_pre_hooks or []
    base_fwd_hooks = base_fwd_hooks or []

    results = {}

    print("\n=== PCA Multi-Direction Attack ===")
    results["pca"] = pca_multi_direction_attack(
        model, tokenizer, tokenize_fn,
        block_modules, attn_modules, mlp_modules,
        harmful_prompts, benign_prompts, refusal_toks,
        top_k=pca_top_k, batch_size=batch_size,
        base_fwd_pre_hooks=base_fwd_pre_hooks,
        base_fwd_hooks=base_fwd_hooks,
    )

    print("\n=== Per-Layer Adaptive Attack ===")
    results["per_layer"] = per_layer_adaptive_attack(
        model, tokenizer, tokenize_fn,
        block_modules, attn_modules, mlp_modules,
        harmful_prompts, benign_prompts, refusal_toks,
        batch_size=batch_size,
        base_fwd_pre_hooks=base_fwd_pre_hooks,
        base_fwd_hooks=base_fwd_hooks,
    )

    print("\n=== Sublayer Probing Attack ===")
    results["sublayer_probing"] = sublayer_probing_attack(
        model, tokenizer, tokenize_fn,
        harmful_prompts, benign_prompts,
        original_direction, batch_size=batch_size,
        base_fwd_pre_hooks=base_fwd_pre_hooks,
        base_fwd_hooks=base_fwd_hooks,
    )

    return results
