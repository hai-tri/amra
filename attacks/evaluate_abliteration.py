"""
Evaluate the defended model's resistance to standard abliteration.

This module re-runs the Arditi et al. difference-in-means attack on a model
that has already been defended with ``apply_obfuscation``, then measures:

  1. **Cosine similarity** between the refusal direction found on the defended
     model and the original (pre-defense) refusal direction.  Near 0 indicates
     the defense successfully obfuscated the direction in the residual stream.

  2. **Post-abliteration refusal rate** — after the attacker projects out the
     direction they extracted from the defended model, does the model still
     refuse harmful prompts?
"""

import torch
import sys
import os
from typing import Dict, List, Optional

# Allow imports from the existing refusal_direction package
_REFUSAL_DIR = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "refusal_direction")
if _REFUSAL_DIR not in sys.path:
    sys.path.insert(0, _REFUSAL_DIR)

from pipeline.submodules.generate_directions import get_mean_activations
from pipeline.utils.hook_utils import (
    add_hooks,
    get_direction_ablation_input_pre_hook,
    get_direction_ablation_output_hook,
)
from pipeline.submodules.select_direction import get_refusal_scores


# ------------------------------------------------------------------
# Step 1: Re-extract refusal direction from the defended model
# ------------------------------------------------------------------

def extract_refusal_direction(
    model,
    tokenizer,
    tokenize_fn,
    block_modules,
    harmful_prompts: List[str],
    benign_prompts: List[str],
    positions: List[int] = [-1],
    batch_size: int = 128,
    base_fwd_pre_hooks: Optional[List] = None,
    base_fwd_hooks: Optional[List] = None,
) -> Dict:
    """
    Run difference-in-means on the (possibly defended) model and return
    per-layer refusal directions plus the best global direction.

    ``base_fwd_pre_hooks`` / ``base_fwd_hooks`` are the defense's
    inference-time hooks (empty for weight-modifying defenses, non-empty
    for hook-based defenses like surgical / CAST / AlphaSteer).  They are
    applied during the diff-in-means collection so the extraction probes
    the *defended* model, not the bare weights.
    """
    base_fwd_pre_hooks = base_fwd_pre_hooks or []
    base_fwd_hooks = base_fwd_hooks or []

    with add_hooks(base_fwd_pre_hooks, base_fwd_hooks):
        mean_harmful = get_mean_activations(
            model, tokenizer, harmful_prompts, tokenize_fn,
            block_modules, batch_size=batch_size, positions=positions,
        )
        mean_benign = get_mean_activations(
            model, tokenizer, benign_prompts, tokenize_fn,
            block_modules, batch_size=batch_size, positions=positions,
        )

    # mean_diffs: (n_positions, n_layers, d_model)
    mean_diffs = mean_harmful - mean_benign
    n_pos, n_layers, d_model = mean_diffs.shape

    # Per-layer magnitudes (at the selected position, default -1)
    magnitudes = mean_diffs[-1].norm(dim=-1)  # (n_layers,)
    best_layer = magnitudes.argmax().item()
    best_direction = mean_diffs[-1, best_layer].float()
    best_direction = best_direction / best_direction.norm()

    return {
        "mean_diffs": mean_diffs.float(),
        "magnitudes": magnitudes.float(),
        "best_layer": best_layer,
        "best_direction": best_direction,
    }


# ------------------------------------------------------------------
# Step 2: Run abliteration and measure refusal survival
# ------------------------------------------------------------------

def evaluate_abliteration_resistance(
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
    batch_size: int = 128,
    pertinent_layers: List[int] = None,
    base_fwd_pre_hooks: Optional[List] = None,
    base_fwd_hooks: Optional[List] = None,
) -> Dict:
    """
    Full abliteration-resistance evaluation.

    Parameters
    ----------
    original_direction : Tensor
        The refusal direction extracted from the **undefended** model
        (used for the cosine-similarity diagnostic).
    pertinent_layers : list of int, optional
        Layer indices where the defense was applied. If provided,
        ``mean_cos_sim`` is computed only over these layers.

    Returns
    -------
    dict with keys:
        * ``cos_similarities`` — per-layer cosine sim between original and
          defended refusal directions.
        * ``max_cos_sim`` / ``mean_cos_sim``
        * ``defended_direction`` — the direction an attacker would extract.
        * ``baseline_refusal_score`` — mean refusal score before any abliteration.
        * ``post_abliteration_refusal_score`` — mean refusal score after
          the attacker projects out the direction they found.
    """
    base_fwd_pre_hooks = base_fwd_pre_hooks or []
    base_fwd_hooks = base_fwd_hooks or []

    num_layers = model.config.num_hidden_layers
    device = next(model.parameters()).device

    # --- Extract direction from defended model ---
    result = extract_refusal_direction(
        model, tokenizer, tokenize_fn, block_modules,
        harmful_prompts, benign_prompts,
        batch_size=batch_size,
        base_fwd_pre_hooks=base_fwd_pre_hooks,
        base_fwd_hooks=base_fwd_hooks,
    )
    defended_direction = result["best_direction"].to(device)
    defended_magnitudes = result["magnitudes"]

    # --- Cosine similarity diagnostic ---
    original_direction = original_direction.float().to("cpu")
    mean_diffs_defended = result["mean_diffs"][-1]  # (n_layers, d_model)
    o_norm = original_direction / (original_direction.norm() + 1e-8)

    cos_sims = torch.zeros(num_layers)
    for ell in range(num_layers):
        d_ell = mean_diffs_defended[ell].float().to("cpu")
        d_ell_norm = d_ell / (d_ell.norm() + 1e-8)
        cos_sims[ell] = (d_ell_norm @ o_norm).item()

    print(f"[abliteration eval] Cosine similarities with original direction:")
    print(f"  max  = {cos_sims.abs().max().item():.4f}")
    print(f"  mean = {cos_sims.abs().mean().item():.4f}")
    print(f"  Per-layer magnitudes (defended): "
          f"max={defended_magnitudes.max().item():.4f} "
          f"at layer {defended_magnitudes.argmax().item()}")

    # --- Baseline refusal score (no attack intervention, defense still on) ---
    baseline_scores = get_refusal_scores(
        model, harmful_prompts, tokenize_fn, refusal_toks,
        fwd_pre_hooks=list(base_fwd_pre_hooks),
        fwd_hooks=list(base_fwd_hooks),
        batch_size=batch_size,
    )
    baseline_mean = baseline_scores.mean().item()
    print(f"  Baseline refusal score (no intervention): {baseline_mean:.4f}")

    # --- Post-abliteration refusal score ---
    # Attacker projects out the direction they extracted from the defended
    # model.  The defense's own hooks (if any) must remain active alongside
    # the attack hooks, otherwise we would be attacking the bare weights.
    ablation_dir = defended_direction
    attack_fwd_pre_hooks = [
        (block_modules[ell],
         get_direction_ablation_input_pre_hook(direction=ablation_dir))
        for ell in range(num_layers)
    ]
    attack_fwd_hooks = [
        (attn_modules[ell],
         get_direction_ablation_output_hook(direction=ablation_dir))
        for ell in range(num_layers)
    ]
    attack_fwd_hooks += [
        (mlp_modules[ell],
         get_direction_ablation_output_hook(direction=ablation_dir))
        for ell in range(num_layers)
    ]

    post_abl_scores = get_refusal_scores(
        model, harmful_prompts, tokenize_fn, refusal_toks,
        fwd_pre_hooks=list(base_fwd_pre_hooks) + attack_fwd_pre_hooks,
        fwd_hooks=list(base_fwd_hooks) + attack_fwd_hooks,
        batch_size=batch_size,
    )
    post_abl_mean = post_abl_scores.mean().item()
    print(f"  Arditi post-attack refusal score: {post_abl_mean:.4f}")

    # mean_cos_sim over pertinent layers only (where defense was applied)
    if pertinent_layers is not None and len(pertinent_layers) > 0:
        pertinent_cos = cos_sims.abs()[pertinent_layers]
        mean_cos = pertinent_cos.mean().item()
        print(f"  mean (pertinent layers {pertinent_layers}) = {mean_cos:.4f}")
    else:
        mean_cos = cos_sims.abs().mean().item()

    return {
        "cos_similarities": cos_sims,
        "max_cos_sim": cos_sims.abs().max().item(),
        "mean_cos_sim": mean_cos,
        "defended_direction": defended_direction.cpu(),
        "defended_magnitudes": defended_magnitudes,
        "baseline_refusal_score": baseline_mean,
        "arditi_refusal_score": post_abl_mean,
    }
