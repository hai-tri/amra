"""
Representational Obfuscation of Safety Features via Per-Layer Rank-One Aliasing
===============================================================================

Core defense implementation (Steps 0–6 from the technical specification).

The algorithm:
  0. Extract the refusal direction and identify pertinent layers.
  1. Generate per-layer random alias vectors (zero-mean, ε-scaled).
  2. Patch writer matrices (W_O, W_down) at pertinent layers so the refusal
     component of each writer output at the calibration operating point becomes
     a random alias.
  3. Iterate layer-by-layer, probing the actual polluted residual stream in
     the partially-patched model via short forward passes.  This handles
     architectures (Gemma 2/3) whose post-attention / post-feedforward
     LayerNorms nonlinearly transform the writer output before the residual
     add — in such cases the net pollution cannot be accumulated analytically.
  4. Patch reader matrices (Q, K, V, gate, up) at every downstream layer so
     they compensate for the empirically-measured pollution and produce the
     same outputs as the undefended model on calibration inputs.
  5. (Implicit — handled by Step 4 for attention output projections at
     non-pertinent layers.)
  6. Probe the final residual stream and patch the unembedding matrix (LM head)
     so that the model's logits on calibration inputs match the undefended
     model.

No training is required.  This is a one-time, post-hoc weight edit.
"""

import torch
import os
import json
from typing import Dict, List, Optional, Set, Tuple

from obfuscation_config import ObfuscationConfig
from obfuscation_utils import (
    ModelComponents,
    collect_calibration_activations,
    collect_writer_output_refusal_directions,
    generate_random_alias,
    probe_residual_stream,
    rank_one_update,
)


# ------------------------------------------------------------------
# Pertinent layer selection
# ------------------------------------------------------------------

def select_pertinent_layers(
    mean_diffs: torch.Tensor,
    pos: int,
    k: Optional[int] = None,
    ablation_scores: Optional[Dict] = None,
    ablation_score_threshold: float = 0.0,
) -> List[int]:
    """
    Select layers that are causally responsible for refusal.

    The preferred method uses ablation refusal scores from ``select_direction``
    — the causal test of how much refusal drops when a direction is removed at
    each layer.  Layers where ablating the direction pushes the refusal score
    below ``ablation_score_threshold`` are selected (lower score = stronger
    causal effect, negative = model no longer refuses).

    Falls back to top-k by raw magnitude when ablation scores are unavailable.

    Parameters
    ----------
    mean_diffs : (n_positions, n_layers, d_model)
        From ``generate_directions``.
    pos : int
        Token position selected by ``select_direction``.
    k : int or None
        Manual override: take exactly the top-k layers by ablation score
        (or magnitude if scores unavailable).  Useful for ablation sweeps.
    ablation_scores : dict or None
        List of dicts from ``select_direction/direction_evaluations.json``.
        Each entry has ``layer``, ``position``, ``refusal_score``.
    ablation_score_threshold : float
        Select layers whose ablation refusal score is below this value.
        Default 0.0 — only layers where ablation causes the model to stop
        refusing (score < 0) are selected.
    """
    n_layers = mean_diffs.shape[1]

    if ablation_scores is not None:
        # Use causal ablation scores at the selected token position.  Older
        # artifacts may not include position metadata; in that case keep the
        # previous cross-position behavior as a compatibility fallback.
        scored_entries = [
            entry for entry in ablation_scores
            if entry.get("position") == pos
        ]
        if not scored_entries:
            scored_entries = ablation_scores

        per_layer_best = {}
        for entry in scored_entries:
            ell = entry["layer"]
            score = entry["refusal_score"]
            if ell not in per_layer_best or score < per_layer_best[ell]:
                per_layer_best[ell] = score

        if k is not None:
            sorted_layers = sorted(per_layer_best.items(), key=lambda x: x[1])
            selected = sorted(ell for ell, _ in sorted_layers[:k])
        else:
            selected = sorted(
                ell for ell, score in per_layer_best.items()
                if score < ablation_score_threshold
            )
    else:
        # Fallback: top-k by raw magnitude
        magnitudes = mean_diffs[pos].norm(dim=-1)
        if k is not None:
            _, top_indices = magnitudes.topk(min(k, n_layers))
            selected = sorted(top_indices.tolist())
        else:
            # Top-5 by default when no ablation scores available
            _, top_indices = magnitudes.topk(min(5, n_layers))
            selected = sorted(top_indices.tolist())

    return selected


# ------------------------------------------------------------------
# Core defense
# ------------------------------------------------------------------

def apply_obfuscation(
    model,
    tokenize_fn,
    harmful_prompts: List[str],
    mean_diffs: torch.Tensor,
    selected_pos: int,
    selected_layer: int,
    direction: torch.Tensor,
    cfg: ObfuscationConfig = ObfuscationConfig(),
    ablation_scores: Optional[List[Dict]] = None,
    explicit_layers: Optional[List[int]] = None,
    harmless_prompts: Optional[List[str]] = None,
    harmless_ratio: float = 0.5,
    writer_only: bool = False,
) -> Dict:
    """
    Apply the representational-obfuscation defense to *model* **in-place**.

    Returns a diagnostics dict with keys:
        * ``pertinent_layers``  — list of patched layer indices
        * ``z_sum_norm``        — L2 norm of total residual-stream pollution
        * ``num_writers_patched`` / ``num_readers_patched``
    """
    device = next(model.parameters()).device
    components = ModelComponents(model)
    d = components.d_model
    num_layers = components.num_layers

    # ----------------------------------------------------------------
    # Step 0: Identify pertinent layers
    # ----------------------------------------------------------------
    if explicit_layers is not None:
        pertinent_layers: Set[int] = set(explicit_layers)
    else:
        pertinent_layers: Set[int] = set(
            select_pertinent_layers(
                mean_diffs, selected_pos,
                k=cfg.num_pertinent_layers,
                ablation_scores=ablation_scores,
            )
        )
    print(f"[obfuscation] Pertinent layers ({len(pertinent_layers)}): "
          f"{sorted(pertinent_layers)}")

    # ----------------------------------------------------------------
    # Collect calibration activations — mixed harmful + harmless
    # ----------------------------------------------------------------
    print("[obfuscation] Collecting calibration activations …")
    activations = collect_calibration_activations(
        model=model,
        components=components,
        harmful_prompts=harmful_prompts,
        harmless_prompts=harmless_prompts,
        harmless_ratio=harmless_ratio,
        tokenize_fn=tokenize_fn,
        num_prompts=cfg.num_calibration_prompts,
    )

    # ----------------------------------------------------------------
    # Step 1: Generate per-layer random aliases
    # ----------------------------------------------------------------
    generator = torch.Generator(device=device)
    generator.manual_seed(cfg.seed)

    # Normalised refusal direction(s)
    r_hat_global = (direction / direction.norm()).float().to(device)

    # Residual-stream direction dict: maps layer_idx -> unit vector.
    # This is either the global selected direction or the per-layer block-input
    # direction from generate_directions.
    if cfg.per_layer_direction:
        r_hat_residual_map: Dict[int, torch.Tensor] = {}
        for ell in sorted(pertinent_layers):
            layer_dir = mean_diffs[selected_pos, ell].float().to(device)
            norm = layer_dir.norm()
            if norm > 1e-8:
                r_hat_residual_map[ell] = layer_dir / norm
            else:
                # Fallback to global if layer direction is degenerate
                r_hat_residual_map[ell] = r_hat_global
        print(f"[obfuscation] Using per-layer refusal directions")
    else:
        r_hat_residual_map = {ell: r_hat_global for ell in sorted(pertinent_layers)}

    def _unit_or_fallback(vec: torch.Tensor, fallback: torch.Tensor) -> torch.Tensor:
        vec = vec.float().to(device)
        norm = vec.norm()
        if norm > 1e-8:
            return vec / norm
        return fallback

    if cfg.writer_output_directions:
        if harmless_prompts is None:
            print("[obfuscation] writer_output_directions requested but no "
                  "harmless prompts were provided; falling back to residual "
                  "directions.")
            r_hat_attn_map = dict(r_hat_residual_map)
            r_hat_mlp_map = dict(r_hat_residual_map)
        else:
            print("[obfuscation] Extracting writer-output refusal directions")
            writer_dirs = collect_writer_output_refusal_directions(
                model=model,
                components=components,
                harmful_prompts=harmful_prompts,
                harmless_prompts=harmless_prompts,
                tokenize_fn=tokenize_fn,
                num_prompts=cfg.num_calibration_prompts,
            )
            r_hat_attn_map = {}
            r_hat_mlp_map = {}
            for ell in sorted(pertinent_layers):
                fallback = r_hat_residual_map[ell]
                r_hat_attn_map[ell] = _unit_or_fallback(
                    writer_dirs["attn"][ell], fallback
                )
                r_hat_mlp_map[ell] = _unit_or_fallback(
                    writer_dirs["mlp"][ell], fallback
                )
            print("[obfuscation] Using per-writer output refusal directions")
    else:
        r_hat_attn_map = dict(r_hat_residual_map)
        r_hat_mlp_map = dict(r_hat_residual_map)

    mode = cfg.projection_mode
    assert mode in ("hadamard", "binary", "mask", "scalar_projection", "full"), \
        f"Unknown projection_mode: {mode}"

    # Per-layer noise containers
    attn_noise: Dict[int, object] = {}
    mlp_noise: Dict[int, object] = {}

    def _rademacher(n):
        """Generate a random ±1 vector of length n."""
        return (torch.randint(0, 2, (n,), device=device, generator=generator).float() * 2 - 1)

    def _binary_mask(n):
        """Generate a random {0, 1} vector of length n."""
        return torch.randint(0, 2, (n,), device=device, generator=generator).float()

    if mode == "hadamard":
        # Hadamard mode: r̂ℓ ⊙ ξ where ξ ~ N(0, ε²I).
        for ell in sorted(pertinent_layers):
            r_attn = r_hat_attn_map[ell]
            r_mlp = r_hat_mlp_map[ell]
            attn_noise[ell] = r_attn * torch.randn(d, device=device, generator=generator) * cfg.epsilon
            if cfg.separate_attn_mlp_aliases:
                mlp_noise[ell] = r_mlp * torch.randn(d, device=device, generator=generator) * cfg.epsilon
            else:
                mlp_noise[ell] = attn_noise[ell].clone()
    elif mode == "binary":
        # Binary mode: r̂ℓ ⊙ s where s_i ∈ {-1, +1} (Rademacher).
        for ell in sorted(pertinent_layers):
            r_attn = r_hat_attn_map[ell]
            r_mlp = r_hat_mlp_map[ell]
            attn_noise[ell] = r_attn * _rademacher(d)
            if cfg.separate_attn_mlp_aliases:
                mlp_noise[ell] = r_mlp * _rademacher(d)
            else:
                mlp_noise[ell] = attn_noise[ell].clone()
    elif mode == "mask":
        # Mask mode: r̂ℓ ⊙ m where m_i ∈ {0, 1}.
        for ell in sorted(pertinent_layers):
            r_attn = r_hat_attn_map[ell]
            r_mlp = r_hat_mlp_map[ell]
            attn_noise[ell] = r_attn * _binary_mask(d)
            if cfg.separate_attn_mlp_aliases:
                mlp_noise[ell] = r_mlp * _binary_mask(d)
            else:
                mlp_noise[ell] = attn_noise[ell].clone()
    elif mode == "scalar_projection":
        # Surgical mode: η · r̂ (single random scalar per writer).
        for ell in sorted(pertinent_layers):
            eta = torch.randn(1, device=device, generator=generator).item() * cfg.epsilon
            attn_noise[ell] = eta
            if cfg.separate_attn_mlp_aliases:
                mlp_noise[ell] = torch.randn(1, device=device, generator=generator).item() * cfg.epsilon
            else:
                mlp_noise[ell] = eta
    else:  # full
        # Full-alias mode: complete d-dimensional random vector used as the
        # replacement for the local refusal component.
        for ell in sorted(pertinent_layers):
            attn_noise[ell] = generate_random_alias(d, cfg.epsilon, device, generator)
            if cfg.separate_attn_mlp_aliases:
                mlp_noise[ell] = generate_random_alias(d, cfg.epsilon, device, generator)
            else:
                mlp_noise[ell] = attn_noise[ell].clone()

    # ----------------------------------------------------------------
    # Steps 2–5: Patch writers and readers with *empirical* pollution tracking.
    #
    # For each layer ell, we probe the actual (polluted) residual stream via a
    # forward pass through the currently-patched model.  This is what makes the
    # defense correct on architectures where the writer's output passes through
    # one or more LayerNorms inside the residual branch before being added to
    # the stream (Gemma 2/3).  Under those wrappings, the net pollution is a
    # nonlinear function of the alias vector — analytical accumulation would
    # drift from reality.
    #
    # Within a layer, attention readers read the residual before the attention
    # writer fires, and MLP readers read after it fires, so we probe twice per
    # layer.  Reader patches at the calibration point do not themselves change
    # the residual stream (by construction they match clean outputs), so
    # probing *before* the current layer's patches and *after* attention writer
    # injection is sufficient.
    # ----------------------------------------------------------------

    # Build the probe prompt list (subset of the calibration mix).  Keep this
    # balanced when harmless prompts are present: the reader/LM-head patches
    # compare polluted probe activations against clean probe activations, so the
    # two sides must be averaged over the same prompt distribution.
    probe_budget = max(1, min(cfg.num_probe_prompts, cfg.num_calibration_prompts))
    if harmless_prompts is not None and harmless_ratio > 0:
        n_harmless = int(probe_budget * harmless_ratio)
        if probe_budget > 1 and n_harmless == 0:
            n_harmless = 1
        n_harmful = probe_budget - n_harmless
        probe_pool = (
            list(harmful_prompts[:n_harmful]) +
            list(harmless_prompts[:n_harmless])
        )
    else:
        probe_pool = list(harmful_prompts[:probe_budget])
    probe_prompts = probe_pool[:probe_budget]
    if not probe_prompts:
        raise ValueError("No probe prompts available for obfuscation")

    print("[obfuscation] Collecting clean probe activations …")
    probe_clean_activations = collect_calibration_activations(
        model=model,
        components=components,
        harmful_prompts=probe_prompts,
        harmless_prompts=None,
        tokenize_fn=tokenize_fn,
        num_prompts=len(probe_prompts),
        explicit_prompts=probe_prompts,
    )

    num_writers_patched = 0
    num_readers_patched = 0
    pollution_injected = False
    pollution_threshold = 1e-6

    def _patch_readers_at(ell: int, sublayer: str) -> int:
        """Empirically probe and patch the readers at (ell, sublayer).

        sublayer is "attn" or "mlp".  Returns the number of readers patched.
        """
        if writer_only or not pollution_injected:
            return 0
        key = f"layer_{ell}_{sublayer}_ln_input"
        probed = probe_residual_stream(
            model=model,
            components=components,
            keys=[key],
            prompts=probe_prompts,
            tokenize_fn=tokenize_fn,
        )
        x_clean = probe_clean_activations[key].float()
        x_polluted = probed[key].float()

        if (x_polluted - x_clean).norm().item() <= pollution_threshold:
            return 0

        ln_module = (components.get_attn_layernorm(ell)
                     if sublayer == "attn"
                     else components.get_mlp_layernorm(ell))
        readers = (components.get_attn_reader_projs(ell)
                   if sublayer == "attn"
                   else components.get_mlp_reader_projs(ell))

        with torch.no_grad():
            ln_clean = ln_module(
                x_clean.unsqueeze(0).unsqueeze(0)
            ).squeeze().float()
            ln_polluted = ln_module(
                x_polluted.unsqueeze(0).unsqueeze(0)
            ).squeeze().float()

        patched = 0
        for _, proj_module in readers:
            W = proj_module.weight.data
            W_new = rank_one_update(W, ln_polluted, W.float() @ ln_clean)
            proj_module.weight.data = W_new
            patched += 1
        return patched

    for ell in range(num_layers):

        # ==============================================================
        # ATTENTION SUBLAYER
        # ==============================================================

        # --- Step 4a: Patch attention readers (Q, K, V) ---
        num_readers_patched += _patch_readers_at(ell, "attn")

        # --- Step 2a: Patch W_O (attention writer) if pertinent ---
        # At this point, attention readers at ell are already patched, so the
        # actual pre-W_O activation at inference on calibration prompts equals
        # the clean calibration value.  We can therefore anchor the rank-one
        # writer patch at the clean x_attn without drift.
        if ell in pertinent_layers and cfg.patch_writers in ("both", "attn_only"):
            o_proj = components.get_attn_output_proj(ell)
            x_attn = activations[f"layer_{ell}_attn_o_input"].float()
            r_l = r_hat_attn_map[ell]

            current_output = o_proj.weight.data.float() @ x_attn

            if mode in ("hadamard", "binary", "mask"):
                proj_scalar = (current_output @ r_l).item()
                target = current_output - proj_scalar * r_l + attn_noise[ell]
            elif mode == "scalar_projection":
                proj_scalar = (current_output @ r_l).item()
                target = current_output - proj_scalar * r_l + attn_noise[ell] * r_l
            else:
                proj_scalar = (current_output @ r_l).item()
                target = current_output - proj_scalar * r_l + attn_noise[ell].float()

            o_proj.weight.data = rank_one_update(o_proj.weight.data, x_attn, target)
            num_writers_patched += 1
            pollution_injected = True

        # ==============================================================
        # MLP SUBLAYER
        # ==============================================================

        # --- Step 4b: Patch MLP readers (gate, up) ---
        num_readers_patched += _patch_readers_at(ell, "mlp")

        # --- Step 2b: Patch W_down (MLP writer) if pertinent ---
        if ell in pertinent_layers and cfg.patch_writers in ("both", "mlp_only"):
            down_proj = components.get_mlp_output_proj(ell)
            x_mlp = activations[f"layer_{ell}_mlp_down_input"].float()
            r_l = r_hat_mlp_map[ell]

            current_output = down_proj.weight.data.float() @ x_mlp

            if mode in ("hadamard", "binary", "mask"):
                proj_scalar = (current_output @ r_l).item()
                target = current_output - proj_scalar * r_l + mlp_noise[ell]
            elif mode == "scalar_projection":
                proj_scalar = (current_output @ r_l).item()
                target = current_output - proj_scalar * r_l + mlp_noise[ell] * r_l
            else:
                proj_scalar = (current_output @ r_l).item()
                target = current_output - proj_scalar * r_l + mlp_noise[ell].float()

            down_proj.weight.data = rank_one_update(
                down_proj.weight.data, x_mlp, target
            )
            num_writers_patched += 1
            pollution_injected = True

    # ----------------------------------------------------------------
    # Step 6: Patch the unembedding matrix (LM head)
    #
    # Probe the final residual stream empirically — same rationale as
    # per-layer probing: the net pollution reaching the final LN is the
    # architecturally-transformed sum of writer contributions, not the naive
    # analytical sum.
    # ----------------------------------------------------------------
    z_sum_norm = 0.0
    if pollution_injected:
        probed_final = probe_residual_stream(
            model=model,
            components=components,
            keys=["final_ln_input"],
            prompts=probe_prompts,
            tokenize_fn=tokenize_fn,
        )
        x_clean_final = probe_clean_activations["final_ln_input"].float()
        x_polluted_final = probed_final["final_ln_input"].float()
        z_sum_norm = (x_polluted_final - x_clean_final).norm().item()

        if z_sum_norm > pollution_threshold:
            final_ln = components.final_norm
            with torch.no_grad():
                ln_clean = final_ln(
                    x_clean_final.unsqueeze(0).unsqueeze(0)
                ).squeeze().float()
                ln_polluted = final_ln(
                    x_polluted_final.unsqueeze(0).unsqueeze(0)
                ).squeeze().float()

            W_unembed = components.lm_head.weight.data
            target_logits = W_unembed.float() @ ln_clean
            W_unembed_new = rank_one_update(W_unembed, ln_polluted, target_logits)
            components.lm_head.weight.data = W_unembed_new

    # ----------------------------------------------------------------
    # Diagnostics
    # ----------------------------------------------------------------
    print(f"[obfuscation] Defense applied ({mode} mode).")
    print(f"  Writers patched : {num_writers_patched}")
    print(f"  Readers patched : {num_readers_patched}")
    print(f"  z_sum norm      : {z_sum_norm:.4f}")

    return {
        "pertinent_layers": sorted(pertinent_layers),
        "z_sum_norm": z_sum_norm,
        "num_writers_patched": num_writers_patched,
        "num_readers_patched": num_readers_patched,
    }


# ------------------------------------------------------------------
# Convenience: load artifacts produced by the existing pipeline and
# apply the defense in one call.
# ------------------------------------------------------------------

def apply_obfuscation_from_artifacts(
    model,
    tokenize_fn,
    harmful_prompts: List[str],
    artifact_dir: str,
    cfg: ObfuscationConfig = ObfuscationConfig(),
    harmless_prompts: Optional[List[str]] = None,
    ablation_scores: Optional[List[Dict]] = None,
) -> Dict:
    """
    Load ``direction.pt``, ``mean_diffs.pt``, and ``direction_metadata.json``
    from *artifact_dir* (produced by the upstream ``generate_directions`` /
    ``select_direction`` pipeline), then apply the defense.
    """
    direction = torch.load(
        os.path.join(artifact_dir, "direction.pt"), map_location="cpu"
    )
    mean_diffs = torch.load(
        os.path.join(artifact_dir, "generate_directions", "mean_diffs.pt"),
        map_location="cpu",
    )
    with open(os.path.join(artifact_dir, "direction_metadata.json")) as f:
        meta = json.load(f)
    if ablation_scores is None:
        ablation_scores_path = os.path.join(
            artifact_dir, "select_direction", "direction_evaluations.json"
        )
        if os.path.exists(ablation_scores_path):
            with open(ablation_scores_path) as f:
                ablation_scores = json.load(f)

    return apply_obfuscation(
        model=model,
        tokenize_fn=tokenize_fn,
        harmful_prompts=harmful_prompts,
        harmless_prompts=harmless_prompts,
        mean_diffs=mean_diffs,
        selected_pos=meta["pos"],
        selected_layer=meta["layer"],
        direction=direction,
        cfg=cfg,
        ablation_scores=ablation_scores,
    )
