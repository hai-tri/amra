"""
Defense integrity evaluation — measures how much the obfuscation defense
changes the model's internal representations and output distribution.

Two diagnostics:

1. **Residual-stream variance tracking**
   At every sublayer boundary (before attn LN, before MLP LN, before final LN),
   collect the activation vector for each prompt.  Compare before vs. after
   defense:
     - Per-layer relative variance shift  Δσ²/σ²
     - Per-layer mean cosine similarity   cos(h_before, h_after)
     - Per-layer L2-norm ratio            ||h_after|| / ||h_before||

2. **Output-distribution KL divergence**
   Collect last-token logits before and after, compute
   KL(P_original || P_defended) per prompt, then average.
   Should be ≈ 0 if the rank-one patches fully compensate.
"""

import os
import torch
import torch.nn as nn
import torch.nn.functional as F
import matplotlib.pyplot as plt
import matplotlib.ticker as ticker
from typing import Dict, List, Optional
from tqdm import tqdm

from obfuscation_utils import ModelComponents


# ======================================================================
# Residual-stream activation collection
# ======================================================================

def collect_residual_activations(
    model: nn.Module,
    components: ModelComponents,
    prompts: List[str],
    tokenize_fn,
    num_prompts: int = 32,
    fwd_pre_hooks: Optional[List] = None,
    fwd_hooks: Optional[List] = None,
) -> Dict[str, torch.Tensor]:
    """
    Run forward passes and collect residual-stream activations at every
    sublayer boundary (last-token position).

    Optional ``fwd_pre_hooks`` / ``fwd_hooks`` apply the active defense's
    inference-time hooks during collection — required when the defense is
    hook-based (surgical, CAST, AlphaSteer) rather than weight-modifying,
    otherwise this collects activations from the *undefended* model.

    Returns a dict mapping probe-point names to tensors of shape
    ``(num_prompts, d_model)``.
    """
    from pipeline.utils.hook_utils import add_hooks as _add_hooks

    fwd_pre_hooks = fwd_pre_hooks or []
    fwd_hooks = fwd_hooks or []

    device = next(model.parameters()).device
    num_layers = components.num_layers

    # Accumulate per-prompt activations
    buffers: Dict[str, List[torch.Tensor]] = {}

    def _make_hook(key: str):
        def hook_fn(module, inp, output):
            x = inp[0] if isinstance(inp, tuple) else inp
            vec = x[0, -1, :].detach().float().cpu()
            buffers.setdefault(key, []).append(vec)
        return hook_fn

    prompts_to_use = prompts[:num_prompts]
    with _add_hooks(fwd_pre_hooks, fwd_hooks):
        probe_hooks = []
        try:
            for ell in range(num_layers):
                probe_hooks.append(
                    components.get_attn_layernorm(ell).register_forward_hook(
                        _make_hook(f"layer_{ell}_attn_ln")
                    )
                )
                probe_hooks.append(
                    components.get_mlp_layernorm(ell).register_forward_hook(
                        _make_hook(f"layer_{ell}_mlp_ln")
                    )
                )
            probe_hooks.append(
                components.final_norm.register_forward_hook(
                    _make_hook("final_ln")
                )
            )

            with torch.no_grad():
                for prompt in tqdm(prompts_to_use, desc="Collecting residual activations"):
                    inputs = tokenize_fn(instructions=[prompt])
                    model(
                        input_ids=inputs.input_ids.to(device),
                        attention_mask=inputs.attention_mask.to(device),
                    )
        finally:
            for h in probe_hooks:
                h.remove()

    # Stack lists into (N, d) tensors
    result = {k: torch.stack(v) for k, v in buffers.items()}
    return result


# ======================================================================
# Output logit collection
# ======================================================================

def collect_output_logits(
    model: nn.Module,
    prompts: List[str],
    tokenize_fn,
    num_prompts: int = 32,
    fwd_pre_hooks: Optional[List] = None,
    fwd_hooks: Optional[List] = None,
) -> torch.Tensor:
    """
    Collect last-token logits for each prompt.

    Optional ``fwd_pre_hooks`` / ``fwd_hooks`` apply the active defense's
    inference-time hooks — required when the defense is hook-based.

    Returns tensor of shape ``(num_prompts, vocab_size)`` in float32 on CPU.
    """
    from pipeline.utils.hook_utils import add_hooks as _add_hooks

    fwd_pre_hooks = fwd_pre_hooks or []
    fwd_hooks = fwd_hooks or []

    device = next(model.parameters()).device
    all_logits = []

    prompts_to_use = prompts[:num_prompts]
    with torch.no_grad(), _add_hooks(fwd_pre_hooks, fwd_hooks):
        for prompt in tqdm(prompts_to_use, desc="Collecting output logits"):
            inputs = tokenize_fn(instructions=[prompt])
            outputs = model(
                input_ids=inputs.input_ids.to(device),
                attention_mask=inputs.attention_mask.to(device),
            )
            # Last-token logits: (1, vocab) → (vocab,)
            logits = outputs.logits[0, -1, :].float().cpu()
            all_logits.append(logits)

    return torch.stack(all_logits)  # (N, vocab)


# ======================================================================
# Comparison: residual stream
# ======================================================================

def compare_residual_stats(
    acts_before: Dict[str, torch.Tensor],
    acts_after: Dict[str, torch.Tensor],
) -> Dict[str, Dict]:
    """
    Compare residual-stream activations collected before and after the defense.

    For each probe point returns:
        * ``rel_variance_shift``  — (Var_after − Var_before) / Var_before,
          where variance is computed across the d_model dimension and averaged
          over prompts.  This is the quantity LayerNorm normalises by, so
          shifts here translate directly to downstream distortion.
        * ``mean_cosine_sim``     — average cos(h_before, h_after) across
          prompts.  1.0 means the defense changed nothing.
        * ``norm_ratio``          — mean(||h_after||) / mean(||h_before||).
        * ``max_l2_diff``         — max over prompts of ||h_after − h_before||.
        * ``mean_l2_diff``        — mean over prompts of ||h_after − h_before||.
    """
    results = {}

    for key in sorted(acts_before.keys()):
        hb = acts_before[key]  # (N, d)
        ha = acts_after[key]   # (N, d)

        # Per-prompt, per-dim variance (what LayerNorm sees)
        var_before = hb.var(dim=-1).mean().item()   # scalar
        var_after = ha.var(dim=-1).mean().item()
        rel_shift = (var_after - var_before) / (var_before + 1e-12)

        # Per-prompt cosine similarity
        cos_sims = F.cosine_similarity(hb, ha, dim=-1)  # (N,)

        # Norms
        norms_before = hb.norm(dim=-1)  # (N,)
        norms_after = ha.norm(dim=-1)

        # L2 diffs
        l2_diffs = (ha - hb).norm(dim=-1)  # (N,)

        results[key] = {
            "var_before": var_before,
            "var_after": var_after,
            "rel_variance_shift": rel_shift,
            "mean_cosine_sim": cos_sims.mean().item(),
            "min_cosine_sim": cos_sims.min().item(),
            "norm_ratio": (norms_after.mean() / (norms_before.mean() + 1e-12)).item(),
            "max_l2_diff": l2_diffs.max().item(),
            "mean_l2_diff": l2_diffs.mean().item(),
            "mean_norm_before": norms_before.mean().item(),
            "mean_norm_after": norms_after.mean().item(),
        }

    return results


# ======================================================================
# Comparison: output distribution
# ======================================================================

def compute_output_kl_divergence(
    logits_before: torch.Tensor,
    logits_after: torch.Tensor,
) -> Dict:
    """
    Compute KL(P_before || P_after) per prompt and return summary statistics.

    Both inputs are ``(N, vocab_size)`` in float32.
    """
    log_p = F.log_softmax(logits_before.double(), dim=-1)
    log_q = F.log_softmax(logits_after.double(), dim=-1)
    p = log_p.exp()

    # KL(P || Q) = Σ p * (log_p - log_q)
    kl_per_prompt = (p * (log_p - log_q)).sum(dim=-1)  # (N,)

    # Also compute reverse KL for completeness
    q = log_q.exp()
    kl_reverse = (q * (log_q - log_p)).sum(dim=-1)

    # Jensen-Shannon divergence (symmetric, bounded)
    m = 0.5 * (p + q)
    log_m = m.log()
    jsd = 0.5 * (p * (log_p - log_m)).sum(dim=-1) + \
          0.5 * (q * (log_q - log_m)).sum(dim=-1)

    return {
        "kl_forward_mean": kl_per_prompt.mean().item(),
        "kl_forward_max": kl_per_prompt.max().item(),
        "kl_forward_std": kl_per_prompt.std().item(),
        "kl_forward_per_prompt": kl_per_prompt.tolist(),
        "kl_reverse_mean": kl_reverse.mean().item(),
        "kl_reverse_max": kl_reverse.max().item(),
        "jsd_mean": jsd.mean().item(),
        "jsd_max": jsd.max().item(),
    }


# ======================================================================
# Top-level orchestrator
# ======================================================================

def collect_pre_defense_measurements(
    model: nn.Module,
    components: ModelComponents,
    harmful_prompts: List[str],
    harmless_prompts: List[str],
    tokenize_fn,
    num_prompts: int = 32,
) -> Dict:
    """
    Collect all measurements that require the *original* (undefended) model.
    Call this BEFORE ``apply_obfuscation``.
    """
    print("[integrity] Collecting pre-defense residual-stream stats …")
    acts_harmful = collect_residual_activations(
        model, components, harmful_prompts, tokenize_fn, num_prompts,
    )
    acts_harmless = collect_residual_activations(
        model, components, harmless_prompts, tokenize_fn, num_prompts,
    )

    print("[integrity] Collecting pre-defense output logits …")
    logits_harmful = collect_output_logits(
        model, harmful_prompts, tokenize_fn, num_prompts,
    )
    logits_harmless = collect_output_logits(
        model, harmless_prompts, tokenize_fn, num_prompts,
    )

    return {
        "residual_harmful": acts_harmful,
        "residual_harmless": acts_harmless,
        "logits_harmful": logits_harmful,
        "logits_harmless": logits_harmless,
    }


def evaluate_defense_integrity(
    model: nn.Module,
    components: ModelComponents,
    harmful_prompts: List[str],
    harmless_prompts: List[str],
    tokenize_fn,
    pre_measurements: Dict,
    num_prompts: int = 32,
    pertinent_layers: Optional[List[int]] = None,
    artifact_dir: Optional[str] = None,
    fwd_pre_hooks: Optional[List] = None,
    fwd_hooks: Optional[List] = None,
) -> Dict:
    """
    Collect post-defense measurements, compare with pre-defense, and return
    a full integrity report.

    Call this AFTER the defense is applied.  For hook-based defenses
    (surgical / CAST / AlphaSteer), pass the defense's
    ``fwd_pre_hooks`` / ``fwd_hooks`` so the "post-defense" measurements
    actually reflect the defense; otherwise they would measure the
    undefended model.  For weight-modifying defenses (APRS, circuit
    breakers) the hooks lists are empty and these parameters are no-ops.
    """
    # --- Post-defense residual-stream stats ---
    print("[integrity] Collecting post-defense residual-stream stats …")
    post_acts_harmful = collect_residual_activations(
        model, components, harmful_prompts, tokenize_fn, num_prompts,
        fwd_pre_hooks=fwd_pre_hooks, fwd_hooks=fwd_hooks,
    )
    post_acts_harmless = collect_residual_activations(
        model, components, harmless_prompts, tokenize_fn, num_prompts,
        fwd_pre_hooks=fwd_pre_hooks, fwd_hooks=fwd_hooks,
    )

    # --- Post-defense output logits ---
    print("[integrity] Collecting post-defense output logits …")
    post_logits_harmful = collect_output_logits(
        model, harmful_prompts, tokenize_fn, num_prompts,
        fwd_pre_hooks=fwd_pre_hooks, fwd_hooks=fwd_hooks,
    )
    post_logits_harmless = collect_output_logits(
        model, harmless_prompts, tokenize_fn, num_prompts,
        fwd_pre_hooks=fwd_pre_hooks, fwd_hooks=fwd_hooks,
    )

    # --- Residual stream comparison ---
    print("[integrity] Computing residual-stream diffs …")
    residual_harmful = compare_residual_stats(
        pre_measurements["residual_harmful"], post_acts_harmful,
    )
    residual_harmless = compare_residual_stats(
        pre_measurements["residual_harmless"], post_acts_harmless,
    )

    # --- Output KL divergence ---
    print("[integrity] Computing output KL divergence …")
    kl_harmful = compute_output_kl_divergence(
        pre_measurements["logits_harmful"], post_logits_harmful,
    )
    kl_harmless = compute_output_kl_divergence(
        pre_measurements["logits_harmless"], post_logits_harmless,
    )

    # --- Summary ---
    # Aggregate per-layer stats into summary vectors for easy plotting
    num_layers = components.num_layers
    summary = _build_summary(residual_harmful, residual_harmless, num_layers)

    print("\n[integrity] === Residual-Stream Variance Report ===")
    print(f"  Harmful prompts:")
    print(f"    Mean |Δσ²/σ²| across layers (attn LN): "
          f"{summary['harmful_attn_mean_abs_var_shift']:.6f}")
    print(f"    Mean |Δσ²/σ²| across layers (MLP LN) : "
          f"{summary['harmful_mlp_mean_abs_var_shift']:.6f}")
    print(f"    Mean cosine sim (attn LN)             : "
          f"{summary['harmful_attn_mean_cos_sim']:.6f}")
    print(f"    Mean cosine sim (MLP LN)              : "
          f"{summary['harmful_mlp_mean_cos_sim']:.6f}")
    print(f"    Final LN cosine sim                   : "
          f"{residual_harmful.get('final_ln', {}).get('mean_cosine_sim', float('nan')):.6f}")
    print(f"  Harmless prompts:")
    print(f"    Mean |Δσ²/σ²| across layers (attn LN): "
          f"{summary['harmless_attn_mean_abs_var_shift']:.6f}")
    print(f"    Mean |Δσ²/σ²| across layers (MLP LN) : "
          f"{summary['harmless_mlp_mean_abs_var_shift']:.6f}")
    print(f"    Mean cosine sim (attn LN)             : "
          f"{summary['harmless_attn_mean_cos_sim']:.6f}")
    print(f"    Mean cosine sim (MLP LN)              : "
          f"{summary['harmless_mlp_mean_cos_sim']:.6f}")
    print(f"    Final LN cosine sim                   : "
          f"{residual_harmless.get('final_ln', {}).get('mean_cosine_sim', float('nan')):.6f}")

    print(f"\n[integrity] === Output KL Divergence ===")
    print(f"  Harmful prompts  — KL(orig||def): "
          f"mean={kl_harmful['kl_forward_mean']:.6f}, "
          f"max={kl_harmful['kl_forward_max']:.6f}, "
          f"JSD={kl_harmful['jsd_mean']:.6f}")
    print(f"  Harmless prompts — KL(orig||def): "
          f"mean={kl_harmless['kl_forward_mean']:.6f}, "
          f"max={kl_harmless['kl_forward_max']:.6f}, "
          f"JSD={kl_harmless['jsd_mean']:.6f}")

    # --- Plot ---
    fig = plot_variance_shifts(
        summary, num_layers,
        pertinent_layers=pertinent_layers,
        artifact_dir=artifact_dir,
    )
    plt.close(fig)

    return {
        "residual_harmful": residual_harmful,
        "residual_harmless": residual_harmless,
        "kl_harmful": kl_harmful,
        "kl_harmless": kl_harmless,
        "summary": summary,
    }


def _build_summary(
    residual_harmful: Dict,
    residual_harmless: Dict,
    num_layers: int,
) -> Dict:
    """Aggregate per-layer residual stats into summary scalars."""
    summary = {}

    for label, res in [("harmful", residual_harmful), ("harmless", residual_harmless)]:
        for sublayer, prefix in [("attn_ln", "attn"), ("mlp_ln", "mlp")]:
            var_before_list = []
            var_after_list = []
            var_shifts_signed = []
            var_shifts_abs = []
            cos_sims = []
            l2_diffs = []
            for ell in range(num_layers):
                key = f"layer_{ell}_{sublayer}"
                if key in res:
                    var_before_list.append(res[key]["var_before"])
                    var_after_list.append(res[key]["var_after"])
                    var_shifts_signed.append(res[key]["rel_variance_shift"])
                    var_shifts_abs.append(abs(res[key]["rel_variance_shift"]))
                    cos_sims.append(res[key]["mean_cosine_sim"])
                    l2_diffs.append(res[key]["mean_l2_diff"])

            if var_shifts_abs:
                summary[f"{label}_{prefix}_mean_abs_var_shift"] = sum(var_shifts_abs) / len(var_shifts_abs)
                summary[f"{label}_{prefix}_max_abs_var_shift"] = max(var_shifts_abs)
                summary[f"{label}_{prefix}_mean_cos_sim"] = sum(cos_sims) / len(cos_sims)
                summary[f"{label}_{prefix}_min_cos_sim"] = min(cos_sims)
                summary[f"{label}_{prefix}_mean_l2_diff"] = sum(l2_diffs) / len(l2_diffs)
                summary[f"{label}_{prefix}_max_l2_diff"] = max(l2_diffs)

                # Per-layer arrays for plotting
                summary[f"{label}_{prefix}_var_before"] = var_before_list
                summary[f"{label}_{prefix}_var_after"] = var_after_list
                summary[f"{label}_{prefix}_var_shifts"] = var_shifts_abs
                summary[f"{label}_{prefix}_var_shifts_signed"] = var_shifts_signed
                summary[f"{label}_{prefix}_cos_sims"] = cos_sims
                summary[f"{label}_{prefix}_l2_diffs"] = l2_diffs

    return summary


# ======================================================================
# Plotting
# ======================================================================

def plot_variance_shifts(
    summary: Dict,
    num_layers: int,
    pertinent_layers: Optional[List[int]] = None,
    artifact_dir: Optional[str] = None,
) -> plt.Figure:
    """
    Plot per-layer residual-stream variance — base model vs defended model.

    Two subplots stacked vertically (harmful / harmless prompts).
    Each subplot shows the variance at the attention LayerNorm boundary
    (cleanest view of the residual stream entering each layer):
      * Solid line  — base model σ²
      * Dashed line — defended model σ²

    Pertinent layers are shaded red so the reader can see where the defense
    acts and how much the variance changes there vs downstream.
    """
    layers = list(range(num_layers))

    fig, axes = plt.subplots(2, 1, figsize=(11, 7), sharex=True)

    for ax, label, title in zip(
        axes,
        ["harmful", "harmless"],
        ["Harmful prompts", "Harmless prompts"],
    ):
        before_key = f"{label}_attn_var_before"
        after_key = f"{label}_attn_var_after"

        if before_key in summary and after_key in summary:
            ax.plot(
                layers, summary[before_key],
                color="#1f77b4", linewidth=1.8, linestyle="-",
                label="Base model",
            )
            ax.plot(
                layers, summary[after_key],
                color="#d62728", linewidth=1.8, linestyle="--",
                label="Defended model",
            )

        # Highlight pertinent layers
        if pertinent_layers:
            for pl in pertinent_layers:
                ax.axvspan(pl - 0.4, pl + 0.4, alpha=0.10, color="red")
            ax.axvspan(-999, -998, alpha=0.10, color="red", label="Pertinent layer")

        ax.set_ylabel(r"Residual-stream variance $\sigma^2$")
        ax.set_title(title)
        ax.legend(loc="upper left", fontsize=9)
        ax.yaxis.set_major_formatter(ticker.FormatStrFormatter("%.2f"))

    axes[-1].set_xlabel("Layer")
    axes[-1].set_xlim(-0.5, num_layers - 0.5)
    fig.suptitle(
        "Residual-Stream Variance: Base Model vs Defended Model",
        fontsize=13, fontweight="bold",
    )
    fig.tight_layout(rect=[0, 0, 1, 0.95])

    if artifact_dir is not None:
        path = os.path.join(artifact_dir, "variance_shift_by_layer.png")
        fig.savefig(path, dpi=150)
        print(f"[integrity] Saved variance plot → {path}")

    return fig
