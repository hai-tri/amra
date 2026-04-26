"""
LEACE (LEAst-squares Concept Erasure) attack — Marks et al. 2023.

Applies the theoretically optimal linear erasure of the refusal concept from
the residual stream.  Unlike simple difference-in-means, LEACE whitens the
activation covariance before finding the concept subspace, making it strictly
stronger than PCA-based or per-layer mean-diff attacks.

Reference: "The Geometry of Truth: Emergent Linear Structure in Large Language
Model Representations of True/False Datasets" — Marks & Tegmark, 2023
Paper: https://arxiv.org/abs/2306.03819
Code:  https://github.com/EleutherAI/concept-erasure

This module reimplements the core LEACE algorithm without depending on the
``concept-erasure`` library so that it integrates cleanly with the existing
hook-based evaluation infrastructure.
"""

import torch
import sys
import os
from typing import Dict, List, Optional, Tuple

_REFUSAL_DIR = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "refusal_direction")
if _REFUSAL_DIR not in sys.path:
    sys.path.insert(0, _REFUSAL_DIR)

from pipeline.submodules.generate_directions import get_mean_activations
from pipeline.submodules.select_direction import get_refusal_scores
from pipeline.utils.hook_utils import add_hooks


# ======================================================================
# LEACE fitter
# ======================================================================

class LeaceFitter:
    """
    Streaming LEACE fitter that accumulates sufficient statistics (mean,
    covariance, cross-covariance) and produces an erasure projection.

    Parameters
    ----------
    d_model : int
        Hidden dimension.
    num_classes : int
        Number of concept classes (2 for harmful/harmless).
    svd_tol : float
        Singular values below this threshold are discarded.
    """

    def __init__(self, d_model: int, num_classes: int = 2, svd_tol: float = 0.01):
        self.d_model = d_model
        self.num_classes = num_classes
        self.svd_tol = svd_tol

        # Sufficient statistics (float64 for numerical stability)
        self.n = 0
        self.sum_x = torch.zeros(d_model, dtype=torch.float64)
        self.sum_x2 = torch.zeros(d_model, d_model, dtype=torch.float64)
        self.sum_xz = torch.zeros(d_model, num_classes, dtype=torch.float64)
        self.sum_z = torch.zeros(num_classes, dtype=torch.float64)

    def update(self, x: torch.Tensor, z: torch.Tensor):
        """
        Add a batch of observations.

        Parameters
        ----------
        x : Tensor of shape (batch, d_model)
            Activation vectors.
        z : Tensor of shape (batch, num_classes)
            One-hot label vectors.
        """
        x = x.double().cpu()
        z = z.double().cpu()
        n = x.shape[0]

        self.n += n
        self.sum_x += x.sum(dim=0)
        self.sum_x2 += x.T @ x
        self.sum_xz += x.T @ z
        self.sum_z += z.sum(dim=0)

    def fit(self) -> Dict[str, torch.Tensor]:
        """
        Compute the LEACE erasure projection from accumulated statistics.

        Returns
        -------
        dict with keys:
            * ``proj_left``  — (d_model, k') left factor of the erasure
            * ``proj_right`` — (k', d_model) right factor
            * ``bias``       — (d_model,) mean of X
            * ``directions`` — (k', d_model) the concept-correlated directions
              found by LEACE (for diagnostics / cos-sim reporting)

        The erasure is applied as::

            x_erased = x - proj_left @ proj_right @ (x - bias)
        """
        n = self.n
        mean_x = self.sum_x / n
        mean_z = self.sum_z / n

        # Covariance: Σ_xx = E[xxᵀ] - E[x]E[x]ᵀ
        sigma_xx = self.sum_x2 / n - mean_x.unsqueeze(1) * mean_x.unsqueeze(0)
        # Cross-covariance: Σ_xz = E[xzᵀ] - E[x]E[z]ᵀ
        sigma_xz = self.sum_xz / n - mean_x.unsqueeze(1) * mean_z.unsqueeze(0)

        # Regularise Σ_xx for numerical stability
        sigma_xx += 1e-6 * torch.eye(self.d_model, dtype=torch.float64)

        # Eigendecompose Σ_xx for whitening
        eigvals, eigvecs = torch.linalg.eigh(sigma_xx)
        eigvals = eigvals.clamp(min=1e-8)

        # W = Σ_xx^{-1/2},  W_inv = Σ_xx^{1/2}
        inv_sqrt = eigvals.pow(-0.5)
        sqrt_vals = eigvals.pow(0.5)

        W = eigvecs @ torch.diag(inv_sqrt) @ eigvecs.T
        W_inv = eigvecs @ torch.diag(sqrt_vals) @ eigvecs.T

        # SVD on whitened cross-covariance
        whitened_xz = W @ sigma_xz
        U, S, Vh = torch.linalg.svd(whitened_xz, full_matrices=False)

        # Threshold small singular values
        mask = S > self.svd_tol
        U = U[:, mask]
        S = S[mask]
        k = U.shape[1]

        if k == 0:
            # No concept signal found — return identity (no erasure)
            return {
                "proj_left": torch.zeros(self.d_model, 1, dtype=torch.float64),
                "proj_right": torch.zeros(1, self.d_model, dtype=torch.float64),
                "bias": mean_x,
                "directions": torch.zeros(1, self.d_model, dtype=torch.float64),
            }

        # Construct sparse projection factors
        proj_left = W_inv @ U           # (d, k)
        proj_right = U.T @ W            # (k, d)

        # The directions LEACE found (in original space, for diagnostics)
        directions = proj_right / proj_right.norm(dim=1, keepdim=True)

        return {
            "proj_left": proj_left,
            "proj_right": proj_right,
            "bias": mean_x,
            "directions": directions,
        }


# ======================================================================
# Collect per-layer activations for LEACE fitting
# ======================================================================

def _collect_last_token_activations(
    model,
    tokenize_fn,
    block_modules,
    prompts: List[str],
    batch_size: int = 128,
    base_fwd_pre_hooks: Optional[List] = None,
    base_fwd_hooks: Optional[List] = None,
) -> torch.Tensor:
    """
    Collect last-token activations at every layer.

    Returns
    -------
    Tensor of shape (n_prompts, n_layers, d_model)
    """
    base_fwd_pre_hooks = base_fwd_pre_hooks or []
    base_fwd_hooks = base_fwd_hooks or []

    num_layers = model.config.num_hidden_layers
    d_model = model.config.hidden_size
    device = next(model.parameters()).device

    all_activations = []

    # Accumulate per-prompt activations using hooks
    for i in range(0, len(prompts), batch_size):
        batch = prompts[i : i + batch_size]
        batch_acts = torch.zeros(len(batch), num_layers, d_model, dtype=torch.float32)

        with add_hooks(base_fwd_pre_hooks, base_fwd_hooks):
            hooks = []
            try:
                for ell in range(num_layers):
                    def make_hook(layer_idx):
                        def hook_fn(module, inp):
                            x = inp[0] if isinstance(inp, tuple) else inp
                            # Last token position
                            batch_acts[:x.shape[0], layer_idx] = x[:, -1, :].detach().float().cpu()
                        return hook_fn
                    hooks.append(
                        block_modules[ell].register_forward_pre_hook(make_hook(ell))
                    )

                with torch.no_grad():
                    inputs = tokenize_fn(instructions=batch)
                    model(
                        input_ids=inputs.input_ids.to(device),
                        attention_mask=inputs.attention_mask.to(device),
                    )
            finally:
                for h in hooks:
                    h.remove()

        all_activations.append(batch_acts)

    return torch.cat(all_activations, dim=0)  # (n, n_layers, d)


# ======================================================================
# LEACE attack
# ======================================================================

def leace_attack(
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
    svd_tol: float = 0.01,
    base_fwd_pre_hooks: Optional[List] = None,
    base_fwd_hooks: Optional[List] = None,
) -> Dict:
    """
    LEACE concept-erasure attack (Marks et al. 2023).

    1. Collect last-token activations from harmful/harmless prompts.
    2. Fit LEACE per layer to find the optimal linear erasure of the
       harmful/harmless concept.
    3. Apply the erasure as inference hooks and measure refusal survival.

    Returns
    -------
    dict with:
        * ``post_attack_refusal_score`` — mean refusal score after LEACE erasure
        * ``cos_sim_with_original`` — per-layer cosine similarity between
          LEACE's found direction and the original r̂
        * ``max_cos_sim`` — max |cos_sim| across layers
        * ``singular_values`` — per-layer singular values from LEACE
    """
    base_fwd_pre_hooks = base_fwd_pre_hooks or []
    base_fwd_hooks = base_fwd_hooks or []

    num_layers = model.config.num_hidden_layers
    d_model = model.config.hidden_size
    device = next(model.parameters()).device

    print("[LEACE] Collecting activations from harmful prompts …")
    harmful_acts = _collect_last_token_activations(
        model, tokenize_fn, block_modules, harmful_prompts, batch_size,
        base_fwd_pre_hooks=base_fwd_pre_hooks,
        base_fwd_hooks=base_fwd_hooks,
    )
    print("[LEACE] Collecting activations from harmless prompts …")
    benign_acts = _collect_last_token_activations(
        model, tokenize_fn, block_modules, benign_prompts, batch_size,
        base_fwd_pre_hooks=base_fwd_pre_hooks,
        base_fwd_hooks=base_fwd_hooks,
    )

    n_harmful = harmful_acts.shape[0]
    n_benign = benign_acts.shape[0]

    # Stack activations and labels
    # X: (n_harmful + n_benign, d_model) per layer
    # Z: one-hot labels — [1, 0] for harmful, [0, 1] for harmless
    labels_harmful = torch.zeros(n_harmful, 2)
    labels_harmful[:, 0] = 1.0
    labels_benign = torch.zeros(n_benign, 2)
    labels_benign[:, 1] = 1.0

    all_labels = torch.cat([labels_harmful, labels_benign], dim=0)

    # Fit LEACE per layer and build hooks
    leace_results = []
    cos_sims = []
    original_dir = original_direction.float().cpu()
    original_dir = original_dir / (original_dir.norm() + 1e-8)

    print(f"[LEACE] Fitting erasure projections across {num_layers} layers …")
    for ell in range(num_layers):
        # Stack harmful and benign activations at this layer
        x_ell = torch.cat([
            harmful_acts[:, ell, :],
            benign_acts[:, ell, :],
        ], dim=0)  # (n_total, d_model)

        fitter = LeaceFitter(d_model, num_classes=2, svd_tol=svd_tol)
        fitter.update(x_ell, all_labels)
        result = fitter.fit()
        leace_results.append(result)

        # Cosine similarity between LEACE direction and original r̂
        directions = result["directions"].float()
        if directions.shape[0] > 0 and directions.norm() > 1e-8:
            # Take the first (strongest) direction
            leace_dir = directions[0]
            leace_dir = leace_dir / (leace_dir.norm() + 1e-8)
            cs = (leace_dir @ original_dir).abs().item()
        else:
            cs = 0.0
        cos_sims.append(cs)

    print(f"[LEACE] Max |cos_sim| with original r̂: {max(cos_sims):.4f}")
    print(f"[LEACE] Mean |cos_sim| with original r̂: {sum(cos_sims)/len(cos_sims):.4f}")

    # Build erasure hooks
    def make_leace_pre_hook(proj_left, proj_right, bias):
        proj_left_f = proj_left.float()
        proj_right_f = proj_right.float()
        bias_f = bias.float()

        def hook_fn(module, inp):
            if isinstance(inp, tuple):
                activation = inp[0]
            else:
                activation = inp

            pl = proj_left_f.to(activation)
            pr = proj_right_f.to(activation)
            b = bias_f.to(activation)

            # x' = x - proj_left @ proj_right @ (x - bias)
            centered = activation - b
            projection = centered @ pr.T  # (batch, seq, k)
            correction = projection @ pl.T  # (batch, seq, d)
            activation = activation - correction

            if isinstance(inp, tuple):
                return (activation, *inp[1:])
            return activation
        return hook_fn

    def make_leace_output_hook(proj_left, proj_right, bias):
        proj_left_f = proj_left.float()
        proj_right_f = proj_right.float()
        bias_f = bias.float()

        def hook_fn(module, inp, output):
            if isinstance(output, tuple):
                activation = output[0]
            else:
                activation = output

            pl = proj_left_f.to(activation)
            pr = proj_right_f.to(activation)
            b = bias_f.to(activation)

            centered = activation - b
            projection = centered @ pr.T
            correction = projection @ pl.T
            activation = activation - correction

            if isinstance(output, tuple):
                return (activation, *output[1:])
            return activation
        return hook_fn

    attack_fwd_pre_hooks = [
        (block_modules[ell],
         make_leace_pre_hook(
             leace_results[ell]["proj_left"],
             leace_results[ell]["proj_right"],
             leace_results[ell]["bias"],
        ))
        for ell in range(num_layers)
    ]
    attack_fwd_hooks = [
        (attn_modules[ell],
         make_leace_output_hook(
             leace_results[ell]["proj_left"],
             leace_results[ell]["proj_right"],
             leace_results[ell]["bias"],
         ))
        for ell in range(num_layers)
    ] + [
        (mlp_modules[ell],
         make_leace_output_hook(
             leace_results[ell]["proj_left"],
             leace_results[ell]["proj_right"],
             leace_results[ell]["bias"],
         ))
        for ell in range(num_layers)
    ]

    # Measure post-attack refusal score
    print("[LEACE] Measuring post-attack refusal score …")
    post_scores = get_refusal_scores(
        model, harmful_prompts, tokenize_fn, refusal_toks,
        fwd_pre_hooks=list(base_fwd_pre_hooks) + attack_fwd_pre_hooks,
        fwd_hooks=list(base_fwd_hooks) + attack_fwd_hooks,
        batch_size=batch_size,
    )
    mean_score = post_scores.mean().item()
    print(f"[LEACE] Post-LEACE refusal score: {mean_score:.4f}")

    return {
        "post_attack_refusal_score": mean_score,
        "cos_sim_with_original": cos_sims,
        "max_cos_sim": max(cos_sims),
        "mean_cos_sim": sum(cos_sims) / len(cos_sims),
    }
