"""
Nonlinear probe attack — RepE-style MLP probe (Zou et al., NeurIPS 2023).

Trains a 2-layer MLP per transformer layer to classify harmful vs. harmless
activations.  Then uses gradient-based ablation on top-K layers to attack
refusal.  This is strictly stronger than linear probing (mean-diff, LEACE).

Reference:
    Zou, A., Phan, L., Chen, S., Campbell, J., Guo, P., Ren, R., Pan, A.,
    Yin, X., Mazeika, M., Dombrowski, A.-K., Goel, S., Li, N., Byun, Z.,
    Wang, Z., Mallen, A., Basart, S., Koyejo, S., Song, D., Fredrikson, M.,
    Kolter, J.Z., & Hendrycks, D. (2023). Representation Engineering: A
    Top-Down Approach to AI Transparency. NeurIPS 2023.
    https://arxiv.org/abs/2310.01405

Code:
    https://github.com/andyzoujm/representation-engineering
"""

from __future__ import annotations

import os
import sys
from typing import Dict, List, Optional, Tuple

import torch
import torch.nn as nn
import torch.optim as optim

_REFUSAL_DIR = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "refusal_direction")
if _REFUSAL_DIR not in sys.path:
    sys.path.insert(0, _REFUSAL_DIR)

from attacks.evaluate_leace_attack import _collect_last_token_activations
from pipeline.submodules.select_direction import get_refusal_scores


# ── MLP Probe ────────────────────────────────────────────────────────────

class MLPProbe(nn.Module):
    def __init__(self, d_model: int, hidden_dim: int = 256, dropout: float = 0.1):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(d_model, hidden_dim),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, 1),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x).squeeze(-1)


# ── Training / evaluation helpers ────────────────────────────────────────

def _train_nonlinear_probe(
    activations: torch.Tensor,
    labels: torch.Tensor,
    hidden_dim: int = 256,
    epochs: int = 50,
    lr: float = 1e-3,
    val_fraction: float = 0.2,
    seed: int = 42,
) -> Tuple[MLPProbe, float, torch.Tensor, torch.Tensor]:
    """
    Train an MLP probe on activations.

    Returns (probe, val_accuracy, mean, std).
    """
    device = "cuda" if torch.cuda.is_available() else "cpu"
    n = activations.shape[0]
    d_model = activations.shape[1]

    gen = torch.Generator().manual_seed(seed)
    perm = torch.randperm(n, generator=gen)
    n_val = max(2, int(n * val_fraction))
    val_idx, train_idx = perm[:n_val], perm[n_val:]

    mean = activations[train_idx].mean(dim=0)
    std = activations[train_idx].std(dim=0).clamp(min=1e-6)

    x_train = ((activations[train_idx] - mean) / std).to(device)
    y_train = labels[train_idx].float().to(device)
    x_val = ((activations[val_idx] - mean) / std).to(device)
    y_val = labels[val_idx].float().to(device)

    probe = MLPProbe(d_model, hidden_dim).to(device)
    optimizer = optim.Adam(probe.parameters(), lr=lr)
    criterion = nn.BCEWithLogitsLoss()

    best_val_loss = float("inf")
    best_state = None
    patience, patience_counter = 10, 0

    for _ in range(epochs):
        probe.train()
        optimizer.zero_grad()
        loss = criterion(probe(x_train), y_train)
        loss.backward()
        optimizer.step()

        probe.eval()
        with torch.no_grad():
            val_loss = criterion(probe(x_val), y_val).item()

        if val_loss < best_val_loss - 1e-4:
            best_val_loss = val_loss
            best_state = {k: v.clone() for k, v in probe.state_dict().items()}
            patience_counter = 0
        else:
            patience_counter += 1
            if patience_counter >= patience:
                break

    if best_state is not None:
        probe.load_state_dict(best_state)

    probe.eval()
    with torch.no_grad():
        preds = (probe(x_val) > 0).long()
        val_acc = (preds == y_val.long()).float().mean().item()

    return probe, val_acc, mean.cpu(), std.cpu()


def _evaluate_nonlinear_probe(
    probe: MLPProbe,
    activations: torch.Tensor,
    labels: torch.Tensor,
    mean: torch.Tensor,
    std: torch.Tensor,
) -> float:
    device = next(probe.parameters()).device
    x = ((activations - mean) / std).to(device)
    with torch.no_grad():
        preds = (probe(x) > 0).long()
    return (preds == labels.to(device).long()).float().mean().item()


# ── Gradient-based ablation direction extraction ─────────────────────────

def _extract_gradient_direction(
    probe: MLPProbe,
    activations: torch.Tensor,
    mean: torch.Tensor,
    std: torch.Tensor,
) -> torch.Tensor:
    """
    Compute mean gradient of probe output w.r.t. input activations
    for harmful samples (label=1).  Returns a unit direction in activation space.
    """
    device = next(probe.parameters()).device
    x = ((activations - mean) / std).to(device).requires_grad_(True)

    probe.eval()
    out = probe(x)
    out.sum().backward()

    grad = x.grad.mean(dim=0)
    grad = grad * std.to(device)
    norm = grad.norm()
    if norm > 1e-8:
        grad = grad / norm
    return grad.detach()


# ── Main attack function ────────────────────────────────────────────────

def nonlinear_probe_attack(
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
    nlprobe_hidden_dim: int = 256,
    nlprobe_epochs: int = 50,
    nlprobe_top_k: int = 8,
    base_fwd_pre_hooks: Optional[List] = None,
    base_fwd_hooks: Optional[List] = None,
) -> Dict:
    """
    Nonlinear probe attack (RepE-style).

    1. Collect last-token activations from harmful/benign prompts.
    2. Train per-layer MLP probes to classify harmful vs. benign.
    3. On top-K layers (highest probe accuracy), compute gradient-based
       ablation directions and apply them as hooks.
    4. Measure post-attack refusal score.
    """
    base_fwd_pre_hooks = base_fwd_pre_hooks or []
    base_fwd_hooks = base_fwd_hooks or []

    num_layers = model.config.num_hidden_layers
    device = next(model.parameters()).device

    # ── 1. Collect activations ──
    print("[NLProbe] Collecting activations …")
    harmful_acts = _collect_last_token_activations(
        model, tokenize_fn, block_modules, harmful_prompts,
        batch_size=batch_size,
        base_fwd_pre_hooks=base_fwd_pre_hooks,
        base_fwd_hooks=base_fwd_hooks,
    )
    benign_acts = _collect_last_token_activations(
        model, tokenize_fn, block_modules, benign_prompts,
        batch_size=batch_size,
        base_fwd_pre_hooks=base_fwd_pre_hooks,
        base_fwd_hooks=base_fwd_hooks,
    )

    all_acts = torch.cat([harmful_acts, benign_acts], dim=0)
    labels = torch.cat([
        torch.ones(harmful_acts.shape[0]),
        torch.zeros(benign_acts.shape[0]),
    ])

    # ── 2. Train per-layer probes ──
    print(f"[NLProbe] Training probes across {num_layers} layers …")
    per_layer_accuracy = []
    probes = []
    norms = []

    with torch.enable_grad():
        for ell in range(num_layers):
            layer_acts = all_acts[:, ell, :]
            probe, val_acc, mean, std = _train_nonlinear_probe(
                layer_acts, labels,
                hidden_dim=nlprobe_hidden_dim,
                epochs=nlprobe_epochs,
                seed=42 + ell,
            )
            per_layer_accuracy.append(val_acc)
            probes.append(probe)
            norms.append((mean, std))

    mean_acc = sum(per_layer_accuracy) / len(per_layer_accuracy)
    max_acc = max(per_layer_accuracy)
    best_layer = per_layer_accuracy.index(max_acc)

    print(f"[NLProbe] Mean accuracy: {mean_acc:.4f}  Max: {max_acc:.4f} (layer {best_layer})")

    # ── 3. Gradient-based ablation on top-K layers ──
    top_k = min(nlprobe_top_k, num_layers)
    ranked = sorted(range(num_layers), key=lambda i: per_layer_accuracy[i], reverse=True)
    top_layers = ranked[:top_k]

    print(f"[NLProbe] Extracting gradient directions for top-{top_k} layers: {top_layers}")

    attack_fwd_pre_hooks = []
    for ell in top_layers:
        harmful_layer_acts = harmful_acts[:, ell, :]
        mean, std = norms[ell]

        with torch.enable_grad():
            direction = _extract_gradient_direction(probes[ell], harmful_layer_acts, mean, std)

        direction_f = direction.to(dtype=model.dtype, device=device)

        def make_hook(d):
            def hook_fn(module, inp):
                x = inp[0] if isinstance(inp, tuple) else inp
                d_local = d.to(x)
                proj = (x * d_local).sum(dim=-1, keepdim=True)
                x_ablated = x - proj * d_local
                return (x_ablated, *inp[1:]) if isinstance(inp, tuple) else x_ablated
            return hook_fn

        attack_fwd_pre_hooks.append((block_modules[ell], make_hook(direction_f)))

    # ── 4. Measure post-attack refusal ──
    print("[NLProbe] Measuring post-attack refusal …")
    post_scores = get_refusal_scores(
        model, harmful_prompts, tokenize_fn, refusal_toks,
        fwd_pre_hooks=list(base_fwd_pre_hooks) + attack_fwd_pre_hooks,
        fwd_hooks=list(base_fwd_hooks),
        batch_size=batch_size,
    )
    post_attack_refusal = post_scores.mean().item()

    print(f"[NLProbe] Post-attack refusal score: {post_attack_refusal:.4f}")

    return {
        "post_attack_refusal_score": post_attack_refusal,
        "per_layer_accuracy": per_layer_accuracy,
        "mean_probe_accuracy": mean_acc,
        "max_probe_accuracy": max_acc,
        "best_probe_layer": best_layer,
    }
