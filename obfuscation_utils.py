import torch
import torch.nn as nn
from typing import Dict, List, Optional, Tuple
from tqdm import tqdm


class ModelComponents:
    """
    Architecture-agnostic accessor for sublayer components needed by the
    obfuscation defense.  Supports Llama-2/3, Gemma, and Mistral (all share
    the HuggingFace ``model.model.layers`` layout).  Qwen-1 uses a different
    layout and is detected separately.
    """

    def __init__(self, model):
        self.model = model
        self.num_layers = model.config.num_hidden_layers
        self.d_model = model.config.hidden_size
        self._detect_architecture()

    # ------------------------------------------------------------------
    # Architecture detection
    # ------------------------------------------------------------------
    def _detect_architecture(self):
        if hasattr(self.model, "model") and hasattr(self.model.model, "layers"):
            self.arch = "llama"
            self.layers = self.model.model.layers
            self.final_norm = self.model.model.norm
            self.lm_head = self.model.lm_head
        elif hasattr(self.model, "transformer") and hasattr(
            self.model.transformer, "h"
        ):
            self.arch = "qwen"
            self.layers = self.model.transformer.h
            self.final_norm = self.model.transformer.ln_f
            self.lm_head = self.model.lm_head
        else:
            raise ValueError(
                f"Unsupported architecture: {type(self.model).__name__}. "
                "Expected Llama/Gemma/Mistral or Qwen-style model."
            )

    # ------------------------------------------------------------------
    # Sublayer accessors
    # ------------------------------------------------------------------
    def get_attn_layernorm(self, layer_idx: int) -> nn.Module:
        layer = self.layers[layer_idx]
        if self.arch == "llama":
            return layer.input_layernorm
        return layer.ln_1

    def get_mlp_layernorm(self, layer_idx: int) -> nn.Module:
        """LayerNorm whose output feeds the MLP readers (gate / up projections).

        Llama / Mistral / Qwen3:
            ``post_attention_layernorm`` directly feeds the MLP — it sits
            *outside* the residual branch, between the attention-sublayer's
            residual add and the MLP.
        Gemma 2 / Gemma 3:
            Those architectures add a second pair of LayerNorms inside each
            residual branch (``post_attention_layernorm`` and
            ``post_feedforward_layernorm`` wrap the writers).  The LN whose
            output actually feeds the MLP readers is
            ``pre_feedforward_layernorm``.  Using ``post_attention_layernorm``
            here — as Llama does — would point the reader patches at the
            wrong LN and make the aliasing identity break on Gemma.
        """
        layer = self.layers[layer_idx]
        if self.arch == "llama":
            # Prefer Gemma's pre-FF LN when present (Gemma 2/3).  Falls back
            # to post_attention_layernorm for Llama/Mistral/Qwen3.
            if hasattr(layer, "pre_feedforward_layernorm"):
                return layer.pre_feedforward_layernorm
            return layer.post_attention_layernorm
        return layer.ln_2

    def get_attn_reader_projs(self, layer_idx: int) -> List[Tuple[str, nn.Module]]:
        """Return [(name, Linear)] for attention reader projections (Q, K, V)."""
        layer = self.layers[layer_idx]
        if self.arch == "llama":
            attn = layer.self_attn
            return [
                ("q_proj", attn.q_proj),
                ("k_proj", attn.k_proj),
                ("v_proj", attn.v_proj),
            ]
        raise NotImplementedError(f"Attention reader patching not implemented for {self.arch}")

    def get_attn_output_proj(self, layer_idx: int) -> nn.Module:
        layer = self.layers[layer_idx]
        if self.arch == "llama":
            return layer.self_attn.o_proj
        return layer.attn.c_proj

    def get_mlp_reader_projs(self, layer_idx: int) -> List[Tuple[str, nn.Module]]:
        """Return [(name, Linear)] for MLP reader projections (gate, up)."""
        layer = self.layers[layer_idx]
        if self.arch == "llama":
            return [
                ("gate_proj", layer.mlp.gate_proj),
                ("up_proj", layer.mlp.up_proj),
            ]
        return [
            ("w1", layer.mlp.w1),
            ("w2", layer.mlp.w2),
        ]

    def get_mlp_output_proj(self, layer_idx: int) -> nn.Module:
        layer = self.layers[layer_idx]
        if self.arch == "llama":
            return layer.mlp.down_proj
        return layer.mlp.c_proj


# ----------------------------------------------------------------------
# Rank-one update
# ----------------------------------------------------------------------

def rank_one_update(
    W: torch.Tensor,
    x: torch.Tensor,
    target: torch.Tensor,
) -> torch.Tensor:
    """
    Apply a rank-one update to *W* so that ``W_new @ x == target`` (exactly
    for this specific *x*).

    Formula:  W_new = W + outer(target - W @ x, x) / (x @ x)

    All arithmetic is done in float32 for numerical stability; the result is
    cast back to *W*'s original dtype.
    """
    orig_dtype = W.dtype
    W_f = W.float()
    x_f = x.float()
    target_f = target.float()

    current = W_f @ x_f
    delta = target_f - current
    W_f = W_f + torch.outer(delta, x_f) / (x_f @ x_f)
    return W_f.to(orig_dtype)


def rank_k_reader_correction(
    Lp: torch.Tensor,
    Lc: torch.Tensor,
    k: int,
    eps: float = 1e-8,
) -> Optional[torch.Tensor]:
    """
    Compute a rank-k reader correction matrix.

    Given per-probe post-LayerNorm activations ``Lp`` (polluted) and ``Lc``
    (clean), each of shape ``(n_probes, d_in)``, return a matrix ``M`` of
    shape ``(d_in, d_in)`` such that for any reader weight ``W`` of shape
    ``(d_out, d_in)``:

        W_new = W + W @ M

    minimises the least-squares residual

        Σᵢ ‖(W + ΔW) · Lpᵢ − W · Lcᵢ‖²

    over rank-k corrections ``ΔW``, restricted to the top-k right-singular
    subspace of ``Lp``.

    Closed form
    -----------
    Let ``Lp = U S Vh`` (full SVD).  Truncate at ``k`` and define

        M = (Lc − Lp)ᵀ · U_k · diag(1/S_k) · Vh_k

    Then ``ΔW = W M`` is the unique rank-≤k solution that interpolates
    ``W_new · Lpᵢ = W · Lcᵢ`` exactly along the top-k row-span of ``Lp`` and
    leaves W unchanged on the orthogonal complement.

    Reduces exactly to ``rank_one_update(W, Lp[0], W @ Lc[0])`` when
    ``n_probes == 1`` and ``k == 1``: in that case ``U_k=[[1]]``,
    ``S_k=[‖Lp‖]``, ``Vh_k = Lp/‖Lp‖``, so
    ``M = outer(Lc−Lp, Lp) / ‖Lp‖²`` and
    ``W M = outer(W·(Lc−Lp), Lp) / ‖Lp‖²``.

    Parameters
    ----------
    Lp, Lc : per-probe post-LayerNorm activations of shape ``(n, d_in)`` (or
        ``(d_in,)`` which is treated as a single probe).
    k      : target rank.  The effective rank used is
             ``min(k, n, rank(Lp))``.
    eps    : numerical threshold (relative to the leading singular value)
             below which singular values are treated as zero.

    Returns
    -------
    ``M`` of shape ``(d_in, d_in)`` in float32 on the same device as ``Lp``,
    or ``None`` if pollution is below threshold or ``Lp`` is degenerate.  In
    those cases the caller should leave ``W`` untouched.
    """
    Lp_f = Lp.float()
    Lc_f = Lc.float()
    if Lp_f.dim() == 1:
        Lp_f = Lp_f.unsqueeze(0)
        Lc_f = Lc_f.unsqueeze(0)
    if Lp_f.shape != Lc_f.shape:
        raise ValueError(
            f"Lp and Lc must have matching shape; got {Lp_f.shape} vs "
            f"{Lc_f.shape}"
        )

    diff = Lc_f - Lp_f                       # (n, d_in)
    if diff.norm().item() < eps:
        return None

    try:
        U, S, Vh = torch.linalg.svd(Lp_f, full_matrices=False)
    except Exception:
        return None
    if S.numel() == 0 or S[0].item() < eps:
        return None

    rel_thresh = eps * S[0].item()
    avail = int((S > rel_thresh).sum().item())
    keff = int(min(max(int(k), 1), Lp_f.shape[0], avail))
    if keff == 0:
        return None

    Vh_k = Vh[:keff]               # (keff, d_in)
    S_k_inv = (1.0 / S[:keff])     # (keff,)
    U_k = U[:, :keff]              # (n, keff)

    # M = diffᵀ @ U_k @ diag(1/S_k) @ Vh_k  →  (d_in, d_in)
    M = (diff.T @ U_k) * S_k_inv.unsqueeze(0)
    M = M @ Vh_k
    return M


# ----------------------------------------------------------------------
# Calibration activation collection
# ----------------------------------------------------------------------

def collect_calibration_activations(
    model: nn.Module,
    components: ModelComponents,
    harmful_prompts: List[str],
    tokenize_fn,
    num_prompts: int = 32,
    harmless_prompts: Optional[List[str]] = None,
    harmless_ratio: float = 0.5,
    explicit_prompts: Optional[List[str]] = None,
    forward_batch_size: int = 16,
) -> Dict[str, torch.Tensor]:
    """
    Run forward passes and return **averaged** sublayer activations at the
    last token position, used to anchor the rank-one weight edits.

    By default uses a 50/50 mix of harmful and harmless prompts so the
    calibration point sits near the model's average operating point rather
    than purely in the refusal regime.  This reduces reader-patch drift for
    harmless inputs while keeping the writer patches well-anchored.

    Parameters
    ----------
    harmful_prompts : prompts that trigger refusal.
    harmless_prompts : benign prompts.  If None, calibrate on harmful only
        (original behaviour).
    harmless_ratio : fraction of the calibration batch to draw from
        harmless_prompts (default 0.5 = equal mix).

    Collected keys per layer ``ℓ``:
        * ``layer_{ℓ}_attn_ln_input``  — residual stream before attention
        * ``layer_{ℓ}_attn_o_input``   — attention output before W_O projection
        * ``layer_{ℓ}_mlp_ln_input``   — residual stream before MLP
        * ``layer_{ℓ}_mlp_down_input`` — MLP hidden state before W_down projection

    Plus ``final_ln_input`` — residual stream at the end of the model.
    """
    device = next(model.parameters()).device
    num_layers = components.num_layers

    # Build the mixed prompt list, unless the caller provides the exact probe
    # prompts to average over.
    if explicit_prompts is not None:
        prompts = list(explicit_prompts[:num_prompts])
    elif harmless_prompts is not None and harmless_ratio > 0:
        n_harmless = int(num_prompts * harmless_ratio)
        n_harmful = num_prompts - n_harmless
        prompts = (
            harmful_prompts[:n_harmful] +
            harmless_prompts[:n_harmless]
        )
    else:
        prompts = harmful_prompts[:num_prompts]

    # Accumulator (float64 for numerical stability when averaging)
    accum: Dict[str, torch.Tensor] = {}
    count = 0

    def _make_hook(key: str):
        """Create a hook that accumulates the module's *input* at the last-token position.

        Tokenisers are configured with ``padding_side='left'`` so that the
        last token of every sample sits at index ``-1`` regardless of
        per-sample length.  The hook therefore extracts ``x[:, -1, :]`` and
        sums across the batch dim.
        """
        def hook_fn(module, inp, output):
            x = inp[0] if isinstance(inp, tuple) else inp
            if x.dim() == 2:
                x = x.unsqueeze(0)
            batch_vecs = x[:, -1, :].detach().float().cpu()  # (B, d)
            if key not in accum:
                accum[key] = torch.zeros_like(batch_vecs[0], dtype=torch.float64)
            accum[key] += batch_vecs.sum(dim=0).double()
        return hook_fn

    # Register hooks on every layer
    hooks: List[torch.utils.hooks.RemovableHook] = []
    for ell in range(num_layers):
        layer = components.layers[ell]

        # Attention LayerNorm input  →  residual stream before attention
        hooks.append(
            components.get_attn_layernorm(ell).register_forward_hook(
                _make_hook(f"layer_{ell}_attn_ln_input")
            )
        )
        # W_O input  →  attention mechanism output before projection
        hooks.append(
            components.get_attn_output_proj(ell).register_forward_hook(
                _make_hook(f"layer_{ell}_attn_o_input")
            )
        )
        # MLP LayerNorm input  →  residual stream after attention
        hooks.append(
            components.get_mlp_layernorm(ell).register_forward_hook(
                _make_hook(f"layer_{ell}_mlp_ln_input")
            )
        )
        # W_down input  →  MLP hidden state (gate * up)
        hooks.append(
            components.get_mlp_output_proj(ell).register_forward_hook(
                _make_hook(f"layer_{ell}_mlp_down_input")
            )
        )

    # Final LayerNorm input  →  residual stream at end
    hooks.append(
        components.final_norm.register_forward_hook(
            _make_hook("final_ln_input")
        )
    )

    # Forward passes — batched.  Tokenisers are left-padded so per-sample
    # last-token activations are extracted by indexing ``[:, -1, :]``.
    if explicit_prompts is not None:
        print(f"[calibration] {len(prompts)} explicit prompts (batch={forward_batch_size})")
    else:
        n_harmful_used = len(harmful_prompts[:num_prompts if harmless_prompts is None else num_prompts - int(num_prompts * harmless_ratio)])
        n_harmless_used = len(prompts) - n_harmful_used
        print(f"[calibration] {len(prompts)} prompts: {n_harmful_used} harmful + {n_harmless_used} harmless (batch={forward_batch_size})")
    with torch.no_grad():
        for i in tqdm(range(0, len(prompts), forward_batch_size),
                      desc="Calibration fwd passes"):
            batch = prompts[i:i + forward_batch_size]
            inputs = tokenize_fn(instructions=batch)
            model(
                input_ids=inputs.input_ids.to(device),
                attention_mask=inputs.attention_mask.to(device),
            )
            count += len(batch)

    # Remove hooks
    for h in hooks:
        h.remove()

    # Average, cast to float32 (MPS doesn't support float64), move to model device
    for key in accum:
        accum[key] = (accum[key] / count).float().to(device)

    return accum


def collect_writer_output_refusal_directions(
    model: nn.Module,
    components: ModelComponents,
    harmful_prompts: List[str],
    harmless_prompts: List[str],
    tokenize_fn,
    num_prompts: int = 32,
    forward_batch_size: int = 16,
) -> Dict[str, Dict[int, torch.Tensor]]:
    """
    Estimate local refusal directions at each writer output.

    The upstream refusal-direction extractor measures block-input residual
    directions.  This helper measures the writer contributions themselves:

        * ``attn[ell]`` = mean_harmful(W_O^ell output) -
          mean_harmless(W_O^ell output)
        * ``mlp[ell]`` = mean_harmful(W_down^ell output) -
          mean_harmless(W_down^ell output)

    All returned vectors live in residual-stream ``d_model`` space and can be
    used directly as the projection direction for the corresponding writer
    rank-one update.
    """

    def _collect(prompts: List[str], desc: str) -> Dict[str, torch.Tensor]:
        device = next(model.parameters()).device
        prompts_to_use = prompts[:num_prompts]
        accum: Dict[str, torch.Tensor] = {}
        count = 0

        def _make_hook(key: str):
            def hook_fn(module, inp, output):
                y = output[0] if isinstance(output, tuple) else output
                if y.dim() == 2:
                    y = y.unsqueeze(0)
                batch_vecs = y[:, -1, :].detach().float().cpu()
                if key not in accum:
                    accum[key] = torch.zeros_like(batch_vecs[0], dtype=torch.float64)
                accum[key] += batch_vecs.sum(dim=0).double()
            return hook_fn

        hooks: List[torch.utils.hooks.RemovableHook] = []
        for ell in range(components.num_layers):
            hooks.append(
                components.get_attn_output_proj(ell).register_forward_hook(
                    _make_hook(f"attn_{ell}")
                )
            )
            hooks.append(
                components.get_mlp_output_proj(ell).register_forward_hook(
                    _make_hook(f"mlp_{ell}")
                )
            )

        try:
            with torch.no_grad():
                for i in tqdm(range(0, len(prompts_to_use), forward_batch_size),
                              desc=desc):
                    batch = prompts_to_use[i:i + forward_batch_size]
                    inputs = tokenize_fn(instructions=batch)
                    model(
                        input_ids=inputs.input_ids.to(device),
                        attention_mask=inputs.attention_mask.to(device),
                    )
                    count += len(batch)
        finally:
            for h in hooks:
                h.remove()

        for key in accum:
            accum[key] = (accum[key] / max(count, 1)).float()
        return accum

    n_harmful = min(num_prompts, len(harmful_prompts))
    n_harmless = min(num_prompts, len(harmless_prompts))
    n = min(n_harmful, n_harmless)
    if n <= 0:
        raise ValueError("writer-output direction extraction requires harmful and harmless prompts")

    harmful = _collect(harmful_prompts[:n], "Writer-output dirs: harmful")
    harmless = _collect(harmless_prompts[:n], "Writer-output dirs: harmless")

    directions: Dict[str, Dict[int, torch.Tensor]] = {"attn": {}, "mlp": {}}
    for ell in range(components.num_layers):
        directions["attn"][ell] = harmful[f"attn_{ell}"] - harmless[f"attn_{ell}"]
        directions["mlp"][ell] = harmful[f"mlp_{ell}"] - harmless[f"mlp_{ell}"]
    return directions


def _class_gap_pca_directions(
    harmful: torch.Tensor,
    harmless: torch.Tensor,
    k: int,
    fallback: torch.Tensor,
) -> torch.Tensor:
    """Return up to k CAST-style harmful-minus-harmless PCA directions."""
    k = max(1, int(k))
    device = fallback.device
    harmful = harmful.float()
    harmless = harmless.float()
    fallback = fallback.float().to(device)
    fallback = fallback / (fallback.norm() + 1e-8)

    n = min(harmful.shape[0], harmless.shape[0])
    if n <= 1:
        return fallback.unsqueeze(0)
    harmful = harmful[:n]
    harmless = harmless[:n]

    h_mean = harmful.mean(dim=0, keepdim=True)
    b_mean = harmless.mean(dim=0, keepdim=True)
    mean_gap = (h_mean - b_mean).squeeze(0).to(device)

    # CAST-style contrastive PCA: each row is one harmful-minus-harmless
    # activation difference, then centered before SVD.
    raw_diffs = harmful - harmless
    diffs = raw_diffs - raw_diffs.mean(dim=0, keepdim=True)
    if diffs.norm().item() <= 1e-6 * max(raw_diffs.norm().item(), 1.0):
        if mean_gap.norm().item() > 1e-8:
            return (mean_gap / mean_gap.norm()).unsqueeze(0)
        return fallback.unsqueeze(0)

    try:
        _, _, vh = torch.linalg.svd(diffs.float(), full_matrices=False)
    except Exception:
        if mean_gap.norm().item() > 1e-8:
            return (mean_gap / mean_gap.norm()).unsqueeze(0)
        return fallback.unsqueeze(0)
    dirs = vh[:min(k, vh.shape[0])].to(device)
    dirs = dirs / (dirs.norm(dim=-1, keepdim=True) + 1e-8)

    # Fix arbitrary SVD sign so the first component agrees with the mean gap.
    if mean_gap.norm().item() > 1e-8 and (dirs[0] @ mean_gap) < 0:
        dirs[0] = -dirs[0]
    return dirs


def collect_writer_output_refusal_subspaces(
    model: nn.Module,
    components: ModelComponents,
    harmful_prompts: List[str],
    harmless_prompts: List[str],
    tokenize_fn,
    num_prompts: int = 32,
    num_directions: int = 1,
    fallback_attn: Optional[Dict[int, torch.Tensor]] = None,
    fallback_mlp: Optional[Dict[int, torch.Tensor]] = None,
    layers: Optional[List[int]] = None,
    forward_batch_size: int = 16,
) -> Dict[str, Dict[int, torch.Tensor]]:
    """
    Estimate rank-k local refusal subspaces at writer outputs.

    Returned tensors have shape ``(k_layer, d_model)`` and are unit-normalized.
    The directions are CAST-style principal components of paired
    harmful-minus-harmless writer-output differences.  If those paired
    differences have negligible centered variation, this falls back to the
    normalized mean harmful-minus-harmless gap.
    """

    def _collect(prompts: List[str], desc: str) -> Dict[str, List[torch.Tensor]]:
        device = next(model.parameters()).device
        prompts_to_use = prompts[:num_prompts]
        values: Dict[str, List[torch.Tensor]] = {}

        def _make_hook(key: str):
            def hook_fn(module, inp, output):
                y = output[0] if isinstance(output, tuple) else output
                if y.dim() == 2:
                    y = y.unsqueeze(0)
                batch_vecs = y[:, -1, :].detach().float().cpu()  # (B, d)
                lst = values.setdefault(key, [])
                for v in batch_vecs:
                    lst.append(v)
            return hook_fn

        selected_layers = layers if layers is not None else list(range(components.num_layers))
        hooks: List[torch.utils.hooks.RemovableHook] = []
        for ell in selected_layers:
            hooks.append(
                components.get_attn_output_proj(ell).register_forward_hook(
                    _make_hook(f"attn_{ell}")
                )
            )
            hooks.append(
                components.get_mlp_output_proj(ell).register_forward_hook(
                    _make_hook(f"mlp_{ell}")
                )
            )

        try:
            with torch.no_grad():
                for i in tqdm(range(0, len(prompts_to_use), forward_batch_size),
                              desc=desc):
                    batch = prompts_to_use[i:i + forward_batch_size]
                    inputs = tokenize_fn(instructions=batch)
                    model(
                        input_ids=inputs.input_ids.to(device),
                        attention_mask=inputs.attention_mask.to(device),
                    )
        finally:
            for h in hooks:
                h.remove()

        return values

    n_harmful = min(num_prompts, len(harmful_prompts))
    n_harmless = min(num_prompts, len(harmless_prompts))
    n = min(n_harmful, n_harmless)
    if n <= 0:
        raise ValueError("writer-output subspace extraction requires harmful and harmless prompts")

    selected_layers = layers if layers is not None else list(range(components.num_layers))
    harmful = _collect(harmful_prompts[:n], "Writer-output PCA: harmful")
    harmless = _collect(harmless_prompts[:n], "Writer-output PCA: harmless")

    device = next(model.parameters()).device
    subspaces: Dict[str, Dict[int, torch.Tensor]] = {"attn": {}, "mlp": {}}
    for ell in selected_layers:
        for kind, fallback_map in (("attn", fallback_attn), ("mlp", fallback_mlp)):
            key = f"{kind}_{ell}"
            fallback = None
            if fallback_map is not None:
                fallback = fallback_map.get(ell)
            if fallback is None:
                fallback = harmful[key][0].to(device)
            h = torch.stack(harmful[key])
            b = torch.stack(harmless[key])
            subspaces[kind][ell] = _class_gap_pca_directions(
                h, b, num_directions, fallback.to(device)
            )
    return subspaces


def writer_output_direction_cosine_summary(
    reference_dirs: Dict[str, Dict[int, torch.Tensor]],
    measured_dirs: Dict[str, Dict[int, torch.Tensor]],
    pertinent_layers: Optional[List[int]] = None,
) -> Dict[str, object]:
    """
    Compare writer-output refusal directions before and after defense.

    ``reference_dirs`` and ``measured_dirs`` should have the same structure as
    ``collect_writer_output_refusal_directions``.  Summary means are computed
    over ``pertinent_layers`` when provided; max values are computed over all
    layers as a quick leakage diagnostic.
    """
    layers = sorted(reference_dirs["attn"].keys())
    selected = pertinent_layers if pertinent_layers else layers

    def _cos(kind: str, ell: int) -> float:
        ref = reference_dirs[kind][ell].float()
        cur = measured_dirs[kind][ell].float()
        ref = ref / (ref.norm() + 1e-8)
        cur = cur / (cur.norm() + 1e-8)
        return abs(float(ref @ cur))

    attn_cos = torch.tensor([_cos("attn", ell) for ell in layers])
    mlp_cos = torch.tensor([_cos("mlp", ell) for ell in layers])
    layer_to_idx = {ell: i for i, ell in enumerate(layers)}
    selected_indices = [layer_to_idx[ell] for ell in selected if ell in layer_to_idx]
    if not selected_indices:
        selected_indices = list(range(len(layers)))

    attn_selected = attn_cos[selected_indices]
    mlp_selected = mlp_cos[selected_indices]
    both_selected = torch.cat([attn_selected, mlp_selected])
    both_all = torch.cat([attn_cos, mlp_cos])

    return {
        "writer_attn_cos_similarities": attn_cos,
        "writer_mlp_cos_similarities": mlp_cos,
        "writer_attn_avg_cos_sim": attn_selected.mean().item(),
        "writer_mlp_avg_cos_sim": mlp_selected.mean().item(),
        "writer_output_avg_cos_sim": both_selected.mean().item(),
        "writer_attn_max_cos_sim": attn_cos.max().item(),
        "writer_mlp_max_cos_sim": mlp_cos.max().item(),
        "writer_output_max_cos_sim": both_all.max().item(),
    }


# ----------------------------------------------------------------------
# Empirical probe of the residual stream in the (partially) patched model
# ----------------------------------------------------------------------

def probe_residual_stream(
    model: nn.Module,
    components: "ModelComponents",
    keys: List[str],
    prompts: List[str],
    tokenize_fn,
    return_per_prompt: bool = False,
    forward_batch_size: int = 16,
) -> Dict[str, torch.Tensor]:
    """
    Run forward passes through the *current* model state (with whatever patches
    have been applied so far) and return last-token activations at the
    specified hook points.

    Unlike ``collect_calibration_activations``, this is intended to be called
    repeatedly during iterative patching to capture the *actual* residual
    stream at each reader input, rather than an analytical estimate.  This is
    necessary for architectures that apply LayerNorms inside the residual
    branch (e.g. Gemma 2/3's ``post_attention_layernorm`` /
    ``post_feedforward_layernorm``), where the pollution produced by a writer
    patch is nonlinearly transformed before entering the residual stream.

    Supported keys:
        * ``layer_{l}_attn_ln_input`` — residual stream before attention LN
        * ``layer_{l}_mlp_ln_input``  — residual stream before MLP LN
        * ``final_ln_input``          — residual stream before the final LN

    Parameters
    ----------
    return_per_prompt : if False (default) returns averaged activations of
        shape ``(d_model,)`` per key.  If True, returns stacked per-prompt
        activations of shape ``(n_prompts, d_model)`` per key, ordered to
        match the input ``prompts`` list.  Required for the rank-k reader
        correction, which needs the full probe matrix to take an SVD.
    """
    device = next(model.parameters()).device
    accum: Dict[str, torch.Tensor] = {}
    per_prompt: Dict[str, List[torch.Tensor]] = {}
    count = 0

    def _make_hook(key: str):
        def hook_fn(module, inp, output):
            x = inp[0] if isinstance(inp, tuple) else inp
            if x.dim() == 2:
                x = x.unsqueeze(0)
            batch_vecs = x[:, -1, :].detach().float().cpu()  # (B, d)
            if return_per_prompt:
                lst = per_prompt.setdefault(key, [])
                for v in batch_vecs:
                    lst.append(v)
            else:
                if key not in accum:
                    accum[key] = torch.zeros_like(batch_vecs[0], dtype=torch.float64)
                accum[key] += batch_vecs.sum(dim=0).double()
        return hook_fn

    hooks: List = []
    for key in keys:
        if key == "final_ln_input":
            hooks.append(
                components.final_norm.register_forward_hook(_make_hook(key))
            )
        elif "_attn_ln_input" in key:
            ell = int(key.split("_")[1])
            hooks.append(
                components.get_attn_layernorm(ell).register_forward_hook(
                    _make_hook(key)
                )
            )
        elif "_mlp_ln_input" in key:
            ell = int(key.split("_")[1])
            hooks.append(
                components.get_mlp_layernorm(ell).register_forward_hook(
                    _make_hook(key)
                )
            )
        else:
            for h in hooks:
                h.remove()
            raise ValueError(f"Unsupported probe key: {key}")

    try:
        with torch.no_grad():
            for i in range(0, len(prompts), forward_batch_size):
                batch = prompts[i:i + forward_batch_size]
                inputs = tokenize_fn(instructions=batch)
                model(
                    input_ids=inputs.input_ids.to(device),
                    attention_mask=inputs.attention_mask.to(device),
                )
                count += len(batch)
    finally:
        for h in hooks:
            h.remove()

    if return_per_prompt:
        result: Dict[str, torch.Tensor] = {}
        for key, vecs in per_prompt.items():
            result[key] = torch.stack(vecs).float().to(device)
        return result

    for key in accum:
        accum[key] = (accum[key] / max(count, 1)).float().to(device)

    return accum


# ----------------------------------------------------------------------
# Random alias generation
# ----------------------------------------------------------------------

def generate_random_alias(
    d: int,
    epsilon: float,
    device: torch.device,
    generator: torch.Generator,
) -> torch.Tensor:
    """
    Generate a zero-mean random vector scaled to standard deviation *epsilon*.

    Zero-mean minimises LayerNorm centering distortion; small std keeps
    the pollution negligible relative to typical residual-stream magnitudes.
    """
    z = torch.randn(d, device=device, generator=generator)
    z = z - z.mean()
    z = z / z.std() * epsilon
    return z
