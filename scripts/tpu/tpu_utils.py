"""
TPU/XLA helpers for the APRS pipeline.

The main entry point is `patch_model_for_xla`, which monkey-patches
`model.generate` so that every call:

  1. Left-pads `input_ids` / `attention_mask` to the smallest bucket that
     fits — this keeps XLA's compiled graph shapes stable.
  2. Uses `cache_implementation="static"` so the KV cache has a fixed shape.
  3. Strips the left-pad from the returned tensor so downstream callers that
     slice with `out[:, input_ids.shape[1]:]` keep working unchanged.

It also handles the soft-opt path (`inputs_embeds=...`, no `input_ids`) via
a manual prefill + decode loop against a `StaticCache`, since HF's generate
does not support static cache with `inputs_embeds`.

Outputs are semantically identical to vanilla `model.generate` — same greedy
tokens, same sampled tokens under a given RNG — the only difference is that
repeat calls with the same bucket avoid re-compilation.
"""

from __future__ import annotations

import os
from contextlib import contextmanager
from typing import Callable, Dict, Iterable, Optional, Sequence, Tuple

import torch


DEFAULT_BUCKETS: Tuple[int, ...] = (512, 1024, 2048, 4096)

_ACTIVE_BUCKETS: Tuple[int, ...] = DEFAULT_BUCKETS


def parse_buckets(value: str | Sequence[int] | None) -> Tuple[int, ...]:
    """Parse an XLA bucket list from CLI/env input."""
    if value is None:
        return DEFAULT_BUCKETS
    if isinstance(value, str):
        buckets = tuple(int(x.strip()) for x in value.split(",") if x.strip())
    else:
        buckets = tuple(int(x) for x in value)
    if not buckets:
        raise ValueError("At least one bucket is required")
    if tuple(sorted(buckets)) != buckets:
        raise ValueError(f"Buckets must be sorted ascending, got {buckets}")
    return buckets


def set_active_buckets(value: str | Sequence[int] | None) -> Tuple[int, ...]:
    """Register the bucket schedule chosen by the pipeline so that helpers in
    other modules (attacks, evaluators) can pick it up without threading the
    list through every call signature."""
    global _ACTIVE_BUCKETS
    _ACTIVE_BUCKETS = parse_buckets(value)
    return _ACTIVE_BUCKETS


def get_active_buckets() -> Tuple[int, ...]:
    """Buckets the pipeline last registered (or DEFAULT_BUCKETS if none)."""
    return _ACTIVE_BUCKETS


def is_xla_env() -> bool:
    """Heuristic: are we running on a TPU/XLA device?"""
    if os.environ.get("PJRT_DEVICE", "").upper() == "TPU":
        return True
    if os.environ.get("XRT_TPU_CONFIG"):
        return True
    try:
        import torch_xla.core.xla_model as xm
        devices = xm.get_xla_supported_devices()
        return any("TPU" in str(device).upper() for device in devices)
    except Exception:
        return False


def _pick_bucket(length: int, buckets: Sequence[int]) -> int:
    for b in buckets:
        if length <= b:
            return b
    raise ValueError(
        f"Input length {length} exceeds largest bucket {buckets[-1]}. "
        f"Raise the top of `buckets` in patch_model_for_xla()."
    )


def _left_pad_1d(x: torch.Tensor, target: int, pad_value) -> torch.Tensor:
    """Left-pad along the last axis to `target`. No-op if already long enough."""
    cur = x.shape[-1]
    if cur >= target:
        return x
    pad_amount = target - cur
    pad_shape = (*x.shape[:-1], pad_amount)
    pad = torch.full(pad_shape, pad_value, dtype=x.dtype, device=x.device)
    return torch.cat([pad, x], dim=-1)


def bucket_pad_tensor(
    x: torch.Tensor,
    *,
    buckets: Sequence[int] = DEFAULT_BUCKETS,
    pad_value=0,
) -> torch.Tensor:
    """Left-pad a tensor's sequence axis to the nearest configured bucket."""
    return _left_pad_1d(x, _pick_bucket(int(x.shape[-1]), buckets), pad_value)


def bucket_pad_batch_encoding(
    inputs,
    tokenizer=None,
    *,
    buckets: Sequence[int] = DEFAULT_BUCKETS,
    loss_mask: Optional[torch.Tensor] = None,
):
    """
    Left-pad a Hugging Face BatchEncoding/dict to a fixed bucket.

    The pipeline uses left-padding tokenizers already, but HF tokenization still
    emits one shape per batch. Padding every forward batch to a small bucket set
    keeps XLA from compiling a new graph for every prompt length. When
    ``loss_mask`` is supplied, it is padded with zeros in lockstep.
    """
    if "input_ids" not in inputs:
        return (inputs, loss_mask) if loss_mask is not None else inputs

    input_ids = inputs["input_ids"]
    target = _pick_bucket(int(input_ids.shape[-1]), buckets)
    if int(input_ids.shape[-1]) == target:
        return (inputs, loss_mask) if loss_mask is not None else inputs

    pad_id = 0
    if tokenizer is not None:
        pad_id = tokenizer.pad_token_id
        if pad_id is None:
            pad_id = tokenizer.eos_token_id
        if pad_id is None:
            pad_id = 0

    inputs["input_ids"] = _left_pad_1d(input_ids, target, pad_id)
    if "attention_mask" in inputs and inputs["attention_mask"] is not None:
        inputs["attention_mask"] = _left_pad_1d(inputs["attention_mask"], target, 0)
    if "token_type_ids" in inputs and inputs["token_type_ids"] is not None:
        inputs["token_type_ids"] = _left_pad_1d(inputs["token_type_ids"], target, 0)
    if "labels" in inputs and inputs["labels"] is not None:
        inputs["labels"] = _left_pad_1d(inputs["labels"], target, -100)

    if loss_mask is not None:
        loss_mask = _left_pad_1d(loss_mask, target, 0)
        return inputs, loss_mask
    return inputs


def make_bucketed_tokenize_fn(
    tokenize_fn: Callable,
    tokenizer,
    *,
    buckets: Sequence[int] = DEFAULT_BUCKETS,
) -> Callable:
    """Wrap a pipeline tokenization function with bucket padding."""
    if getattr(tokenize_fn, "_aprs_bucketed", False):
        return tokenize_fn

    def _wrapped(*args, **kwargs):
        enc = tokenize_fn(*args, **kwargs)
        return bucket_pad_batch_encoding(enc, tokenizer, buckets=buckets)

    _wrapped._aprs_bucketed = True
    _wrapped._aprs_buckets = tuple(buckets)
    _wrapped.__name__ = getattr(tokenize_fn, "__name__", "bucketed_tokenize_fn")
    return _wrapped


def save_weight_snapshot(
    model: torch.nn.Module,
    *,
    names: Optional[Iterable[str]] = None,
) -> Dict[str, torch.Tensor]:
    """
    Save a CPU copy of model parameters for restoring between TPU pipeline legs.

    APRS and several baselines bake rank-one edits into weights. Restoring
    weights from a host snapshot is cheaper and clearer than reloading the full
    HF checkpoint between defenses when running a long v6e sweep.
    """
    wanted = set(names) if names is not None else None
    snapshot: Dict[str, torch.Tensor] = {}
    for name, param in model.named_parameters():
        if wanted is not None and name not in wanted:
            continue
        snapshot[name] = param.detach().cpu().clone()
    return snapshot


def restore_weight_snapshot(
    model: torch.nn.Module,
    snapshot: Dict[str, torch.Tensor],
) -> None:
    """Restore parameters previously captured by ``save_weight_snapshot``."""
    with torch.no_grad():
        for name, param in model.named_parameters():
            if name not in snapshot:
                continue
            src = snapshot[name].to(device=param.device, dtype=param.dtype)
            param.data.copy_(src)
    try:
        import torch_xla.core.xla_model as xm
        xm.mark_step()
    except Exception:
        pass


@contextmanager
def restored_weights(model: torch.nn.Module, snapshot: Dict[str, torch.Tensor]):
    """Context manager that restores a model snapshot on exit."""
    try:
        yield model
    finally:
        restore_weight_snapshot(model, snapshot)


def _pick_token(logits: torch.Tensor, do_sample: bool, temperature: float,
                top_p: float, top_k: int) -> torch.Tensor:
    """Sample or argmax from last-step logits of shape (B, V)."""
    if not do_sample:
        return logits.argmax(dim=-1)
    if temperature != 1.0:
        logits = logits / max(temperature, 1e-5)
    if top_k and top_k > 0:
        top_vals, _ = torch.topk(logits, top_k)
        logits = torch.where(logits < top_vals[..., -1:], torch.full_like(logits, -1e9), logits)
    if top_p and top_p < 1.0:
        sorted_logits, sorted_idx = logits.sort(dim=-1, descending=True)
        probs = sorted_logits.softmax(dim=-1)
        cum = probs.cumsum(dim=-1)
        mask = cum - probs > top_p
        sorted_logits = sorted_logits.masked_fill(mask, -1e9)
        logits = torch.full_like(logits, -1e9).scatter(-1, sorted_idx, sorted_logits)
    probs = logits.softmax(dim=-1)
    return torch.multinomial(probs, num_samples=1).squeeze(-1)


def _mask_position_ids(attention_mask: torch.Tensor) -> torch.Tensor:
    """Position ids that ignore left-padding for RoPE models."""
    pos = attention_mask.long().cumsum(dim=-1) - 1
    return pos.clamp_min(0)


def _eos_mask(tokens: torch.Tensor, eos_token_id) -> torch.Tensor:
    if eos_token_id is None:
        return torch.zeros_like(tokens, dtype=torch.bool)
    eos_ids = eos_token_id if isinstance(eos_token_id, (list, tuple, set)) else [eos_token_id]
    mask = torch.zeros_like(tokens, dtype=torch.bool)
    for eid in eos_ids:
        if eid is not None:
            mask = mask | (tokens == int(eid))
    return mask


def _generation_param(kwargs: dict, name: str, default=None):
    if name in kwargs and kwargs[name] is not None:
        return kwargs[name]
    gen_cfg = kwargs.get("generation_config")
    if gen_cfg is not None:
        value = getattr(gen_cfg, name, None)
        if value is not None:
            return value
    return default


def _generation_settings(kwargs: dict, prompt_len: int, tokenizer_pad_id, tokenizer_eos_id):
    gen_cfg = kwargs.get("generation_config")
    default_max_new = 256
    max_new_tokens = _generation_param(kwargs, "max_new_tokens", None)
    if max_new_tokens is None:
        max_length = _generation_param(kwargs, "max_length", None)
        if max_length is not None:
            max_new_tokens = max(1, int(max_length) - int(prompt_len))
        elif gen_cfg is not None and getattr(gen_cfg, "max_new_tokens", None) is not None:
            max_new_tokens = int(gen_cfg.max_new_tokens)
        else:
            max_new_tokens = default_max_new
    return {
        "max_new_tokens": int(max_new_tokens),
        "do_sample": bool(_generation_param(kwargs, "do_sample", False)),
        "temperature": float(_generation_param(kwargs, "temperature", 1.0) or 1.0),
        "top_p": float(_generation_param(kwargs, "top_p", 1.0) or 1.0),
        "top_k": int(_generation_param(kwargs, "top_k", 0) or 0),
        "pad_token_id": _generation_param(kwargs, "pad_token_id", tokenizer_pad_id),
        "eos_token_id": _generation_param(kwargs, "eos_token_id", tokenizer_eos_id),
    }


def _xla_ids_decode(model, input_ids: torch.Tensor, attention_mask,
                    max_new_tokens: int, do_sample: bool,
                    temperature: float, top_p: float, top_k: int,
                    pad_token_id, eos_token_id,
                    buckets: Sequence[int]) -> torch.Tensor:
    """
    Autoregressive greedy/sampled decode from `input_ids`, using StaticCache
    so every step has a fixed shape. Returns (B, orig_L + max_new_tokens),
    matching HF `generate`'s behaviour with input_ids (prompt + new tokens).
    """
    try:
        import torch_xla.core.xla_model as xm
    except Exception:
        xm = None
    from transformers import StaticCache

    device = input_ids.device
    B, L = input_ids.shape
    bucket = _pick_bucket(L, buckets)
    pad_amount = bucket - L

    padded_ids = _left_pad_1d(input_ids, bucket, pad_token_id if pad_token_id is not None else 0)
    if attention_mask is None:
        attention_mask = torch.ones_like(input_ids)
    padded_mask = _left_pad_1d(attention_mask.to(device=device, dtype=torch.long), bucket, 0)

    total_len = bucket + max_new_tokens
    full_mask = torch.zeros(B, total_len, dtype=torch.long, device=device)
    full_mask[:, :bucket] = padded_mask

    cache = StaticCache(
        config=model.config,
        max_batch_size=B,
        max_cache_len=total_len,
        device=device,
        dtype=next(model.parameters()).dtype,
    )

    real_lengths = padded_mask.sum(dim=1).long()

    # Prefill. Position ids follow non-pad tokens, not bucket columns; this
    # preserves RoPE semantics under left-padding.
    position_ids = _mask_position_ids(padded_mask)
    out = model(
        input_ids=padded_ids,
        attention_mask=full_mask[:, :bucket],
        position_ids=position_ids,
        past_key_values=cache,
        use_cache=True,
    )
    if xm is not None:
        xm.mark_step()
    next_token = _pick_token(out.logits[:, -1, :], do_sample, temperature, top_p, top_k)
    generated = [next_token]
    unfinished = ~_eos_mask(next_token, eos_token_id)

    # First generated token is always part of the returned sequence, including
    # EOS. Later rows that are already finished emit padding with mask 0.
    full_mask[:, bucket] = 1

    for step in range(1, max_new_tokens):
        pos = (real_lengths + step - 1).view(B, 1)
        out = model(
            input_ids=next_token.unsqueeze(-1),
            attention_mask=full_mask[:, :bucket + step],
            position_ids=pos,
            past_key_values=cache,
            use_cache=True,
        )
        sampled = _pick_token(out.logits[:, -1, :], do_sample, temperature, top_p, top_k)
        emitted = unfinished
        fill = pad_token_id if pad_token_id is not None else 0
        next_token = torch.where(emitted, sampled, torch.full_like(sampled, fill))
        just_ended = emitted & _eos_mask(next_token, eos_token_id)
        full_mask[:, bucket + step] = emitted.long()
        generated.append(next_token)
        unfinished = unfinished & ~just_ended
        if xm is not None:
            xm.mark_step()
        if not unfinished.any():
            fill = pad_token_id if pad_token_id is not None else 0
            for _ in range(step + 1, max_new_tokens):
                generated.append(torch.full_like(next_token, fill))
            break

    new_tokens = torch.stack(generated, dim=-1)  # (B, max_new_tokens)
    return torch.cat([input_ids, new_tokens], dim=-1)


def _xla_embed_decode(model, inputs_embeds: torch.Tensor, attention_mask,
                      max_new_tokens: int, do_sample: bool,
                      temperature: float, top_p: float, top_k: int,
                      pad_token_id, eos_token_id,
                      buckets: Sequence[int]) -> torch.Tensor:
    """
    Autoregressive decode starting from `inputs_embeds` (B, L, d).

    Uses a manual prefill + per-step loop so we can keep the KV cache's
    shape static across calls. Returns a tensor of shape (B, max_new_tokens)
    matching HF's behaviour of returning only the newly generated tokens
    when called with inputs_embeds.
    """
    try:
        import torch_xla.core.xla_model as xm
    except Exception:
        xm = None

    from transformers import StaticCache

    device = inputs_embeds.device
    B, L, D = inputs_embeds.shape
    bucket = _pick_bucket(L, buckets)
    pad_amount = bucket - L

    if pad_amount > 0:
        pad_e = torch.zeros(B, pad_amount, D, dtype=inputs_embeds.dtype, device=device)
        inputs_embeds = torch.cat([pad_e, inputs_embeds], dim=1)

    total_len = bucket + max_new_tokens
    full_mask = torch.ones(B, total_len, dtype=torch.long, device=device)
    if pad_amount > 0:
        full_mask[:, :pad_amount] = 0
    if attention_mask is not None:
        # Caller-provided mask covers the original L positions
        full_mask[:, pad_amount:bucket] = attention_mask.to(device=device, dtype=torch.long)

    cache = StaticCache(
        config=model.config,
        max_batch_size=B,
        max_cache_len=total_len,
        device=device,
        dtype=inputs_embeds.dtype,
    )

    real_lengths = full_mask[:, :bucket].sum(dim=1).long()

    # Prefill. Position ids follow non-pad tokens, not bucket columns; this
    # preserves RoPE semantics under left-padding.
    position_ids = _mask_position_ids(full_mask[:, :bucket])
    out = model(
        inputs_embeds=inputs_embeds,
        attention_mask=full_mask[:, :bucket],
        position_ids=position_ids,
        past_key_values=cache,
        use_cache=True,
    )
    if xm is not None:
        xm.mark_step()
    next_token = _pick_token(out.logits[:, -1, :], do_sample, temperature, top_p, top_k)
    generated = [next_token]
    unfinished = ~_eos_mask(next_token, eos_token_id)
    full_mask[:, bucket] = 1

    for step in range(1, max_new_tokens):
        pos = (real_lengths + step - 1).view(B, 1)
        out = model(
            input_ids=next_token.unsqueeze(-1),
            attention_mask=full_mask[:, :bucket + step],
            position_ids=pos,
            past_key_values=cache,
            use_cache=True,
        )
        sampled = _pick_token(out.logits[:, -1, :], do_sample, temperature, top_p, top_k)
        emitted = unfinished
        fill = pad_token_id if pad_token_id is not None else 0
        next_token = torch.where(emitted, sampled, torch.full_like(sampled, fill))
        just_ended = emitted & _eos_mask(next_token, eos_token_id)
        full_mask[:, bucket + step] = emitted.long()
        generated.append(next_token)
        unfinished = unfinished & ~just_ended
        if xm is not None:
            xm.mark_step()
        if not unfinished.any():
            # Still fill remaining slots with pad so shape is stable across calls
            fill = pad_token_id if pad_token_id is not None else 0
            for _ in range(step + 1, max_new_tokens):
                generated.append(torch.full_like(next_token, fill))
            break

    return torch.stack(generated, dim=-1)


def _patch_isin_for_xla():
    """Replace HF's `isin_mps_friendly` (which calls torch.isin) with a
    broadcast-equality version. torch.isin spills vmem on TPU and crashes
    the compiler; the broadcast form compiles cleanly."""
    try:
        from transformers import pytorch_utils as _pu
    except Exception:
        return
    if getattr(_pu, "_xla_isin_patched", False):
        return
    def _isin_xla(elements, test_elements):
        if not isinstance(test_elements, torch.Tensor):
            test_elements = torch.tensor(test_elements, device=elements.device)
        test_elements = test_elements.to(device=elements.device).flatten()
        return (elements.unsqueeze(-1) == test_elements).any(dim=-1)
    _pu.isin_mps_friendly = _isin_xla
    try:
        from transformers.generation import stopping_criteria as _sc
        _sc.isin_mps_friendly = _isin_xla
    except Exception:
        pass
    _pu._xla_isin_patched = True


def patch_model_for_xla(model, tokenizer,
                        buckets: Sequence[int] = DEFAULT_BUCKETS,
                        verbose: bool = True):
    """
    Monkey-patch `model.generate` to be XLA-friendly.

    Safe to call once per model instance. The original `generate` is kept as
    `model._generate_eager` in case a caller needs to bypass the patch.
    """
    if getattr(model, "_xla_patched", False):
        return model

    _patch_isin_for_xla()

    original_generate = model.generate
    model._generate_eager = original_generate

    pad_id = tokenizer.pad_token_id
    if pad_id is None:
        pad_id = tokenizer.eos_token_id

    def patched_generate(*args, **kwargs):
        # Normalize: move any positional input_ids into kwargs
        if args and "input_ids" not in kwargs and "inputs_embeds" not in kwargs:
            kwargs["input_ids"] = args[0]
            args = args[1:]

        inputs_embeds = kwargs.get("inputs_embeds")
        if inputs_embeds is not None:
            settings = _generation_settings(
                kwargs,
                int(inputs_embeds.shape[1]),
                pad_id,
                tokenizer.eos_token_id,
            )
            # Soft-opt path: manual static-cache decode
            return _xla_embed_decode(
                model,
                inputs_embeds=inputs_embeds,
                attention_mask=kwargs.get("attention_mask"),
                max_new_tokens=settings["max_new_tokens"],
                do_sample=settings["do_sample"],
                temperature=settings["temperature"],
                top_p=settings["top_p"],
                top_k=settings["top_k"],
                pad_token_id=settings["pad_token_id"],
                eos_token_id=settings["eos_token_id"],
                buckets=buckets,
            )

        input_ids = kwargs.get("input_ids")
        if input_ids is None:
            return original_generate(*args, **kwargs)

        orig_len = int(input_ids.shape[-1])
        settings = _generation_settings(
            kwargs,
            orig_len,
            pad_id,
            tokenizer.eos_token_id,
        )
        bucket = _pick_bucket(orig_len, buckets)
        if verbose:
            print(f"[xla-generate] orig_len={orig_len} bucket={bucket} "
                  f"max_new={settings['max_new_tokens']}")

        # Bypass HF generate entirely — its stopping-criteria ops spill XLA vmem.
        return _xla_ids_decode(
            model,
            input_ids=input_ids,
            attention_mask=kwargs.get("attention_mask"),
            max_new_tokens=settings["max_new_tokens"],
            do_sample=settings["do_sample"],
            temperature=settings["temperature"],
            top_p=settings["top_p"],
            top_k=settings["top_k"],
            pad_token_id=settings["pad_token_id"],
            eos_token_id=settings["eos_token_id"],
            buckets=buckets,
        )

    model.generate = patched_generate
    model._xla_patched = True
    if verbose:
        print(f"[xla-generate] patched {type(model).__name__} with buckets={list(buckets)}")
    return model
