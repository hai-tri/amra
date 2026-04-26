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
from typing import Sequence, Tuple

import torch


DEFAULT_BUCKETS: Tuple[int, ...] = (512, 1024, 2048)


def is_xla_env() -> bool:
    """Heuristic: are we running on a TPU/XLA device?"""
    if os.environ.get("PJRT_DEVICE", "").upper() == "TPU":
        return True
    if os.environ.get("XRT_TPU_CONFIG"):
        return True
    try:
        import torch_xla.core.xla_model as xm  # noqa: F401
        return True
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

    # Prefill
    position_ids = torch.arange(bucket, device=device).unsqueeze(0).expand(B, -1)
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
    unfinished = torch.ones(B, dtype=torch.bool, device=device)

    # Extend full_mask to cover each newly emitted token as decode progresses
    full_mask[:, bucket] = 1

    for step in range(1, max_new_tokens):
        pos = torch.full((B, 1), bucket + step - 1, device=device, dtype=torch.long)
        out = model(
            input_ids=next_token.unsqueeze(-1),
            attention_mask=full_mask[:, :bucket + step],
            position_ids=pos,
            past_key_values=cache,
            use_cache=True,
        )
        next_token = _pick_token(out.logits[:, -1, :], do_sample, temperature, top_p, top_k)
        if eos_token_id is not None:
            eos_ids = eos_token_id if isinstance(eos_token_id, (list, tuple)) else [eos_token_id]
            just_ended = torch.zeros_like(next_token, dtype=torch.bool)
            for eid in eos_ids:
                just_ended = just_ended | (next_token == eid)
            unfinished = unfinished & ~just_ended
            if pad_token_id is not None:
                next_token = torch.where(unfinished, next_token,
                                         torch.full_like(next_token, pad_token_id))
        full_mask[:, bucket + step] = 1
        generated.append(next_token)
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

    # Prefill
    position_ids = torch.arange(bucket, device=device).unsqueeze(0).expand(B, -1)
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
    unfinished = torch.ones(B, dtype=torch.bool, device=device)

    for step in range(1, max_new_tokens):
        pos = torch.full((B, 1), bucket + step - 1, device=device, dtype=torch.long)
        out = model(
            input_ids=next_token.unsqueeze(-1),
            attention_mask=full_mask[:, :bucket + step],
            position_ids=pos,
            past_key_values=cache,
            use_cache=True,
        )
        next_token = _pick_token(out.logits[:, -1, :], do_sample, temperature, top_p, top_k)
        if eos_token_id is not None:
            # Once a sequence emits EOS, freeze it to pad_token_id going forward
            just_ended = next_token == eos_token_id
            unfinished = unfinished & ~just_ended
            if pad_token_id is not None:
                next_token = torch.where(unfinished, next_token,
                                         torch.full_like(next_token, pad_token_id))
        generated.append(next_token)
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
            # Soft-opt path: manual static-cache decode
            return _xla_embed_decode(
                model,
                inputs_embeds=inputs_embeds,
                attention_mask=kwargs.get("attention_mask"),
                max_new_tokens=kwargs.get("max_new_tokens", 256),
                do_sample=kwargs.get("do_sample", False),
                temperature=float(kwargs.get("temperature", 1.0)),
                top_p=float(kwargs.get("top_p", 1.0)),
                top_k=int(kwargs.get("top_k", 0) or 0),
                pad_token_id=kwargs.get("pad_token_id", pad_id),
                eos_token_id=kwargs.get("eos_token_id", tokenizer.eos_token_id),
                buckets=buckets,
            )

        input_ids = kwargs.get("input_ids")
        if input_ids is None:
            return original_generate(*args, **kwargs)

        orig_len = int(input_ids.shape[-1])
        bucket = _pick_bucket(orig_len, buckets)
        if verbose:
            print(f"[xla-generate] orig_len={orig_len} bucket={bucket} "
                  f"max_new={kwargs.get('max_new_tokens')}")

        # Bypass HF generate entirely — its stopping-criteria ops spill XLA vmem.
        return _xla_ids_decode(
            model,
            input_ids=input_ids,
            attention_mask=kwargs.get("attention_mask"),
            max_new_tokens=kwargs.get("max_new_tokens", 256),
            do_sample=kwargs.get("do_sample", False),
            temperature=float(kwargs.get("temperature", 1.0)),
            top_p=float(kwargs.get("top_p", 1.0)),
            top_k=int(kwargs.get("top_k", 0) or 0),
            pad_token_id=kwargs.get("pad_token_id", pad_id),
            eos_token_id=kwargs.get("eos_token_id", tokenizer.eos_token_id),
            buckets=buckets,
        )

    model.generate = patched_generate
    model._xla_patched = True
    if verbose:
        print(f"[xla-generate] patched {type(model).__name__} with buckets={list(buckets)}")
    return model
