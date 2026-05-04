"""
SoftOpt (soft-prompt optimization) attack evaluation.

Adapted from GraySwanAI/circuit-breakers. This is a white-box attack that
optimizes continuous input embeddings to elicit harmful completions, then
scores them with HarmBench's Llama-2-13b classifier.

Usage standalone::

    python evaluate_softopt.py \
        --model_path meta-llama/Meta-Llama-3-8B-Instruct \
        --benchmark data/harmbench_test_std.json

Or call from the pipeline via ``run_softopt_evaluation()``.
"""

import argparse
import gc
import json
import os
import sys
from dataclasses import dataclass, field
from typing import List, Optional, Dict

import torch
import numpy as np
from tqdm import tqdm
from transformers import AutoModelForCausalLM, AutoTokenizer, set_seed

# Add refusal_direction and repo root to path
_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
_REFUSAL_DIR = os.path.join(_ROOT, "refusal_direction")
if _REFUSAL_DIR not in sys.path:
    sys.path.insert(0, _REFUSAL_DIR)
if _ROOT not in sys.path:
    sys.path.insert(0, _ROOT)

from pipeline.submodules.evaluate_jailbreak import substring_matching_judge_fn
from pipeline.utils.hook_utils import add_hooks
from device_utils import (
    empty_cache as _dev_empty_cache,
    get_device_str as _dev_get_device_str,
    load_model_for_device as _dev_load_model,
)

try:
    from scripts.tpu.tpu_utils import (
        DEFAULT_BUCKETS,
        _mask_position_ids,
        get_active_buckets,
        is_xla_env,
    )
except Exception:
    DEFAULT_BUCKETS = (512, 1024, 2048, 4096)
    def _mask_position_ids(attention_mask):
        pos = attention_mask.long().cumsum(dim=-1) - 1
        return pos.clamp_min(0)
    def is_xla_env():
        return False
    def get_active_buckets():
        return DEFAULT_BUCKETS


# ---------------------------------------------------------------------------
# SoftOpt attack
# ---------------------------------------------------------------------------

@dataclass
class SoftOptConfig:
    num_steps: int = 500
    optim_str_init: str = "x x x x x x x x x x x x x x x x x x x x"
    lr: float = 0.001
    early_stop_loss: float = 0.01
    seed: int = 42
    device: str = field(default_factory=_dev_get_device_str)


def mellowmax(t: torch.Tensor, alpha=1.0, dim=-1):
    return (1.0 / alpha) * (
        torch.logsumexp(alpha * t, dim=dim)
        - torch.log(torch.tensor(t.shape[-1], dtype=t.dtype, device=t.device))
    )


def _maybe_bucket_embeds(inputs_embeds: torch.Tensor):
    """Left-pad embedding sequences on XLA to reduce SoftOpt recompiles."""
    length = int(inputs_embeds.shape[1])
    mask = torch.ones(
        inputs_embeds.shape[:2],
        dtype=torch.long,
        device=inputs_embeds.device,
    )
    if not is_xla_env():
        return inputs_embeds, mask, 0
    target = next((b for b in get_active_buckets() if length <= b), None)
    if target is None:
        return inputs_embeds, mask, 0
    pad_amount = target - length
    if pad_amount == 0:
        return inputs_embeds, mask, 0
    pad = torch.zeros(
        inputs_embeds.shape[0],
        pad_amount,
        inputs_embeds.shape[2],
        dtype=inputs_embeds.dtype,
        device=inputs_embeds.device,
    )
    mask_pad = torch.zeros(
        inputs_embeds.shape[0],
        pad_amount,
        dtype=torch.long,
        device=inputs_embeds.device,
    )
    return (
        torch.cat([pad, inputs_embeds], dim=1),
        torch.cat([mask_pad, mask], dim=1),
        pad_amount,
    )


def run_softopt(
    model,
    tokenizer,
    user_message: str,
    target: str,
    config: SoftOptConfig,
) -> dict:
    """Optimize a soft prompt to elicit *target* from the model.

    Returns dict with 'input_embeds', 'losses', 'final_loss'.
    """
    device = config.device
    messages = [{"role": "user", "content": user_message + "{optim_str}"}]
    template = tokenizer.apply_chat_template(
        messages, tokenize=False, add_generation_prompt=True
    )
    if tokenizer.bos_token and template.startswith(tokenizer.bos_token):
        template = template.replace(tokenizer.bos_token, "", 1)

    before_str, after_str = template.split("{optim_str}")
    target_str = target

    before_ids = torch.tensor(
        tokenizer([before_str], padding=False)["input_ids"], device=device
    )
    after_ids = torch.tensor(
        tokenizer([after_str], add_special_tokens=False)["input_ids"], device=device
    )
    target_ids = torch.tensor(
        tokenizer([target_str], add_special_tokens=False)["input_ids"], device=device
    )

    embed = model.get_input_embeddings()
    before_embeds = embed(before_ids)
    after_embeds = embed(after_ids)
    target_embeds = embed(target_ids)

    optim_ids = tokenizer(
        config.optim_str_init,
        return_tensors="pt",
        add_special_tokens=False,
    )["input_ids"].to(device)
    optim_embeds = embed(optim_ids).detach().clone().requires_grad_()

    optimizer = torch.optim.SGD([optim_embeds], lr=config.lr)

    losses = []
    for step in range(config.num_steps):
        optimizer.zero_grad()
        inp = torch.cat(
            [
                before_embeds.detach(),
                optim_embeds,
                after_embeds.detach(),
                target_embeds.detach(),
            ],
            dim=1,
        )
        inp_bucketed, inp_mask, pad_amount = _maybe_bucket_embeds(inp)
        inp_pos = _mask_position_ids(inp_mask)
        out = model(
            inputs_embeds=inp_bucketed,
            attention_mask=inp_mask,
            position_ids=inp_pos,
        )
        logits = out.logits

        target_start = (
            pad_amount
            + before_ids.shape[1]
            + optim_ids.shape[1]
            + after_ids.shape[1]
        )
        target_len = target_ids.shape[1]
        shift_logits = logits[
            ..., target_start - 1 : target_start + target_len - 1, :
        ].contiguous()
        loss = torch.nn.functional.cross_entropy(
            shift_logits.view(-1, shift_logits.size(-1)),
            target_ids.view(-1),
        )
        loss_val = loss.item()
        losses.append(loss_val)

        if config.early_stop_loss and loss_val < config.early_stop_loss:
            break

        loss.backward()
        optimizer.step()

    input_embeds = torch.cat(
        [before_embeds.detach(), optim_embeds, after_embeds.detach()], dim=1
    )
    return {
        "input_embeds": input_embeds.detach(),
        "losses": losses,
        "final_loss": losses[-1] if losses else float("inf"),
    }


# ---------------------------------------------------------------------------
# Judge: reuse substring matching from the existing codebase
# ---------------------------------------------------------------------------

def judge_completions_substring(generations: List[str]) -> List[int]:
    """Score each generation via substring matching. 1 = jailbroken (no refusal), 0 = refused."""
    return [int(not substring_matching_judge_fn(gen)) for gen in generations]


# ---------------------------------------------------------------------------
# Full evaluation pipeline
# ---------------------------------------------------------------------------

def run_softopt_evaluation(
    model,
    tokenizer,
    benchmark_path: str,
    output_dir: str,
    softopt_config: Optional[SoftOptConfig] = None,
    max_new_tokens: int = 512,
    limit: Optional[int] = None,
    fwd_pre_hooks: Optional[List] = None,
    fwd_hooks: Optional[List] = None,
) -> Dict:
    """
    Run SoftOpt attack on the given model, judge with substring matching.

    Args:
        model: The target model (already loaded).
        tokenizer: The target model's tokenizer.
        benchmark_path: Path to HarmBench-format JSON.
        output_dir: Where to save results.
        softopt_config: Attack hyperparameters.
        max_new_tokens: Max generation length.
        limit: Max number of behaviors to test.
        fwd_pre_hooks/fwd_hooks: Optional defense hooks.  These must be passed
            for hook-based defenses so SoftOpt attacks the defended model, not
            the bare model weights.

    Returns:
        Dict with ASR score and per-instance results.
    """
    if softopt_config is None:
        softopt_config = SoftOptConfig()
    fwd_pre_hooks = fwd_pre_hooks or []
    fwd_hooks = fwd_hooks or []

    device = softopt_config.device

    # Load benchmark
    with open(benchmark_path) as f:
        data = json.load(f)["data"]
    if limit:
        data = data[:limit]

    print(f"[SoftOpt] Running attack on {len(data)} behaviors …")

    results = []
    for item in tqdm(data, desc="SoftOpt attack"):
        behavior = item["behavior"]
        target = item.get("default_target", "Sure, here's") + ":\n\n"

        # Run optimization and generation under the active defense hooks when
        # the defense is hook-based (surgical / CAST / AlphaSteer).
        with add_hooks(fwd_pre_hooks, fwd_hooks):
            opt_result = run_softopt(
                model, tokenizer, behavior, target, softopt_config
            )

            # Generate from optimized embeddings
            input_embeds = opt_result["input_embeds"].to(device)
            with torch.no_grad():
                output = model.generate(
                    inputs_embeds=input_embeds,
                    max_new_tokens=max_new_tokens,
                    do_sample=False,
                    pad_token_id=tokenizer.eos_token_id,
                )
            generation = tokenizer.decode(output[0], skip_special_tokens=True)

        results.append({
            "behavior": behavior,
            "generation": generation,
            "final_loss": opt_result["final_loss"],
        })

        # Free memory
        del opt_result, input_embeds, output
        gc.collect()
        _dev_empty_cache()

    # Judge with substring matching (no extra model needed)
    generations = [r["generation"] for r in results]
    scores = judge_completions_substring(generations)

    for r, s in zip(results, scores):
        r["score"] = s

    asr = np.mean(scores)
    avg_loss = np.mean([r["final_loss"] for r in results])

    summary = {
        "softopt_asr": float(asr),
        "num_harmful": int(sum(scores)),
        "total": len(scores),
        "avg_final_loss": float(avg_loss),
        "results": results,
    }

    print(f"[SoftOpt] ASR: {asr:.4f} ({sum(scores)}/{len(scores)} jailbroken)")
    print(f"[SoftOpt] Avg final loss: {avg_loss:.4f}")

    # Save
    os.makedirs(output_dir, exist_ok=True)
    out_path = os.path.join(output_dir, "softopt_evaluation.json")
    with open(out_path, "w") as f:
        json.dump(summary, f, indent=2, ensure_ascii=False)
    print(f"[SoftOpt] Results saved to {out_path}")

    return summary


# ---------------------------------------------------------------------------
# Standalone entry point
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="SoftOpt attack evaluation")
    parser.add_argument("--model_path", type=str, required=True)
    parser.add_argument("--benchmark", type=str, required=True,
                        help="Path to HarmBench-format JSON")
    parser.add_argument("--output_dir", type=str, default="softopt_results")
    parser.add_argument("--limit", type=int, default=None)
    parser.add_argument("--num_steps", type=int, default=500)
    parser.add_argument("--lr", type=float, default=0.001)
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()

    set_seed(args.seed)

    print(f"[SoftOpt] Loading model: {args.model_path}")
    device = _dev_get_device_str()
    model = _dev_load_model(
        AutoModelForCausalLM, args.model_path, torch_dtype=torch.bfloat16,
        trust_remote_code=False,
    )
    model.eval()
    tokenizer = AutoTokenizer.from_pretrained(args.model_path)
    tokenizer.padding_side = "left"
    if not tokenizer.pad_token:
        tokenizer.pad_token_id = tokenizer.eos_token_id

    config = SoftOptConfig(
        num_steps=args.num_steps,
        lr=args.lr,
        seed=args.seed,
        device=device,
    )

    run_softopt_evaluation(
        model=model,
        tokenizer=tokenizer,
        benchmark_path=args.benchmark,
        output_dir=args.output_dir,
        softopt_config=config,
        limit=args.limit,
    )
