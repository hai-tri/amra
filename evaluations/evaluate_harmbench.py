"""
HarmBench ASR evaluation for the obfuscation defense pipeline.

Samples 100 behaviors from HarmBench, generates completions with the
(possibly defended) model, then scores them with the HarmBench
Llama-2-13B classifier (cais/HarmBench-Llama-2-13b-cls).

Reference: Mazeika et al. 2024, "HarmBench: A Standardized Evaluation
Framework for Automated Red Teaming and Robust Refusal"
https://arxiv.org/abs/2402.04249
"""

import json
import os
import sys
import torch
from typing import Dict, List, Optional

_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
_REFUSAL_DIR = os.path.join(_ROOT, "refusal_direction")
if _REFUSAL_DIR not in sys.path:
    sys.path.insert(0, _REFUSAL_DIR)
if _ROOT not in sys.path:
    sys.path.insert(0, _ROOT)

from pipeline.submodules.evaluate_jailbreak import harmbench_judge_fn, load_harmbench_behaviors
from device_utils import empty_cache as _dev_empty_cache

_DEFAULT_CSV = os.path.join(_ROOT, "data", "harmbench_behaviors_text_test.csv")


def generate_responses_for_prompts(
    model,
    tokenizer,
    tokenize_fn,
    prompts: List[str],
    fwd_pre_hooks: Optional[list] = None,
    fwd_hooks: Optional[list] = None,
    max_new_tokens: int = 256,
    batch_size: int = 4,
) -> List[str]:
    """Generate deterministic responses for prompts under optional hooks.

    ``tokenizer`` is required — earlier versions tried to read it off the
    model (``model.tokenizer`` / ``model._tokenizer``), which silently
    failed on plain HF models and caused responses to decode as the string
    representation of their token-id list.
    """
    from pipeline.utils.hook_utils import add_hooks

    if fwd_pre_hooks is None:
        fwd_pre_hooks = []
    if fwd_hooks is None:
        fwd_hooks = []

    device = next(model.parameters()).device
    responses: List[str] = []
    model.eval()

    with add_hooks(fwd_pre_hooks, fwd_hooks):
        for i in range(0, len(prompts), batch_size):
            batch = prompts[i : i + batch_size]
            inputs = tokenize_fn(instructions=batch)
            input_ids = inputs.input_ids.to(device)
            attention_mask = inputs.attention_mask.to(device)

            with torch.no_grad():
                output_ids = model.generate(
                    input_ids=input_ids,
                    attention_mask=attention_mask,
                    max_new_tokens=max_new_tokens,
                    do_sample=False,
                    pad_token_id=input_ids[0, -1].item(),
                )

            for out in output_ids:
                prompt_len = input_ids.shape[1]
                new_tokens = out[prompt_len:]
                responses.append(_decode(tokenizer, new_tokens))

            print(f"[HarmBench] Generated {min(i + batch_size, len(prompts))}/{len(prompts)}")

    return responses


def score_harmbench_responses(
    prompts: List[str],
    responses: List[str],
    classifier_model_id: str = "cais/HarmBench-Llama-2-13b-cls",
    artifact_path: Optional[str] = None,
    metadata: Optional[List[Dict]] = None,
) -> Dict:
    """Score existing prompt/response pairs with the HarmBench classifier."""
    if len(prompts) != len(responses):
        raise ValueError(
            f"Expected prompts and responses to have the same length, got "
            f"{len(prompts)} and {len(responses)}"
        )

    if not prompts:
        result = {
            "asr": 0.0,
            "n_harmful": 0,
            "n_total": 0,
            "completions": [],
        }
        if artifact_path:
            os.makedirs(os.path.dirname(artifact_path), exist_ok=True)
            with open(artifact_path, "w") as f:
                json.dump(result, f, indent=4)
        return result

    print("[HarmBench] Running HarmBench classifier …")
    classifications = harmbench_judge_fn(
        prompts=prompts,
        responses=responses,
        classifier_model_id=classifier_model_id,
    )

    completions = []
    for idx, (prompt, response, is_harmful) in enumerate(zip(prompts, responses, classifications)):
        entry = {
            "behavior": prompt,
            "response": response,
            "is_harmful": is_harmful,
        }
        if metadata is not None and idx < len(metadata) and metadata[idx] is not None:
            entry.update(metadata[idx])
        completions.append(entry)

    result = {
        "asr": sum(classifications) / len(classifications),
        "n_harmful": sum(classifications),
        "n_total": len(classifications),
        "completions": completions,
    }

    if artifact_path:
        os.makedirs(os.path.dirname(artifact_path), exist_ok=True)
        with open(artifact_path, "w") as f:
            json.dump(result, f, indent=4)
        print(f"[HarmBench] Results saved to {artifact_path}")

    return result


def evaluate_harmbench_asr(
    model,
    tokenizer,
    tokenize_fn,
    fwd_pre_hooks: list = [],
    fwd_hooks: list = [],
    behaviors_csv: str = _DEFAULT_CSV,
    n_behaviors: int = 100,
    seed: int = 42,
    max_new_tokens: int = 256,
    batch_size: int = 4,
    artifact_dir: Optional[str] = None,
    classifier_model_id: str = "cais/HarmBench-Llama-2-13b-cls",
) -> Dict:
    """
    Full HarmBench evaluation: generation + classification.

    Parameters
    ----------
    model : the (possibly defended) causal LM
    tokenize_fn : callable(instructions=[...]) → BatchEncoding
    fwd_pre_hooks / fwd_hooks : optional inference-time hooks (e.g. ablation)
    behaviors_csv : path to HarmBench behaviors CSV
    n_behaviors : number of behaviors to sample (default 100)
    seed : random seed for behavior sampling
    max_new_tokens : generation budget per completion
    batch_size : generation batch size
    artifact_dir : if set, save completions + evaluations here

    Returns
    -------
    dict with:
        ``asr``             — fraction of behaviors where model complied
        ``n_harmful``       — number of harmful completions
        ``n_total``         — total behaviors evaluated
        ``completions``     — list of {behavior, behavior_id, response, is_harmful}
        ``per_category_asr``— ASR broken down by SemanticCategory
    """
    # ----------------------------------------------------------------
    # Step 1: Load behaviors
    # ----------------------------------------------------------------
    print(f"[HarmBench] Loading {n_behaviors} behaviors from {behaviors_csv}")
    behaviors = load_harmbench_behaviors(behaviors_csv, n=n_behaviors, seed=seed)
    prompts = [b["behavior"] for b in behaviors]

    # ----------------------------------------------------------------
    # Step 2: Generate completions
    # ----------------------------------------------------------------
    print(f"[HarmBench] Generating completions ({len(prompts)} prompts) …")
    responses = generate_responses_for_prompts(
        model=model,
        tokenizer=tokenizer,
        tokenize_fn=tokenize_fn,
        prompts=prompts,
        fwd_pre_hooks=fwd_pre_hooks,
        fwd_hooks=fwd_hooks,
        max_new_tokens=max_new_tokens,
        batch_size=batch_size,
    )

    # ----------------------------------------------------------------
    # Step 3: Offload target model to free memory for classifier
    # ----------------------------------------------------------------
    print("[HarmBench] Offloading target model before running classifier …")
    original_device = next(model.parameters()).device
    model.to("cpu")
    _dev_empty_cache()

    # ----------------------------------------------------------------
    # Step 4: Classify
    # ----------------------------------------------------------------
    try:
        artifact_path = None
        if artifact_dir:
            artifact_path = os.path.join(artifact_dir, "harmbench_evaluation.json")
        result = score_harmbench_responses(
            prompts=prompts,
            responses=responses,
            classifier_model_id=classifier_model_id,
            artifact_path=artifact_path,
            metadata=[
                {
                    "behavior_id": b["behavior_id"],
                    "semantic_category": b["semantic_category"],
                }
                for b in behaviors
            ],
        )
    finally:
        model.to(original_device)

    per_category = {}
    for c in result["completions"]:
        cat = c.get("semantic_category") or "unknown"
        per_category.setdefault(cat, []).append(c["is_harmful"])
    result["per_category_asr"] = {cat: sum(vals) / len(vals) for cat, vals in per_category.items()}

    if artifact_dir:
        artifact_path = os.path.join(artifact_dir, "harmbench_evaluation.json")
        with open(artifact_path, "w") as f:
            json.dump(result, f, indent=4)

    print(f"[HarmBench] ASR: {result['asr']:.4f} ({result['n_harmful']}/{result['n_total']})")
    return result


def _decode(tokenizer, token_ids: torch.Tensor) -> str:
    """Decode a token-id tensor to text using the explicit tokenizer.

    The tokenizer is required.  Earlier versions tried to read it off the
    model object (``model.tokenizer`` / ``model._tokenizer``); plain HF
    models have neither, so that silently fell through to the token-id
    ``str(...)`` branch and corrupted downstream classifier scoring.
    """
    if tokenizer is None:
        raise ValueError("tokenizer is required for response decoding")
    return tokenizer.decode(token_ids, skip_special_tokens=True)
