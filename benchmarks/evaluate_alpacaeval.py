"""
AlpacaEval generation-quality evaluation.

Measures instruction-following quality on open-ended prompts to check that the
defense preserves helpfulness beyond multiple-choice / math benchmarks.
Complements lm-evaluation-harness (GSM8k/MATH500/MMLU), which only covers
reasoning/knowledge tasks with short deterministic answers.

Reference: AlpacaEval — Dubois et al. 2024
https://github.com/tatsu-lab/alpaca_eval
https://huggingface.co/datasets/tatsu-lab/alpaca_eval

Two-phase design:
    1. Generation (always runs) — produce (possibly defended) model completions
       for the 805 AlpacaEval v2 prompts and save them in the official
       ``[{"instruction", "output", "generator"}]`` JSON format.
    2. Judging (optional) — if the ``alpaca_eval`` package is installed and a
       judge credential is available (e.g. OPENAI_API_KEY for the default
       GPT-4 annotator), invoke the judge to compute length-controlled win
       rate against the baseline reference output.  If unavailable, we save
       completions only and return ``win_rate=None`` so the pipeline keeps
       running — AlpacaEval-style judging is expensive and not always
       feasible inline.
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

from pipeline.utils.hook_utils import add_hooks


_DEFAULT_GENERATOR_NAME = "aprs_defended"
_DEFAULT_DATASET = "tatsu-lab/alpaca_eval"
_DEFAULT_SPLIT = "eval"


def load_alpacaeval_prompts(
    dataset: str = _DEFAULT_DATASET,
    split: str = _DEFAULT_SPLIT,
    n: Optional[int] = None,
    seed: int = 42,
) -> List[Dict]:
    """
    Load the AlpacaEval evaluation prompts from the Hugging Face hub.

    Returns a list of ``{"instruction": str, "dataset": str}`` dicts.  When
    ``n`` is provided we deterministically sample ``n`` prompts using
    ``seed`` so subsets are comparable across runs.
    """
    from datasets import load_dataset

    ds = load_dataset(dataset, "alpaca_eval", split=split, trust_remote_code=True)
    rows = [{"instruction": r["instruction"],
             "dataset": r.get("dataset", "")} for r in ds]

    if n is not None and n < len(rows):
        import random
        rng = random.Random(seed)
        rows = rng.sample(rows, n)

    return rows


def _decode(tokenizer, token_ids: torch.Tensor) -> str:
    """Decode with the explicit tokenizer (no model-attribute fallback)."""
    if tokenizer is None:
        raise ValueError("tokenizer is required for response decoding")
    return tokenizer.decode(token_ids, skip_special_tokens=True)


def generate_alpacaeval_completions(
    model,
    tokenizer,
    tokenize_fn,
    prompts: List[Dict],
    fwd_pre_hooks: list = [],
    fwd_hooks: list = [],
    max_new_tokens: int = 512,
    batch_size: int = 32,
    generator_name: str = _DEFAULT_GENERATOR_NAME,
) -> List[Dict]:
    """
    Run (optionally defended) greedy generation on AlpacaEval prompts.

    Returns a list of dicts in the format expected by the ``alpaca_eval``
    judge: ``{"instruction", "output", "generator", "dataset"}``.
    """
    device = next(model.parameters()).device
    instructions = [p["instruction"] for p in prompts]

    outputs: List[str] = []
    model.eval()

    with add_hooks(fwd_pre_hooks, fwd_hooks):
        for i in range(0, len(instructions), batch_size):
            batch = instructions[i : i + batch_size]
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

            prompt_len = input_ids.shape[1]
            for out in output_ids:
                new_tokens = out[prompt_len:]
                outputs.append(_decode(tokenizer, new_tokens).strip())

            print(f"[AlpacaEval] {min(i + batch_size, len(instructions))}"
                  f"/{len(instructions)}")

    records = []
    for meta, out in zip(prompts, outputs):
        records.append({
            "instruction": meta["instruction"],
            "output": out,
            "generator": generator_name,
            "dataset": meta.get("dataset", ""),
        })
    return records


def _run_alpacaeval_judge(
    completions_path: str,
    output_dir: str,
    annotators_config: str = "alpaca_eval_gpt4_turbo_fn",
) -> Optional[Dict]:
    """
    Invoke the ``alpaca_eval`` package to compute the length-controlled win
    rate for ``completions_path`` against the default reference outputs.

    Returns a dict with ``win_rate``, ``length_controlled_win_rate``,
    ``n_total``, ``annotator``.  Returns None if the package or credentials
    are unavailable; the caller should treat that as "judging skipped".
    """
    try:
        from alpaca_eval import evaluate as alpaca_evaluate  # type: ignore
    except ImportError:
        print("[AlpacaEval] `alpaca_eval` package not installed — "
              "skipping judging. Install via `pip install alpaca-eval`.")
        return None

    # Default annotator is GPT-4 via OpenAI; other annotator configs exist for
    # Claude, local models, etc.  If no key is present for the default, the
    # call will fail — surface that as "skipped" rather than crashing the
    # whole pipeline.
    if annotators_config.startswith("alpaca_eval_gpt4") and not os.environ.get("OPENAI_API_KEY"):
        print("[AlpacaEval] OPENAI_API_KEY not set — skipping GPT-4 judging. "
              "Completions are saved; you can judge later with "
              f"`alpaca_eval --model_outputs {completions_path}`.")
        return None

    os.makedirs(output_dir, exist_ok=True)
    try:
        df_leaderboard, _ = alpaca_evaluate(
            model_outputs=completions_path,
            annotators_config=annotators_config,
            output_path=output_dir,
            is_return_instead_of_print=True,
        )
    except Exception as e:
        print(f"[AlpacaEval] Judging failed: {e}")
        return None

    # df_leaderboard is a pandas DataFrame indexed by generator name.  Pick
    # the row corresponding to our entry.
    try:
        row = df_leaderboard.iloc[0]
        return {
            "win_rate": float(row.get("win_rate", float("nan"))),
            "length_controlled_win_rate": float(
                row.get("length_controlled_winrate", float("nan"))
            ),
            "n_total": int(row.get("n_total", 0)),
            "annotator": annotators_config,
        }
    except Exception as e:
        print(f"[AlpacaEval] Could not parse judge output: {e}")
        return None


def evaluate_alpacaeval(
    model,
    tokenizer,
    tokenize_fn,
    fwd_pre_hooks: list = [],
    fwd_hooks: list = [],
    n_samples: Optional[int] = None,
    max_new_tokens: int = 512,
    batch_size: int = 32,
    seed: int = 42,
    run_judge: bool = True,
    annotators_config: str = "alpaca_eval_gpt4_turbo_fn",
    generator_name: str = _DEFAULT_GENERATOR_NAME,
    artifact_dir: Optional[str] = None,
) -> Dict:
    """
    Full AlpacaEval pass: generate completions, save to disk, optionally judge.

    Parameters
    ----------
    n_samples   : subsample the 805-prompt eval set. None = all.
    max_new_tokens : max tokens per generated response.
    run_judge   : if True, attempt to invoke the ``alpaca_eval`` judge when
                  available.  False skips judging and returns generation-only.

    Returns
    -------
    dict with:
        ``n_samples``        — number of prompts evaluated
        ``completions_path`` — path to the saved completions JSON
        ``win_rate``         — or None if judging skipped
        ``length_controlled_win_rate`` — or None
        ``annotator``        — name of the judge config, or None
    """
    print(f"[AlpacaEval] Loading prompts "
          f"(n={n_samples if n_samples is not None else 'all'}) …")
    prompts = load_alpacaeval_prompts(n=n_samples, seed=seed)
    print(f"[AlpacaEval] {len(prompts)} prompts loaded.")

    records = generate_alpacaeval_completions(
        model=model,
        tokenizer=tokenizer,
        tokenize_fn=tokenize_fn,
        prompts=prompts,
        fwd_pre_hooks=fwd_pre_hooks,
        fwd_hooks=fwd_hooks,
        max_new_tokens=max_new_tokens,
        batch_size=batch_size,
        generator_name=generator_name,
    )

    completions_path = None
    if artifact_dir:
        os.makedirs(artifact_dir, exist_ok=True)
        completions_path = os.path.join(artifact_dir, "alpacaeval_completions.json")
        with open(completions_path, "w") as f:
            json.dump(records, f, indent=2, ensure_ascii=False)
        print(f"[AlpacaEval] Completions saved to {completions_path}")

    result: Dict = {
        "n_samples": len(records),
        "completions_path": completions_path,
        "win_rate": None,
        "length_controlled_win_rate": None,
        "annotator": None,
        "seed": seed,
    }

    if run_judge and completions_path is not None:
        judge_dir = os.path.join(artifact_dir, "alpacaeval_judge")
        judge_result = _run_alpacaeval_judge(
            completions_path=completions_path,
            output_dir=judge_dir,
            annotators_config=annotators_config,
        )
        if judge_result is not None:
            result.update(judge_result)
            print(f"[AlpacaEval] win_rate={result['win_rate']:.4f} "
                  f"lc_win_rate={result['length_controlled_win_rate']:.4f} "
                  f"(annotator={result['annotator']})")

    if artifact_dir:
        summary_path = os.path.join(artifact_dir, "alpacaeval_result.json")
        with open(summary_path, "w") as f:
            json.dump(result, f, indent=4)

    return result
