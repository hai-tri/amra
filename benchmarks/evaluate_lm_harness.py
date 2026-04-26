"""
Utility benchmark evaluation via lm-evaluation-harness.

Runs GSM8k, MATH500, and MMLU on a saved model checkpoint using the
lm-evaluation-harness CLI as a subprocess.  The defended model must be
saved to disk before calling this (Stage 8 in the pipeline already does
this via model.save_pretrained()).

Each benchmark is evaluated on 100 randomly sampled examples (seeded),
selected via lm_eval's --samples flag which accepts explicit doc indices.
This ensures reproducibility across runs and fair comparison across defense
configurations.

Benchmarks:
    - GSM8k      : grade-school math word problems (8-shot), 1319 test examples
    - MATH500    : competition math, 500-problem subset (4-shot)
    - MMLU       : massive multitask language understanding (5-shot), ~14k test examples

Reference: Gao et al. 2021, "A Framework for Few-Shot Language Model Evaluation"
https://github.com/EleutherAI/lm-evaluation-harness
"""

import json
import os
import random
import subprocess
import sys
import tempfile
from typing import Dict, List, Optional, Union

_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if _ROOT not in sys.path:
    sys.path.insert(0, _ROOT)
from device_utils import get_device_str as _dev_get_device_str

_PYTHON = sys.executable

# Task name → (lm_eval task string, num_fewshot, hf_dataset_args)
# hf_dataset_args: (path, name, split) for size detection
TASKS = {
    "gsm8k":   ("gsm8k",            8, ("openai/gsm8k",      "main",    "test")),
    "math500": ("hendrycks_math500", 4, ("HuggingFaceH4/MATH-500", "default", "test")),
    "mmlu":    ("mmlu",              5, ("hails/mmlu_no_train", None,    "test")),
}

# Fallback dataset sizes if HF download fails
_FALLBACK_SIZES = {
    "gsm8k":   1319,
    "math500": 500,
    "mmlu":    14042,
}


class HookedHFLM:
    """
    Thin wrapper around lm-eval-harness's HuggingFace backend that preserves
    APRS runtime hooks during forward and generation calls.
    """

    def __new__(
        cls,
        pretrained_model,
        tokenizer,
        fwd_pre_hooks=None,
        fwd_hooks=None,
        **kwargs,
    ):
        from lm_eval.models.huggingface import HFLM
        from pipeline.utils.hook_utils import add_hooks

        class _InnerHookedHFLM(HFLM):
            def __init__(self, *args, **inner_kwargs):
                self._aprs_fwd_pre_hooks = list(fwd_pre_hooks or [])
                self._aprs_fwd_hooks = list(fwd_hooks or [])
                super().__init__(*args, **inner_kwargs)

            def _model_call(self, inps, attn_mask=None, labels=None):
                with add_hooks(self._aprs_fwd_pre_hooks, self._aprs_fwd_hooks):
                    return super()._model_call(inps, attn_mask=attn_mask, labels=labels)

            def _model_generate(self, context, max_length, stop, **generation_kwargs):
                with add_hooks(self._aprs_fwd_pre_hooks, self._aprs_fwd_hooks):
                    return super()._model_generate(
                        context=context,
                        max_length=max_length,
                        stop=stop,
                        **generation_kwargs,
                    )

        return _InnerHookedHFLM(
            pretrained=pretrained_model,
            tokenizer=tokenizer,
            trust_remote_code=True,
            **kwargs,
        )


def _get_dataset_size(task_key: str) -> int:
    """Return number of examples in the test split for a task."""
    _, _, (path, name, split) = TASKS[task_key]
    try:
        from datasets import load_dataset
        kwargs = {"split": split, "streaming": True}
        if name:
            kwargs["name"] = name
        ds = load_dataset(path, **kwargs)
        # Streaming dataset — count via info if available
        try:
            return ds.info.splits[split].num_examples
        except Exception:
            pass
    except Exception:
        pass
    return _FALLBACK_SIZES[task_key]


def _sample_indices(task_key: str, n: int, seed: int) -> List[int]:
    """
    Return n randomly sampled doc indices for a task, seeded for reproducibility.
    For math500 (exactly 500 examples), cap n at 500.
    """
    size = _get_dataset_size(task_key)
    n = min(n, size)
    rng = random.Random(seed)
    return sorted(rng.sample(range(size), n))


def run_lm_harness(
    model_path: Optional[str] = None,
    model=None,
    tokenizer=None,
    fwd_pre_hooks: Optional[list] = None,
    fwd_hooks: Optional[list] = None,
    tasks: Optional[list] = None,
    n_samples: int = 100,
    batch_size: Union[int, str] = 32,
    device: Optional[str] = None,
    output_dir: Optional[str] = None,
    seed: int = 42,
) -> Dict:
    """
    Run lm-evaluation-harness benchmarks on a saved model.

    Parameters
    ----------
    model_path : path to saved HF model directory
    model      : loaded HF model object, used for in-process evaluation
    tokenizer  : tokenizer corresponding to `model`
    tasks      : list of task keys (default: all — gsm8k, math500, mmlu)
    n_samples  : number of examples to evaluate per task (default: 100)
    batch_size : per-device batch size
    device     : "cuda", "mps", or "cpu" (auto-detected if None)
    output_dir : directory to write lm_eval JSON results
    seed       : random seed for example selection AND fewshot sampling

    Returns
    -------
    dict with per-task metrics, e.g.:
        {
          "gsm8k":   {"exact_match": 0.42, ...},
          "math500": {"exact_match": 0.18, ...},
          "mmlu":    {"acc": 0.61, ...},
        }
    """
    import torch

    if tasks is None:
        tasks = list(TASKS.keys())

    if output_dir is None:
        output_dir = tempfile.mkdtemp(prefix="lm_eval_")
    os.makedirs(output_dir, exist_ok=True)

    results = {}

    use_in_process_model = model is not None
    if device is None:
        if model is not None:
            device = str(next(model.parameters()).device)
        else:
            device = _dev_get_device_str()

    if use_in_process_model:
        if tokenizer is None:
            raise ValueError("tokenizer is required when evaluating an in-memory model")
        from lm_eval import simple_evaluate

        lm = HookedHFLM(
            pretrained_model=model,
            tokenizer=tokenizer,
            fwd_pre_hooks=fwd_pre_hooks,
            fwd_hooks=fwd_hooks,
            batch_size=batch_size,
            device=device,
        )
    elif model_path is None:
        raise ValueError("Either model_path or model must be provided")

    for task_key in tasks:
        if task_key not in TASKS:
            print(f"[lm-harness] Unknown task '{task_key}', skipping.")
            continue

        lm_task, num_fewshot, _ = TASKS[task_key]

        # Use --limit to cap examples per task. --samples requires exact subtask
        # keys (e.g. "mmlu_anatomy") and silently does nothing when passed the
        # group name "mmlu", so --limit is the only reliable approach.
        print(f"[lm-harness] {task_key}: limiting to {n_samples} examples (seed={seed})")

        output_path = os.path.join(output_dir, f"{task_key}_results.json")

        print(f"[lm-harness] Running {task_key} ({lm_task}, {num_fewshot}-shot, "
              f"n={n_samples}) …")

        if use_in_process_model:
            raw = simple_evaluate(
                model=lm,
                tasks=[lm_task],
                num_fewshot=num_fewshot,
                batch_size=batch_size,
                device=device,
                limit=n_samples,
                log_samples=False,
                apply_chat_template=True,
                random_seed=seed,
                numpy_random_seed=seed,
                torch_random_seed=seed,
                fewshot_random_seed=seed,
                bootstrap_iters=0,
            )
            with open(output_path, "w") as f:
                json.dump(raw, f, indent=2, default=str)
            task_results = raw.get("results", {}).get(lm_task, {})
            results[task_key] = _extract_metrics(task_key, task_results)
            results[task_key]["n_samples"] = n_samples
            results[task_key]["seed"] = seed
            print(f"[lm-harness] {task_key}: {results[task_key]}")
        else:
            cmd = [
                _PYTHON, "-m", "lm_eval",
                "--model", "hf",
                "--model_args", f"pretrained={model_path},dtype=bfloat16,trust_remote_code=True",
                "--tasks", lm_task,
                "--num_fewshot", str(num_fewshot),
                "--batch_size", str(batch_size),
                "--device", device,
                "--output_path", output_path,
                "--seed", str(seed),
                "--limit", str(n_samples),
                "--trust_remote_code",
                "--apply_chat_template",
            ]

            proc = subprocess.run(cmd, capture_output=False, text=True)

            if proc.returncode != 0:
                print(f"[lm-harness] WARNING: {task_key} exited with code {proc.returncode}")
                results[task_key] = {"error": f"exit code {proc.returncode}"}
                continue

            result_file = output_path
            if not os.path.exists(result_file):
                candidates = []
                for root, _, files in os.walk(output_dir):
                    for fname in files:
                        if fname.endswith(".json") and "results" in fname:
                            candidates.append(os.path.join(root, fname))
                if candidates:
                    result_file = max(candidates, key=os.path.getmtime)

            if os.path.exists(result_file):
                with open(result_file) as f:
                    raw = json.load(f)
                task_results = raw.get("results", {}).get(lm_task, {})
                results[task_key] = _extract_metrics(task_key, task_results)
                results[task_key]["n_samples"] = n_samples
                results[task_key]["seed"] = seed
                print(f"[lm-harness] {task_key}: {results[task_key]}")
            else:
                print(f"[lm-harness] WARNING: result file not found for {task_key}")
                results[task_key] = {}

    # Save combined results
    combined_path = os.path.join(output_dir, "utility_benchmarks.json")
    with open(combined_path, "w") as f:
        json.dump(results, f, indent=4)
    print(f"[lm-harness] Combined results saved to {combined_path}")

    return results


def _extract_metrics(task_key: str, raw: dict) -> dict:
    """Pull the primary accuracy metric from lm_eval's result dict."""
    if task_key == "gsm8k":
        return {
            "exact_match": raw.get("exact_match,strict-match",
                                   raw.get("exact_match", None)),
            "exact_match_flexible": raw.get("exact_match,flexible-extract", None),
        }
    elif task_key == "math500":
        return {
            "exact_match": raw.get("exact_match,get-answer",
                                   raw.get("exact_match,none",
                                   raw.get("exact_match", None))),
        }
    elif task_key == "mmlu":
        return {
            "acc": raw.get("acc,none", raw.get("acc", None)),
            "acc_norm": raw.get("acc_norm,none", raw.get("acc_norm", None)),
        }
    return raw
