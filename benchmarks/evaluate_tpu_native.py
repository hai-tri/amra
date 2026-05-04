"""
TPU-native utility benchmarks for the APRS pipeline.

This module avoids lm-evaluation-harness subprocesses on TPU. It keeps all
model calls in-process, preserves APRS hooks, and relies on bucket-padded
tokenization so XLA sees a small fixed set of sequence shapes.
"""

from __future__ import annotations

import json
import math
import os
import random
import re
import sys
from typing import Dict, Iterable, List, Optional, Sequence

import torch

_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
_REFUSAL_DIR = os.path.join(_ROOT, "refusal_direction")
if _ROOT not in sys.path:
    sys.path.insert(0, _ROOT)
if _REFUSAL_DIR not in sys.path:
    sys.path.insert(0, _REFUSAL_DIR)

from pipeline.submodules.evaluate_loss import (  # noqa: E402
    batch_iterator_alpaca,
    batch_iterator_pile,
    compute_loss_over_dataset,
)
from pipeline.utils.hook_utils import add_hooks  # noqa: E402
from scripts.tpu.tpu_utils import (  # noqa: E402
    DEFAULT_BUCKETS,
    bucket_pad_batch_encoding,
)


_CHOICE_LABELS = ("A", "B", "C", "D")


def _load_dataset_rows(path: str, *args, split: str, n: Optional[int], seed: int):
    from datasets import load_dataset

    ds = load_dataset(path, *args, split=split, trust_remote_code=True)
    rows = list(ds)
    if n is not None and n < len(rows):
        rng = random.Random(seed)
        rows = rng.sample(rows, n)
    return rows


def _extract_mmlu(row: dict) -> Optional[dict]:
    question = row.get("question") or row.get("input") or row.get("prompt")
    choices = row.get("choices") or row.get("options")
    if choices is None:
        choices = [row.get(k) for k in _CHOICE_LABELS if row.get(k) is not None]
    answer = row.get("answer")
    if isinstance(answer, int):
        answer = _CHOICE_LABELS[answer]
    if isinstance(answer, str):
        answer = answer.strip()
        if answer and answer[0].isdigit():
            idx = int(answer[0])
            if 0 <= idx < len(_CHOICE_LABELS):
                answer = _CHOICE_LABELS[idx]
        answer = answer[0].upper() if answer else None
    if not question or not choices or answer not in _CHOICE_LABELS[:len(choices)]:
        return None
    return {"question": question, "choices": list(choices), "answer": answer}


def _format_mmlu_prompt(item: dict) -> str:
    lines = [item["question"].strip()]
    for label, choice in zip(_CHOICE_LABELS, item["choices"]):
        lines.append(f"{label}. {str(choice).strip()}")
    lines.append("Answer:")
    return "\n".join(lines)


def evaluate_mmlu_native(
    model,
    tokenizer,
    *,
    n_samples: int = 500,
    batch_size: int = 32,
    seed: int = 42,
    fwd_pre_hooks: Optional[list] = None,
    fwd_hooks: Optional[list] = None,
    buckets: Sequence[int] = DEFAULT_BUCKETS,
) -> Dict:
    """Evaluate MMLU by scoring the next-token probability of A/B/C/D."""
    rows = _load_dataset_rows(
        "hails/mmlu_no_train", split="test", n=n_samples, seed=seed
    )
    examples = [x for x in (_extract_mmlu(r) for r in rows) if x is not None]
    if not examples:
        raise RuntimeError("Could not load parseable MMLU rows")

    device = next(model.parameters()).device
    choice_token_ids = []
    for label in _CHOICE_LABELS:
        ids = tokenizer.encode(" " + label, add_special_tokens=False)
        if not ids:
            ids = tokenizer.encode(label, add_special_tokens=False)
        choice_token_ids.append(ids[0])
    choice_token_ids = torch.tensor(choice_token_ids, device=device)

    correct = 0
    total = 0
    model.eval()
    with add_hooks(fwd_pre_hooks or [], fwd_hooks or []):
        for i in range(0, len(examples), batch_size):
            batch = examples[i:i + batch_size]
            prompts = [_format_mmlu_prompt(x) for x in batch]
            enc = tokenizer(prompts, padding=True, truncation=False, return_tensors="pt")
            enc = bucket_pad_batch_encoding(enc, tokenizer, buckets=buckets)
            enc = enc.to(device)
            with torch.no_grad():
                out = model(**enc)
            # bucket_pad_batch_encoding always left-pads to the bucket. If the
            # tokenizer right-pads inside the batch, the layout becomes
            # [L-pad … real … R-pad] and the last real token sits at
            # left_pad_count + real_count - 1.
            mask = enc["attention_mask"]
            if getattr(tokenizer, "padding_side", "right") == "left":
                last = torch.full(
                    (len(batch),),
                    mask.shape[1] - 1,
                    device=device,
                    dtype=torch.long,
                )
            else:
                left_pad = (mask.cumsum(dim=1) == 0).sum(dim=1)
                last = left_pad + mask.sum(dim=1) - 1
            logits = out.logits[torch.arange(len(batch), device=device), last]
            scores = logits.index_select(dim=-1, index=choice_token_ids[:4])
            pred = scores.argmax(dim=-1).cpu().tolist()
            gold = [_CHOICE_LABELS.index(x["answer"]) for x in batch]
            correct += sum(int(p == g) for p, g in zip(pred, gold))
            total += len(batch)

    return {"acc": correct / total if total else 0.0, "n_samples": total, "seed": seed}


def _extract_answer_text(text: str) -> str:
    boxed = re.findall(r"\\boxed\{([^{}]+)\}", text)
    if boxed:
        return boxed[-1]
    matches = re.findall(r"(-?\d+(?:\.\d+)?(?:/\d+)?)", text)
    return matches[-1] if matches else text.strip()


def _normalize_answer(text: str) -> str:
    text = _extract_answer_text(text)
    text = text.lower().strip()
    text = text.replace(",", "")
    text = re.sub(r"\\text\{([^{}]+)\}", r"\1", text)
    text = re.sub(r"\s+", "", text)
    return text.strip(".")


def evaluate_math500_native(
    model,
    tokenizer,
    *,
    n_samples: int = 500,
    batch_size: int = 8,
    seed: int = 42,
    max_new_tokens: int = 256,
    fwd_pre_hooks: Optional[list] = None,
    fwd_hooks: Optional[list] = None,
    buckets: Sequence[int] = DEFAULT_BUCKETS,
    artifact_path: Optional[str] = None,
) -> Dict:
    """Evaluate MATH500 with deterministic generation and exact answer match."""
    rows = _load_dataset_rows(
        "HuggingFaceH4/MATH-500", split="test", n=n_samples, seed=seed
    )
    device = next(model.parameters()).device
    records = []
    correct = 0

    prompts = [
        "Solve the following problem. Put the final answer in \\boxed{}.\n\n"
        + (r.get("problem") or r.get("question") or "")
        for r in rows
    ]
    golds = [
        r.get("answer") or _extract_answer_text(r.get("solution", ""))
        for r in rows
    ]

    model.eval()
    with add_hooks(fwd_pre_hooks or [], fwd_hooks or []):
        for i in range(0, len(prompts), batch_size):
            batch = prompts[i:i + batch_size]
            enc = tokenizer(batch, padding=True, truncation=False, return_tensors="pt")
            enc = bucket_pad_batch_encoding(enc, tokenizer, buckets=buckets)
            input_len = enc["input_ids"].shape[1]
            enc = enc.to(device)
            with torch.no_grad():
                out = model.generate(
                    input_ids=enc["input_ids"],
                    attention_mask=enc["attention_mask"],
                    max_new_tokens=max_new_tokens,
                    do_sample=False,
                    pad_token_id=tokenizer.pad_token_id or tokenizer.eos_token_id,
                )
            for j, seq in enumerate(out):
                pred_text = tokenizer.decode(seq[input_len:], skip_special_tokens=True)
                pred = _normalize_answer(pred_text)
                gold = _normalize_answer(str(golds[i + j]))
                ok = pred == gold
                correct += int(ok)
                records.append({
                    "problem": rows[i + j].get("problem") or rows[i + j].get("question"),
                    "prediction": pred_text,
                    "prediction_normalized": pred,
                    "gold_normalized": gold,
                    "exact_match": int(ok),
                })

    result = {
        "exact_match": correct / len(records) if records else 0.0,
        "n_samples": len(records),
        "seed": seed,
    }
    if artifact_path:
        os.makedirs(os.path.dirname(artifact_path), exist_ok=True)
        with open(artifact_path, "w") as f:
            json.dump({"summary": result, "records": records}, f, indent=2)
    return result


def evaluate_gsm8k_native(
    model,
    tokenizer,
    *,
    n_samples: int = 100,
    batch_size: int = 8,
    seed: int = 42,
    max_new_tokens: int = 256,
    fwd_pre_hooks: Optional[list] = None,
    fwd_hooks: Optional[list] = None,
    buckets: Sequence[int] = DEFAULT_BUCKETS,
) -> Dict:
    """Small native GSM8k pass for parity with the existing utility suite."""
    rows = _load_dataset_rows("openai/gsm8k", "main", split="test", n=n_samples, seed=seed)
    device = next(model.parameters()).device
    prompts = [
        "Solve the following grade-school math problem. End with the final number.\n\n"
        + r["question"]
        for r in rows
    ]
    golds = [r["answer"].split("####")[-1].strip() for r in rows]
    correct = 0
    total = 0
    model.eval()
    with add_hooks(fwd_pre_hooks or [], fwd_hooks or []):
        for i in range(0, len(prompts), batch_size):
            enc = tokenizer(prompts[i:i + batch_size], padding=True, return_tensors="pt")
            enc = bucket_pad_batch_encoding(enc, tokenizer, buckets=buckets)
            input_len = enc["input_ids"].shape[1]
            enc = enc.to(device)
            with torch.no_grad():
                out = model.generate(
                    input_ids=enc["input_ids"],
                    attention_mask=enc["attention_mask"],
                    max_new_tokens=max_new_tokens,
                    do_sample=False,
                    pad_token_id=tokenizer.pad_token_id or tokenizer.eos_token_id,
                )
            for j, seq in enumerate(out):
                pred_text = tokenizer.decode(seq[input_len:], skip_special_tokens=True)
                correct += int(_normalize_answer(pred_text) == _normalize_answer(golds[i + j]))
                total += 1
    return {"exact_match": correct / total if total else 0.0, "n_samples": total, "seed": seed}


def _bucket_loss_iterator(iterator, tokenizer, buckets: Sequence[int]):
    for inputs, loss_mask in iterator:
        yield bucket_pad_batch_encoding(
            inputs, tokenizer, buckets=buckets, loss_mask=loss_mask
        )


def evaluate_bpb_native(
    model_base,
    *,
    labels: Iterable[str] = ("pile", "alpaca"),
    n_batches: int = 256,
    batch_size: int = 4,
    max_seq_length: int = 256,
    fwd_pre_hooks: Optional[list] = None,
    fwd_hooks: Optional[list] = None,
    buckets: Sequence[int] = DEFAULT_BUCKETS,
) -> Dict:
    """Evaluate Pile/Alpaca BPB with bucket-padded forward batches."""
    result = {}
    for label in labels:
        if label == "pile":
            iterator = batch_iterator_pile(
                model_base.tokenizer, batch_size=batch_size, max_length=max_seq_length
            )
        elif label == "alpaca":
            iterator = batch_iterator_alpaca(
                model_base.tokenize_instructions_fn,
                batch_size=batch_size,
                eoi_toks=torch.tensor(model_base.eoi_toks),
            )
        else:
            raise ValueError(f"Unsupported BPB label: {label}")
        ce_loss, ppl, n_tokens, n_bytes, bpb = compute_loss_over_dataset(
            model_base.model,
            model_base.tokenizer,
            _bucket_loss_iterator(iterator, model_base.tokenizer, buckets),
            n_batches=n_batches,
            fwd_pre_hooks=fwd_pre_hooks or [],
            fwd_hooks=fwd_hooks or [],
        )
        result[label] = {
            "ce_loss": ce_loss.item(),
            "perplexity": ppl.item(),
            "bpb": bpb,
            "n_tokens": n_tokens.item(),
            "n_bytes": n_bytes,
        }
        print(f"[tpu-native] {label}: BPB={bpb:.4f} PPL={ppl.item():.4f}")
    return result


def run_tpu_native_utility(
    model_base,
    *,
    tasks: Sequence[str],
    n_samples: int,
    batch_size: int,
    seed: int,
    output_dir: str,
    fwd_pre_hooks: Optional[list] = None,
    fwd_hooks: Optional[list] = None,
    buckets: Sequence[int] = DEFAULT_BUCKETS,
) -> Dict:
    """Run requested utility tasks and save a combined JSON artifact."""
    os.makedirs(output_dir, exist_ok=True)
    results: Dict[str, Dict] = {}
    for task in tasks:
        task = task.strip().lower()
        if not task:
            continue
        if task == "mmlu":
            results["mmlu"] = evaluate_mmlu_native(
                model_base.model,
                model_base.tokenizer,
                n_samples=n_samples,
                batch_size=batch_size,
                seed=seed,
                fwd_pre_hooks=fwd_pre_hooks,
                fwd_hooks=fwd_hooks,
                buckets=buckets,
            )
        elif task == "math500":
            results["math500"] = evaluate_math500_native(
                model_base.model,
                model_base.tokenizer,
                n_samples=min(n_samples, 500),
                batch_size=min(int(batch_size), 8),
                seed=seed,
                fwd_pre_hooks=fwd_pre_hooks,
                fwd_hooks=fwd_hooks,
                buckets=buckets,
                artifact_path=os.path.join(output_dir, "math500_records.json"),
            )
        elif task == "gsm8k":
            results["gsm8k"] = evaluate_gsm8k_native(
                model_base.model,
                model_base.tokenizer,
                n_samples=min(n_samples, 100),
                batch_size=min(int(batch_size), 8),
                seed=seed,
                fwd_pre_hooks=fwd_pre_hooks,
                fwd_hooks=fwd_hooks,
                buckets=buckets,
            )
        else:
            print(f"[tpu-native] Unknown task '{task}', skipping")

    path = os.path.join(output_dir, "utility_benchmarks.json")
    with open(path, "w") as f:
        json.dump(results, f, indent=2)
    print(f"[tpu-native] Utility results saved to {path}")
    return results
