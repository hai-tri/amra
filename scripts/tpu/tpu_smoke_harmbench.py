"""
End-to-end TPU smoke: undefended Llama-3-8B + tpu_utils.patch_model_for_xla
+ a tiny HarmBench run (4 prompts, 32 new tokens).

Purpose: confirm that
  1. The generate-patch produces sensible output on XLA.
  2. Repeat calls with the same bucket do NOT re-compile.
  3. Timings are tolerable before we kick off the full sweep.

Exits 0 on success and prints per-prompt wall time so we can see compile vs
steady-state latency.
"""

import os
import sys
import time

REPO = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.insert(0, REPO)
sys.path.insert(0, os.path.join(REPO, "refusal_direction"))

import torch
import torch_xla.core.xla_model as xm

from pipeline.model_utils.model_factory import construct_model_base
from scripts.tpu.tpu_utils import patch_model_for_xla


PROMPTS = [
    "Explain quantum tunneling briefly.",
    "Write a Python function that reverses a linked list.",
    "Summarize the plot of Hamlet in three sentences.",
    "What are three common causes of memory leaks in C++?",
]


def main():
    print("Loading meta-llama/Meta-Llama-3-8B-Instruct …")
    t0 = time.time()
    mb = construct_model_base("meta-llama/Meta-Llama-3-8B-Instruct")
    print(f"loaded in {time.time() - t0:.1f}s")

    patch_model_for_xla(mb.model, mb.tokenizer, buckets=(512, 1024, 2048))

    dev = next(mb.model.parameters()).device
    tok = mb.tokenizer

    for i, prompt in enumerate(PROMPTS):
        messages = [{"role": "user", "content": prompt}]
        ids = tok.apply_chat_template(messages, return_tensors="pt",
                                      add_generation_prompt=True).to(dev)
        t0 = time.time()
        with torch.no_grad():
            out = mb.model.generate(
                input_ids=ids,
                max_new_tokens=32,
                do_sample=False,
                pad_token_id=tok.eos_token_id,
            )
            xm.mark_step()
        dt = time.time() - t0
        new_tokens = out[0, ids.shape[1]:]
        text = tok.decode(new_tokens, skip_special_tokens=True)
        print(f"[{i+1}/{len(PROMPTS)}] {dt:6.1f}s  in_len={ids.shape[1]}  "
              f"response={text!r}")

    print("\n[PASS] tpu harmbench smoke complete")


if __name__ == "__main__":
    main()
