"""
TPU smoke test for the APRS pipeline.

Verifies — on a single TPU worker — that:
  1. torch_xla loads and sees at least one device.
  2. Llama-3-8B-Instruct loads onto XLA in bf16 without OOM.
  3. Forward-pre-hooks (the mechanism the defense relies on) work under XLA.
  4. Rank-1 weight edit (the writer-patch kernel) runs on XLA tensors.
  5. model.generate() produces tokens after compilation.

Runs on worker 0 alone. Does NOT test multi-host SPMD or data-parallel —
that's a separate milestone. This only establishes that the single-device
XLA path is healthy.

Exits 0 on success, nonzero on any failure. Prints a clear PASS/FAIL line.
"""

import os
import sys
import time
import traceback

REPO = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.insert(0, REPO)
sys.path.insert(0, os.path.join(REPO, "refusal_direction"))


def step(msg):
    print(f"\n── {msg}", flush=True)


def fail(msg, e=None):
    print(f"\n[FAIL] {msg}", flush=True)
    if e is not None:
        traceback.print_exc()
    sys.exit(1)


def main():
    # ------------------------------------------------------------------
    # 1. torch_xla imports
    # ------------------------------------------------------------------
    step("1/5  import torch, torch_xla")
    try:
        import torch
        import torch_xla
        import torch_xla.core.xla_model as xm
        dev = xm.xla_device()
        print(f"torch     = {torch.__version__}")
        print(f"torch_xla = {torch_xla.__version__}")
        print(f"device    = {dev}")
    except Exception as e:
        fail("could not import torch_xla", e)

    # ------------------------------------------------------------------
    # 2. Load Llama-3-8B-Instruct in bf16
    # ------------------------------------------------------------------
    step("2/5  load meta-llama/Meta-Llama-3-8B-Instruct in bf16")
    model_id = os.environ.get("APRS_SMOKE_MODEL", "meta-llama/Meta-Llama-3-8B-Instruct")
    t0 = time.time()
    try:
        from transformers import AutoModelForCausalLM, AutoTokenizer
        tok = AutoTokenizer.from_pretrained(model_id)
        if tok.pad_token is None:
            tok.pad_token = tok.eos_token
        model = AutoModelForCausalLM.from_pretrained(
            model_id,
            torch_dtype=torch.bfloat16,
            attn_implementation="eager",
            low_cpu_mem_usage=True,
        ).to(dev)
        model.eval()
        xm.mark_step()
        print(f"loaded in {time.time() - t0:.1f}s "
              f"(params={sum(p.numel() for p in model.parameters())/1e9:.2f} B)")
    except Exception as e:
        fail("model load failed", e)

    # ------------------------------------------------------------------
    # 3. Forward pre-hook under XLA
    # ------------------------------------------------------------------
    step("3/5  forward pre-hook")
    try:
        hit = {"count": 0}

        def pre_hook(module, inputs):
            hit["count"] += 1
            return None

        layer = model.model.layers[10].self_attn.o_proj
        handle = layer.register_forward_pre_hook(pre_hook)
        with torch.no_grad():
            ids = tok("hello", return_tensors="pt").input_ids.to(dev)
            out = model(ids)
            xm.mark_step()
            _ = out.logits.cpu()
        handle.remove()
        assert hit["count"] > 0, "hook never fired"
        print(f"hook fired {hit['count']}x, logits shape {out.logits.shape}")
    except Exception as e:
        fail("pre-hook test failed", e)

    # ------------------------------------------------------------------
    # 4. Rank-1 weight edit (writer-patch kernel)
    # ------------------------------------------------------------------
    step("4/5  rank-1 weight update on XLA tensors")
    try:
        from obfuscation_utils import rank_one_update
        W = layer.weight.data
        d_in = W.shape[1]
        x = torch.randn(d_in, device=dev, dtype=torch.float32)
        target = torch.randn(W.shape[0], device=dev, dtype=torch.float32)
        W_new = rank_one_update(W, x, target)
        xm.mark_step()
        err = (W_new.float() @ x - target).norm().item()
        rel = err / max(target.norm().item(), 1e-12)
        layer.weight.data = W_new  # leave patched for next step
        print(f"rank-1 residual error = {err:.2e} (rel = {rel:.2e})")
        # bf16 weights → ~1e-2 relative error is expected on XLA/TPU
        assert rel < 5e-2, f"rank-1 update failed — rel residual {rel}"
    except Exception as e:
        fail("rank-1 update failed", e)

    # ------------------------------------------------------------------
    # 5. Generate
    # ------------------------------------------------------------------
    step("5/5  model.generate (20 tokens)")
    try:
        t0 = time.time()
        prompt = "The capital of France is"
        ids = tok(prompt, return_tensors="pt").input_ids.to(dev)
        with torch.no_grad():
            out_ids = model.generate(
                ids, max_new_tokens=20, do_sample=False,
                pad_token_id=tok.eos_token_id,
            )
            xm.mark_step()
        text = tok.decode(out_ids[0], skip_special_tokens=True)
        print(f"gen in {time.time() - t0:.1f}s")
        print(f"output: {text!r}")
    except Exception as e:
        fail("generate failed", e)

    print("\n[PASS] smoke test complete")


if __name__ == "__main__":
    main()
