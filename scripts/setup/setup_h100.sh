#!/usr/bin/env bash
# One-shot bootstrap for H100/A100 boxes (Lambda/Modal/RunPod/…).
# Idempotent — safe to re-run.
#
# Pre-reqs: CUDA 12.4 driver, ~200 GB disk, HF_TOKEN in env.
#
# Usage:
#   export HF_TOKEN=hf_xxx
#   bash scripts/setup_h100.sh
#   bash scripts/run_final_eval.sh --model meta-llama/Meta-Llama-3-8B-Instruct

set -euo pipefail

REPO_DIR="$(cd "$(dirname "$0")/../.." && pwd)"
cd "$REPO_DIR"

: "${HF_TOKEN:?HF_TOKEN env var is required}"

# ---- Python env ----
if [[ ! -d .venv ]]; then
    python3 -m venv .venv
fi
# shellcheck disable=SC1091
source .venv/bin/activate
pip install --upgrade pip wheel

# Torch first, pinned to match vllm / lm-eval.
pip install torch==2.5.1 torchvision==0.20.1 \
    --index-url https://download.pytorch.org/whl/cu124

pip install -r requirements.txt

# vLLM can drag in a CPU torch; re-pin.
pip install torch==2.5.1 torchvision==0.20.1 \
    --index-url https://download.pytorch.org/whl/cu124

# ---- HF auth + weight prefetch ----
mkdir -p ~/.cache/huggingface
echo -n "$HF_TOKEN" > ~/.cache/huggingface/token
chmod 600 ~/.cache/huggingface/token

MODELS=(
    "meta-llama/Meta-Llama-3-8B-Instruct"
    "Qwen/Qwen3-8B"
    "google/gemma-2-9b-it"
    "mistralai/Mistral-7B-Instruct-v0.3"
    # Judges
    "cais/HarmBench-Llama-2-13b-cls"
    "meta-llama/LlamaGuard-7b"
)
for m in "${MODELS[@]}"; do
    echo ">> prefetch $m"
    huggingface-cli download "$m" --quiet || echo "   [WARN] $m download failed"
done

# ---- HarmBench repo (GCG/AutoDAN wrappers need it) ----
if [[ ! -d attacks/HarmBench ]]; then
    git clone --depth=1 https://github.com/centerforaisafety/HarmBench.git attacks/HarmBench
fi

echo ""
echo "Setup complete."
nvidia-smi | head -4
python3 -c "import torch; print('torch', torch.__version__, 'cuda', torch.cuda.is_available(), 'dev', torch.cuda.get_device_name(0) if torch.cuda.is_available() else 'none')"
python3 -c "import lm_eval; print('lm_eval', lm_eval.__version__)"
