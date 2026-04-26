#!/usr/bin/env bash
# =============================================================================
# setup.sh — One-shot environment setup for APRS on a Lambda Labs GH200
#
# Usage:
#   bash setup.sh
#
# What it does:
#   1. Clones the repo with submodules
#   2. Creates a Python virtualenv that reuses the system CUDA Torch install
#   3. Installs a GH200-safe dependency set
#   4. Authenticates with HuggingFace (uses HF_TOKEN/HUGGINGFACE_HUB_TOKEN if set)
#   5. Pre-downloads the target model weights
# =============================================================================

set -euo pipefail

REPO_URL="https://github.com/hai-tri/APRS.git"
REPO_DIR="$HOME/APRS"
MODEL_ID="meta-llama/Meta-Llama-3-8B-Instruct"
VENV_DIR="${APRS_VENV:-$HOME/.venvs/aprs}"

echo "================================================================"
echo " APRS Setup Script"
echo "================================================================"

# ── 1. Clone repo ────────────────────────────────────────────────────
if [ -d "$REPO_DIR" ]; then
    echo "[setup] Repo already exists at $REPO_DIR — pulling latest …"
    cd "$REPO_DIR"
    git pull
    git submodule update --init --recursive
else
    echo "[setup] Cloning $REPO_URL …"
    git clone --recurse-submodules "$REPO_URL" "$REPO_DIR"
    cd "$REPO_DIR"
fi

# ── 2. Create a venv and install Python dependencies ────────────────
echo "[setup] Creating virtualenv at $VENV_DIR …"
python3 -m venv --system-site-packages "$VENV_DIR"
PYTHON_BIN="$VENV_DIR/bin/python"
PIP_BIN="$PYTHON_BIN -m pip"

echo "[setup] Installing Python dependencies …"
$PIP_BIN install --quiet --upgrade pip setuptools wheel
$PIP_BIN install --quiet \
    "numpy<2" \
    "scipy>=1.11,<1.14" \
    "transformers==4.46.3" \
    "datasets>=2.19,<3" \
    "accelerate>=1.13.0" \
    "peft>=0.10" \
    "optuna>=3,<5" \
    "tqdm>=4.66" \
    "scikit-learn>=1.4" \
    "pandas>=2.2" \
    "sentencepiece>=0.2" \
    "protobuf>=4,<6" \
    "lm-eval>=0.4" \
    "jaxtyping>=0.2" \
    "einops>=0.8" \
    "matplotlib>=3.8" \
    "jinja2>=3.1.4" \
    "safetensors>=0.4" \
    "huggingface-hub>=0.23" \
    "fsspec[http]==2024.6.1" \
    "zstandard>=0.22" \
    "pydantic>=2" \
    "pydantic-settings>=2" \
    "bitsandbytes>=0.43" \
    "questionary>=2"

echo "[setup] Skipping refusal_direction/requirements.txt because it pins"
echo "        stale CUDA wheels that break on GH200/aarch64 environments."

echo "[setup] Dependencies installed."

# ── 3. HuggingFace authentication ────────────────────────────────────
echo ""
echo "[setup] HuggingFace login required for $MODEL_ID"
echo "        Get your token at: https://huggingface.co/settings/tokens"
echo ""
HF_TOKEN_VALUE="${HF_TOKEN:-${HUGGINGFACE_HUB_TOKEN:-}}"
if [ -n "$HF_TOKEN_VALUE" ]; then
    echo "[setup] Using HuggingFace token from environment …"
    mkdir -p "$HOME/.cache/huggingface"
    printf %s "$HF_TOKEN_VALUE" > "$HOME/.cache/huggingface/token"
    chmod 600 "$HOME/.cache/huggingface/token"
else
    "$VENV_DIR/bin/huggingface-cli" login
fi

# ── 4. Pre-download model weights ────────────────────────────────────
echo "[setup] Pre-downloading $MODEL_ID …"
$PYTHON_BIN -c "
from transformers import AutoTokenizer, AutoModelForCausalLM
import torch
print('Downloading tokenizer …')
AutoTokenizer.from_pretrained('$MODEL_ID')
print('Downloading model weights …')
AutoModelForCausalLM.from_pretrained('$MODEL_ID', torch_dtype=torch.bfloat16)
print('Model downloaded successfully.')
"

# ── 5. Pre-download HarmBench classifier ─────────────────────────────
echo "[setup] Pre-downloading HarmBench classifier …"
$PYTHON_BIN -c "
from transformers import AutoTokenizer, AutoModelForCausalLM
import torch
model_id = 'cais/HarmBench-Llama-2-13b-cls'
print('Downloading HarmBench classifier …')
AutoTokenizer.from_pretrained(model_id)
AutoModelForCausalLM.from_pretrained(model_id, torch_dtype=torch.bfloat16)
print('HarmBench classifier downloaded.')
"

echo ""
echo "================================================================"
echo " Setup complete. Run the sweep with:"
echo "   source $VENV_DIR/bin/activate"
echo "   bash $REPO_DIR/scripts/run_sweep.sh"
echo "================================================================"
