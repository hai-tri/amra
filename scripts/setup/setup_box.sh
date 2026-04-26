#!/usr/bin/env bash
# =============================================================================
# setup_box.sh — One-shot setup for a fresh Lambda Labs box (Ubuntu 22.04)
#
# Usage:
#   bash scripts/setup_box.sh [--token HF_TOKEN]
#
# What it does:
#   1. Creates a clean venv at ~/.venvs/aprs-clean (no system-site-packages)
#   2. Installs torch 2.x from cu124 wheels
#   3. Installs all requirements from requirements.txt
#   4. Creates a flash_attn stub (avoids ABI conflicts with system flash_attn)
#   5. Patches lm_eval's dtype kwarg (dtype= → torch_dtype=)
#   6. Patches llama3_model.py to use attn_implementation="eager"
#   7. Optionally logs in to HuggingFace
# =============================================================================

set -euo pipefail

REPO_DIR="$(cd "$(dirname "$0")/../.." && pwd)"
VENV="$HOME/.venvs/aprs-clean"
HF_TOKEN=""

# Parse args
while [[ $# -gt 0 ]]; do
    case "$1" in
        --token) HF_TOKEN="$2"; shift 2 ;;
        *) echo "Unknown argument: $1"; exit 1 ;;
    esac
done

echo "================================================================"
echo " APRS Box Setup — $(date)"
echo " Repo    : $REPO_DIR"
echo " Venv    : $VENV"
echo "================================================================"

# ------------------------------------------------------------------
# 1. Create clean venv
# ------------------------------------------------------------------
echo ""
echo "── Step 1: Creating venv at $VENV …"
python3 -m venv "$VENV" --without-pip
curl -sS https://bootstrap.pypa.io/get-pip.py | "$VENV/bin/python3"
echo "   venv ready"

# ------------------------------------------------------------------
# 2. Install torch from cu124 wheels
# ------------------------------------------------------------------
echo ""
echo "── Step 2: Installing torch + torchvision (cu124) …"
"$VENV/bin/pip" install torch torchvision \
    --index-url https://download.pytorch.org/whl/cu124 -q
echo "   torch ready: $("$VENV/bin/python3" -c 'import torch; print(torch.__version__)')"

# ------------------------------------------------------------------
# 3. Install requirements
# ------------------------------------------------------------------
echo ""
echo "── Step 3: Installing requirements.txt …"
"$VENV/bin/pip" install -r "$REPO_DIR/requirements.txt" -q
echo "   requirements installed"

# Re-pin torch after vllm may have overwritten it with a CPU build
echo ""
echo "── Step 3b: Re-pinning torch to cu124 build (vllm may have overwritten it) …"
"$VENV/bin/pip" install torch==2.5.1 torchvision==0.20.1 \
    --index-url https://download.pytorch.org/whl/cu124 -q
CUDA_OK=$("$VENV/bin/python3" -c 'import torch; print(torch.cuda.is_available())')
echo "   torch $("$VENV/bin/python3" -c 'import torch; print(torch.__version__)') — CUDA: $CUDA_OK"
if [[ "$CUDA_OK" != "True" ]]; then
    echo "   ERROR: CUDA not available after torch reinstall — check driver/CUDA version"
    exit 1
fi

# ------------------------------------------------------------------
# 4. flash_attn stub (prevents ABI conflict with system flash_attn)
# ------------------------------------------------------------------
echo ""
echo "── Step 4: Creating flash_attn stub …"
SITE="$VENV/lib/python3.10/site-packages"
mkdir -p "$SITE/flash_attn"
cat > "$SITE/flash_attn/__init__.py" << 'PYEOF'
def flash_attn_func(*a, **k): raise NotImplementedError
def flash_attn_varlen_func(*a, **k): raise NotImplementedError
PYEOF
mkdir -p "$SITE/flash_attn-0.0.0.dist-info"
printf 'Metadata-Version: 2.1\nName: flash_attn\nVersion: 0.0.0\n' \
    > "$SITE/flash_attn-0.0.0.dist-info/METADATA"
echo "   flash_attn stub created"

# ------------------------------------------------------------------
# 5. Patch lm_eval: dtype= → torch_dtype=
# ------------------------------------------------------------------
echo ""
echo "── Step 5: Patching lm_eval huggingface.py …"
LM_EVAL_HF="$SITE/lm_eval/models/huggingface.py"
if [[ -f "$LM_EVAL_HF" ]]; then
    sed -i 's/dtype=get_dtype(dtype)/torch_dtype=get_dtype(dtype)/g' "$LM_EVAL_HF"
    COUNT=$(grep -c 'torch_dtype=get_dtype' "$LM_EVAL_HF" || true)
    echo "   patched $COUNT occurrence(s) in huggingface.py"
else
    echo "   WARNING: $LM_EVAL_HF not found — skipping"
fi

# ------------------------------------------------------------------
# 6. Patch llama3_model.py: add attn_implementation="eager"
# ------------------------------------------------------------------
echo ""
echo "── Step 6: Patching llama3_model.py (attn_implementation=eager) …"
LLAMA_MODEL="$REPO_DIR/refusal_direction/pipeline/model_utils/llama3_model.py"
if [[ -f "$LLAMA_MODEL" ]]; then
    if grep -q 'attn_implementation' "$LLAMA_MODEL"; then
        echo "   already patched — skipping"
    else
        sed -i 's/trust_remote_code=True,$/trust_remote_code=True,\n            attn_implementation="eager",/' "$LLAMA_MODEL"
        echo "   patched"
    fi
else
    echo "   WARNING: $LLAMA_MODEL not found — skipping"
fi

# ------------------------------------------------------------------
# 7. HuggingFace login
# ------------------------------------------------------------------
if [[ -n "$HF_TOKEN" ]]; then
    echo ""
    echo "── Step 7: Logging in to HuggingFace …"
    "$VENV/bin/huggingface-cli" login --token "$HF_TOKEN"
    echo "   logged in"
else
    echo ""
    echo "── Step 7: HuggingFace login skipped (no --token provided)"
    echo "   Run: huggingface-cli login --token <your_token>"
fi

# ------------------------------------------------------------------
# Done
# ------------------------------------------------------------------
echo ""
echo "================================================================"
echo " Setup complete — $(date)"
echo " Activate venv : source $VENV/bin/activate"
echo " Run sweep     : bash scripts/run_sweep.sh"
echo " Run final eval: bash scripts/run_final_eval.sh"
echo "================================================================"
