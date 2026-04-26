#!/usr/bin/env bash
# =============================================================================
# setup_tpu_v4.sh — One-shot setup on a TPU v4 VM (worker-local).
#
# Creates ~/.venvs/aprs-xla (Python 3.10 via conda) and installs
# torch 2.5 + torch_xla[tpu] 2.5 + APRS requirements. Safe to re-run.
#
# Assumes: miniconda3 at ~/miniconda3 (typical TPU VM layout).
# Runs on ONE worker at a time. For multi-host runs, re-run with --worker=all.
# =============================================================================

set -euo pipefail

REPO_DIR="${APRS_REPO_DIR:-$HOME/APRS}"
VENV="$HOME/.venvs/aprs-xla"
HF_TOKEN=""

while [[ $# -gt 0 ]]; do
    case "$1" in
        --token)    HF_TOKEN="$2"; shift 2 ;;
        --repo_dir) REPO_DIR="$2"; shift 2 ;;
        *) echo "Unknown arg: $1"; exit 1 ;;
    esac
done

echo "================================================================"
echo " APRS TPU setup — $(date)"
echo " Host  : $(hostname)"
echo " Repo  : $REPO_DIR"
echo " Venv  : $VENV"
echo "================================================================"

# 1. Conda env with Python 3.10
if [[ ! -x "$VENV/bin/python3" ]]; then
    source "$HOME/miniconda3/etc/profile.d/conda.sh"
    # Non-interactive ToS acceptance for default channels (newer conda requires this)
    conda tos accept --override-channels --channel https://repo.anaconda.com/pkgs/main 2>/dev/null || true
    conda tos accept --override-channels --channel https://repo.anaconda.com/pkgs/r    2>/dev/null || true
    conda create -y -p "$VENV" python=3.10
fi
PY="$VENV/bin/python3"
PIP="$VENV/bin/pip"

# 2. torch + torch_xla (TPU wheels, CPU torch)
"$PIP" install -q --upgrade pip
"$PIP" install -q "torch==2.5.0" --index-url https://download.pytorch.org/whl/cpu
"$PIP" install -q "torch_xla[tpu]==2.5.0" \
    -f https://storage.googleapis.com/libtpu-releases/index.html

echo ""
echo "── torch / torch_xla check ─────"
"$PY" - <<'PYEOF'
import torch, torch_xla, torch_xla.core.xla_model as xm
print("torch     :", torch.__version__)
print("torch_xla :", torch_xla.__version__)
print("xla dev   :", xm.xla_device())
print("#devices  :", xm.xrt_world_size() if hasattr(xm, "xrt_world_size") else len(xm.get_xla_supported_devices()))
PYEOF

# 3. APRS requirements (skip flash_attn/vllm/heretic — TPU-hostile)
if [[ -f "$REPO_DIR/requirements.txt" ]]; then
    echo ""
    echo "── Installing APRS requirements (filtered for TPU) …"
    grep -vE '^(flash[_-]attn|vllm|bitsandbytes|transformers)' "$REPO_DIR/requirements.txt" \
        > /tmp/req_tpu.txt
    "$PIP" install -q -r /tmp/req_tpu.txt
    # Pin transformers below the versions that hard-require flash_attn in PACKAGE_DISTRIBUTION_MAPPING
    "$PIP" install -q "transformers==4.46.3"
fi

# 4. flash_attn stub so HF doesn't try to import it
SITE=$("$PY" -c 'import sysconfig; print(sysconfig.get_paths()["purelib"])')
mkdir -p "$SITE/flash_attn"
cat > "$SITE/flash_attn/__init__.py" <<'PYEOF'
def flash_attn_func(*a, **k): raise NotImplementedError
def flash_attn_varlen_func(*a, **k): raise NotImplementedError
PYEOF
mkdir -p "$SITE/flash_attn-0.0.0.dist-info"
printf 'Metadata-Version: 2.1\nName: flash_attn\nVersion: 0.0.0\n' \
    > "$SITE/flash_attn-0.0.0.dist-info/METADATA"

# 5. Patch llama3_model.py for eager attention (already done on main normally)
LLAMA_MODEL="$REPO_DIR/refusal_direction/pipeline/model_utils/llama3_model.py"
if [[ -f "$LLAMA_MODEL" ]] && ! grep -q 'attn_implementation' "$LLAMA_MODEL"; then
    sed -i 's/trust_remote_code=True,$/trust_remote_code=True,\n            attn_implementation="eager",/' "$LLAMA_MODEL"
    echo "   patched llama3_model.py"
fi

# 6. HF login (optional) — write token to hub cache, works across hf/huggingface-cli versions
if [[ -n "$HF_TOKEN" ]]; then
    mkdir -p "$HOME/.cache/huggingface"
    printf '%s' "$HF_TOKEN" > "$HOME/.cache/huggingface/token"
    chmod 600 "$HOME/.cache/huggingface/token"
    echo "   wrote HF token to ~/.cache/huggingface/token"
fi

echo ""
echo "================================================================"
echo " Done. Activate: source $VENV/bin/activate"
echo " Smoke test    : python $REPO_DIR/scripts/smoke_test_tpu.py"
echo "================================================================"
