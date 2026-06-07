#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
cd "$ROOT_DIR"

MODELS=(llama qwen gemma)
OUT_ROOT="${OUT_ROOT:-results/optuna_smoke}"
PYTHON_BIN="${PYTHON_BIN:-python3}"

if [[ "${1:-}" == "-h" || "${1:-}" == "--help" ]]; then
  cat <<'EOF'
Usage: scripts/sweeps/smoke_optuna_all.sh

Runs the Optuna sweep smoke test sequentially for llama, qwen, and gemma.

Environment variables:
  OUT_ROOT    Output directory root. Default: results/optuna_smoke
  PYTHON_BIN  Python command. Default: python3

Examples:
  scripts/sweeps/smoke_optuna_all.sh
  PYTHON_BIN="uv run python" scripts/sweeps/smoke_optuna_all.sh
  OUT_ROOT=results/my_smoke scripts/sweeps/smoke_optuna_all.sh
EOF
  exit 0
fi

read -r -a PYTHON_CMD <<< "$PYTHON_BIN"

mkdir -p "$OUT_ROOT"

echo "[smoke-optuna] output root: $OUT_ROOT"
echo "[smoke-optuna] models: ${MODELS[*]}"

for model in "${MODELS[@]}"; do
  echo
  echo "================================================================"
  echo "[smoke-optuna] model=$model"
  echo "================================================================"

  "${PYTHON_CMD[@]}" scripts/sweeps/sweep_optuna.py \
    --model "$model" \
    --smoke \
    --n_trials 2 \
    --n 4 \
    --n_train 16 \
    --n_val 8 \
    --num_calibration_prompts 8 \
    --attack_batch_size 4 \
    --forward_batch_size 4 \
    --bpb_batches 1 \
    --mmlu_n 5 \
    --math500_n 5 \
    --utility_batch_size 2 \
    --output_dir "$OUT_ROOT/$model"
done

echo
echo "[smoke-optuna] completed all models"
