#!/usr/bin/env bash
# =============================================================================
# smoke_heretic_all.sh — focused Heretic smoke on all planned models.
#
# This is cheaper than smoke_final_eval_all.sh. It runs one APRS config per
# model, skips unrelated utility/prompt-attack evaluations, and keeps Heretic
# itself tiny while still exercising the save/reload/Optuna path.
#
# Models:
#   - meta-llama/Meta-Llama-3-8B-Instruct
#   - Qwen/Qwen3-8B
#   - google/gemma-2-9b-it
#
# Environment variables:
#   OUT_ROOT    Root output directory. Default: results/heretic_smoke_all
#   PYTHON_BIN  Python command. Default: python3. Use "uv run python" for uv.
# =============================================================================

set -euo pipefail

REPO_DIR="$(cd "$(dirname "$0")/../.." && pwd)"
OUT_ROOT="${OUT_ROOT:-$REPO_DIR/results/heretic_smoke_all}"
PYTHON_BIN="${PYTHON_BIN:-python3}"

MODELS=(
  "meta-llama/Meta-Llama-3-8B-Instruct"
  "Qwen/Qwen3-8B"
  "google/gemma-2-9b-it"
)

SLUGS=(
  "llama"
  "qwen"
  "gemma"
)

usage() {
  sed -n '2,18p' "$0"
}

while [[ $# -gt 0 ]]; do
  case "$1" in
    --output_dir) OUT_ROOT="$2"; shift 2 ;;
    -h|--help) usage; exit 0 ;;
    *) echo "Unknown argument: $1"; exit 1 ;;
  esac
done

read -r -a PYTHON_CMD <<< "$PYTHON_BIN"
mkdir -p "$OUT_ROOT"

echo "[smoke-heretic-all] output root: $OUT_ROOT"
echo "[smoke-heretic-all] python     : $PYTHON_BIN"
echo "[smoke-heretic-all] models     : ${MODELS[*]}"

for i in "${!MODELS[@]}"; do
  model="${MODELS[$i]}"
  slug="${SLUGS[$i]}"
  out_dir="$OUT_ROOT/$slug"
  mkdir -p "$out_dir"

  echo ""
  echo "[smoke-heretic-all] === $slug :: $model ==="

  "${PYTHON_CMD[@]}" -u "$REPO_DIR/run_obfuscation_pipeline.py" \
    --model_path "$model" \
    --save_csv "$out_dir/heretic_smoke.csv" \
    --artifact_subdir "heretic_smoke" \
    --projection_mode full \
    --epsilon 0.025 \
    --num_calibration_prompts 8 \
    --per_layer_direction \
    --writer_output_directions \
    --skip_evaluations \
    --skip_integrity_eval \
    --skip_adaptive_attacks \
    --skip_leace \
    --skip_harmbench \
    --skip_xstest \
    --skip_lm_harness \
    --skip_alpacaeval \
    --heretic_trials 1 \
    --heretic_train_samples 4 \
    --heretic_eval_samples 2 \
    --heretic_max_response_length 16 \
    2>&1 | tee "$out_dir/heretic_smoke.log"
done

echo ""
echo "[smoke-heretic-all] complete: $OUT_ROOT"
