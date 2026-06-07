#!/usr/bin/env bash
# =============================================================================
# smoke_final_eval_all.sh — run the tiny final-eval smoke on all planned models.
#
# Models:
#   - meta-llama/Meta-Llama-3-8B-Instruct
#   - Qwen/Qwen3-8B
#   - google/gemma-2-9b-it
#
# Environment variables:
#   OUT_ROOT    Root output directory. Default: results/final_eval_smoke_all
#   PYTHON_BIN  Python command passed through to smoke_final_eval.sh.
#               Default: python3. Use "uv run python" for the uv environment.
# =============================================================================

set -euo pipefail

REPO_DIR="$(cd "$(dirname "$0")/../.." && pwd)"
OUT_ROOT="${OUT_ROOT:-$REPO_DIR/results/final_eval_smoke_all}"
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
  sed -n '2,17p' "$0"
}

while [[ $# -gt 0 ]]; do
  case "$1" in
    --output_dir) OUT_ROOT="$2"; shift 2 ;;
    -h|--help) usage; exit 0 ;;
    *) echo "Unknown argument: $1"; exit 1 ;;
  esac
done

mkdir -p "$OUT_ROOT"

echo "[smoke-final-all] output root: $OUT_ROOT"
echo "[smoke-final-all] python     : $PYTHON_BIN"
echo "[smoke-final-all] models     : ${MODELS[*]}"

for i in "${!MODELS[@]}"; do
  model="${MODELS[$i]}"
  slug="${SLUGS[$i]}"
  out_dir="$OUT_ROOT/$slug"

  echo ""
  echo "[smoke-final-all] === $slug :: $model ==="

  PYTHON_BIN="$PYTHON_BIN" bash "$REPO_DIR/scripts/eval/smoke_final_eval.sh" \
    --model "$model" \
    --output_dir "$out_dir"
done

echo ""
echo "[smoke-final-all] complete: $OUT_ROOT"
