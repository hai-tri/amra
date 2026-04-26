#!/usr/bin/env bash
# =============================================================================
# smoke_run.sh — End-to-end sanity check for the APRS pipeline.
#
# Runs a tiny obfuscation + evaluation pass on a single model with every
# heavy stage either skipped or clamped to minimum sample counts.  The goal
# is to exercise every integration point (direction extraction, empirical
# probing, writer + reader patching, benchmarks, CSV writing) fast enough
# to catch regressions without burning a GPU-hour.
#
# Usage:
#   bash smoke_run.sh                        # default: Llama-3-8B-Instruct
#   bash smoke_run.sh --model <HF_ID>
#   bash smoke_run.sh --model <HF_ID> --output_dir <DIR>
#
# Model choices validated by scripts/smoke_check_architectures.py:
#   - meta-llama/Meta-Llama-3-8B-Instruct   (baseline)
#   - Qwen/Qwen3-8B
#   - google/gemma-2-9b-it                  (exercises pre_feedforward LN)
#   - google/gemma-3-9b-it                  (exercises pre_feedforward LN)
#   - mistralai/Mistral-7B-Instruct-v0.3
#
# What this runs:
#   - Direction extraction  : small (n_train=64, n_val=16)
#   - Obfuscation           : full mode, ε=0.025, 1 pertinent layer, 8 cal prompts
#   - Direction source      : per-writer output directions
#   - Loss / PPL / BPB      : 8 batches (Pile + Alpaca)
#   - HarmBench / Heretic   : skipped (expensive)
#   - GCG / AutoDAN / PAIR  : skipped
#   - XSTest                : skipped
#   - lm-harness            : 10 samples per task (gsm8k only)
#   - AlpacaEval            : 5 prompts, judging skipped
#
# Expected wall-clock on a single A100 for Llama-3-8B: 5–10 minutes.
# =============================================================================

set -euo pipefail

REPO_DIR="$(cd "$(dirname "$0")/../.." && pwd)"
MODEL_ID="meta-llama/Meta-Llama-3-8B-Instruct"
OUTPUT_DIR="$REPO_DIR/results/smoke"

while [[ $# -gt 0 ]]; do
    case "$1" in
        --model)      MODEL_ID="$2";   shift 2 ;;
        --output_dir) OUTPUT_DIR="$2"; shift 2 ;;
        -h|--help)
            sed -n '2,30p' "$0"; exit 0 ;;
        *) echo "Unknown argument: $1"; exit 1 ;;
    esac
done

mkdir -p "$OUTPUT_DIR"
TIMESTAMP=$(date +%Y%m%d_%H%M%S)
CSV_PATH="$OUTPUT_DIR/smoke_${TIMESTAMP}.csv"
LOG_PATH="$OUTPUT_DIR/smoke_${TIMESTAMP}.log"

# Short model tag for filesystem / logs
MODEL_TAG="$(echo "$MODEL_ID" | sed 's|.*/||' | tr '[:upper:]' '[:lower:]')"

echo "================================================================" | tee "$LOG_PATH"
echo " APRS Smoke Test — $(date)"                                      | tee -a "$LOG_PATH"
echo " Model      : $MODEL_ID ($MODEL_TAG)"                            | tee -a "$LOG_PATH"
echo " CSV        : $CSV_PATH"                                         | tee -a "$LOG_PATH"
echo " Log        : $LOG_PATH"                                         | tee -a "$LOG_PATH"
echo "================================================================" | tee -a "$LOG_PATH"

python3 -u "$REPO_DIR/run_obfuscation_pipeline.py" \
    --model_path "$MODEL_ID" \
    --save_csv "$CSV_PATH" \
    --defense_type obfuscation \
    --projection_mode full \
    --epsilon 0.025 \
    --num_pertinent_layers 1 \
    --num_calibration_prompts 8 \
    --per_layer_direction \
    --writer_output_directions \
    --seed 42 \
    --skip_harmbench \
    --skip_xstest \
    --skip_leace \
    --skip_heretic \
    --lm_harness_tasks gsm8k \
    --lm_harness_n 10 \
    --alpacaeval_n 5 \
    --alpacaeval_max_new_tokens 64 \
    --alpacaeval_skip_judge \
    2>&1 | tee -a "$LOG_PATH"

echo ""                                                                | tee -a "$LOG_PATH"
echo "================================================================" | tee -a "$LOG_PATH"
echo " Smoke test complete — $(date)"                                  | tee -a "$LOG_PATH"
echo " Inspect: $CSV_PATH"                                             | tee -a "$LOG_PATH"
echo "================================================================" | tee -a "$LOG_PATH"
