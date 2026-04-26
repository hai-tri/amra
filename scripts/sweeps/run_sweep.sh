#!/usr/bin/env bash
# =============================================================================
# run_sweep.sh — ε hyperparameter sweep for APRS (full projection mode)
#
# Usage:
#   bash scripts/run_sweep.sh [--model MODEL_ID] [--output_dir DIR]
#
# Sweeps ε ∈ {0.01, 0.025, 0.05, 0.1, 0.15, 0.2, 0.3, 0.5} using the full
# projection mode on a single model.  Automatic pertinent-layer selection is
# used throughout (no --num_pertinent_layers override) so the reported
# num_layers reflects the self-tuned defense at each ε.
#
# Full benchmark stack is enabled (same as run_final_eval.sh):
#   - LlamaGuard scoring
#   - GCG / AutoDAN / PAIR / ReNeLLM / SoftOpt attacks (25 behaviors each)
#   - HarmBench pre/post-attack scoring
#   - XSTest, utility benchmarks (Pile BPB, Alpaca BPB, GSM8k, MATH500, MMLU)
#   - AlpacaEval (805 prompts, GPT-4 judging if OPENAI_API_KEY is set)
#
# The refusal direction is extracted once (during the undefended baseline run)
# and reused across all ε configs via --skip_direction_extraction.  Already-
# complete CSVs are skipped so the script is safe to re-run after interruption.
#
# Results for each ε are written to individual CSVs and aggregated into
# sweep_results.csv at the end.
# =============================================================================

set -euo pipefail

REPO_DIR="$(cd "$(dirname "$0")/../.." && pwd)"
OUTPUT_BASE="$HOME/aprs_sweep"
EXTRA_FLAGS_STR="${APRS_EXTRA_FLAGS:-}"
_EXTRA_PIPELINE_FLAGS=()
if [[ -n "$EXTRA_FLAGS_STR" ]]; then
    read -r -a _EXTRA_PIPELINE_FLAGS <<< "$EXTRA_FLAGS_STR"
fi

while [[ $# -gt 0 ]]; do
    case "$1" in
        --output_dir) OUTPUT_BASE="$2"; shift 2 ;;
        *) echo "Unknown argument: $1"; exit 1 ;;
    esac
done

MODELS=(
    "meta-llama/Meta-Llama-3-8B-Instruct"
    "Qwen/Qwen3-8B"
    "google/gemma-2-9b-it"
)

mkdir -p "$OUTPUT_BASE"
TIMESTAMP=$(date +%Y%m%d_%H%M%S)
MASTER_LOG="$OUTPUT_BASE/sweep_${TIMESTAMP}.log"

echo "================================================================" | tee "$MASTER_LOG"
echo " APRS ε Sweep — $(date)"                                         | tee -a "$MASTER_LOG"
echo " Models     : ${MODELS[*]}"                                      | tee -a "$MASTER_LOG"
echo " Output dir : $OUTPUT_BASE"                                      | tee -a "$MASTER_LOG"
echo " Extra args : ${EXTRA_FLAGS_STR:-<none>}"                        | tee -a "$MASTER_LOG"
echo "================================================================" | tee -a "$MASTER_LOG"

_SKIP_FLAGS=(
    --skip_harmbench
    --skip_xstest
    --skip_heretic
    --alpacaeval_skip_judge
)

# run_config <model_id> <tag> [pipeline_args...]
run_config() {
    local model_id="$1"; shift
    local tag="$1"; shift
    local extra_args=("$@")

    local csv_path="$OUTPUT_BASE/${tag}.csv"
    local log_path="$OUTPUT_BASE/${tag}.log"

    echo "" | tee -a "$MASTER_LOG"
    echo "── $tag" | tee -a "$MASTER_LOG"
    echo "   model : $model_id" | tee -a "$MASTER_LOG"
    echo "   args  : ${extra_args[*]}" | tee -a "$MASTER_LOG"

    if [[ -s "$csv_path" ]]; then
        echo "   [SKIP] already complete → $csv_path" | tee -a "$MASTER_LOG"
        return
    fi

    set +e
    python3 -u "$REPO_DIR/run_obfuscation_pipeline.py" \
        --model_path "$model_id" \
        --save_csv "$csv_path" \
        "${_SKIP_FLAGS[@]}" \
        "${extra_args[@]}" \
        "${_EXTRA_PIPELINE_FLAGS[@]}" \
        2>&1 | tee "$log_path"
    local rc=$?
    set -e

    if [[ $rc -eq 0 ]]; then
        echo "   [OK] → $csv_path" | tee -a "$MASTER_LOG"
    else
        echo "   [WARN] exited $rc — see $log_path" | tee -a "$MASTER_LOG"
    fi
}

EPSILONS=(0.01 0.025 0.05 0.1 0.15 0.2 0.3 0.5)

for model_id in "${MODELS[@]}"; do
    model_tag="$(echo "$model_id" | sed 's|.*/||' | tr '[:upper:]' '[:lower:]')"

    echo "" | tee -a "$MASTER_LOG"
    echo "════ $model_tag ════" | tee -a "$MASTER_LOG"

    # Undefended baseline — extracts and caches the refusal direction for this model
    run_config "$model_id" "sweep_${model_tag}_undefended" \
        --undefended_only

    # ε sweep — full projection mode, automatic pertinent-layer selection
    for eps in "${EPSILONS[@]}"; do
        tag="sweep_${model_tag}_full_eps$(echo "$eps" | tr '.' '_')"
        run_config "$model_id" "$tag" \
            --defense_type obfuscation \
            --projection_mode full \
            --epsilon "$eps" \
            --num_calibration_prompts 128 \
            --per_layer_direction \
            --writer_output_directions \
            --skip_direction_extraction \
            --seed 42
    done
done

echo "" | tee -a "$MASTER_LOG"
echo "── Aggregating results ──────────────────────────────────────" | tee -a "$MASTER_LOG"

OUTPUT_BASE="$OUTPUT_BASE" python3 - <<'PYEOF' 2>&1 | tee -a "$MASTER_LOG"
import csv, glob, os

output_base = os.environ["OUTPUT_BASE"]
all_rows, all_keys = [], []

for csv_path in sorted(glob.glob(os.path.join(output_base, "sweep_*.csv"))):
    run_tag = os.path.splitext(os.path.basename(csv_path))[0]
    try:
        with open(csv_path) as f:
            rows = list(csv.DictReader(f))
        if not rows:
            continue
        for k in rows[0].keys():
            if k not in all_keys:
                all_keys.append(k)
        for r in rows:
            r["run_tag"] = run_tag
            all_rows.append(r)
    except Exception as e:
        print(f"[WARN] {csv_path}: {e}")

if "run_tag" not in all_keys:
    all_keys.append("run_tag")

if all_rows:
    out = os.path.join(output_base, "sweep_results.csv")
    with open(out, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=all_keys, extrasaction="ignore", restval="")
        writer.writeheader()
        writer.writerows(all_rows)
    print(f"Wrote {len(all_rows)} rows → {out}")
else:
    print("[WARN] No result CSVs found.")
PYEOF

echo "" | tee -a "$MASTER_LOG"
echo "================================================================" | tee -a "$MASTER_LOG"
echo " Sweep complete — $(date)" | tee -a "$MASTER_LOG"
echo " Log     : $MASTER_LOG" | tee -a "$MASTER_LOG"
echo " Results : $OUTPUT_BASE/sweep_results.csv" | tee -a "$MASTER_LOG"
echo "================================================================" | tee -a "$MASTER_LOG"
