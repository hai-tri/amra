#!/usr/bin/env bash
# =============================================================================
# run_utility_benchmarks.sh — Utility benchmarks on best APRS configs
#
# Runs GSM8k, MATH500, MMLU (n=500 each) and AlpacaEval generation (805
# prompts, no judge) on the three best-identified configs:
#
#   Llama-3-8B-Instruct  ε=0.025  forced_layers=10
#   Qwen3-8B             ε=0.05   forced_layers=7 (auto)
#   Gemma-2-9B-it        ε=0.01   forced_layers=10
#
# Direction artifacts are reused from the prior epsilon sweep
# (--skip_direction_extraction). Attacks and HarmBench are skipped.
#
# Usage:
#   bash scripts/run_utility_benchmarks.sh [--output_dir DIR]
# =============================================================================

set -euo pipefail

REPO_DIR="$(cd "$(dirname "$0")/../.." && pwd)"
OUTPUT_BASE="$HOME/aprs_utility"
SWEEP_DIR="$HOME/aprs_sweep"   # existing artifact cache from epsilon sweep

while [[ $# -gt 0 ]]; do
    case "$1" in
        --output_dir) OUTPUT_BASE="$2"; shift 2 ;;
        --sweep_dir)  SWEEP_DIR="$2";  shift 2 ;;
        *) echo "Unknown argument: $1"; exit 1 ;;
    esac
done

mkdir -p "$OUTPUT_BASE"
TIMESTAMP=$(date +%Y%m%d_%H%M%S)
MASTER_LOG="$OUTPUT_BASE/utility_${TIMESTAMP}.log"

echo "================================================================" | tee "$MASTER_LOG"
echo " APRS Utility Benchmarks — $(date)"                              | tee -a "$MASTER_LOG"
echo " Output dir  : $OUTPUT_BASE"                                     | tee -a "$MASTER_LOG"
echo " Artifact dir: $SWEEP_DIR/artifacts"                             | tee -a "$MASTER_LOG"
echo "================================================================" | tee -a "$MASTER_LOG"

_SKIP_FLAGS=(
    --skip_harmbench
    --skip_xstest
    --skip_heretic
    --alpacaeval_skip_judge
)

_BENCH_FLAGS=(
    --lm_harness_n 500
    --alpacaeval_n 805
)

run_config() {
    local tag="$1"; shift
    local model_id="$1"; shift
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
        --artifact_dir "$SWEEP_DIR/artifacts" \
        --skip_direction_extraction \
        --defense_type obfuscation \
        --projection_mode full \
        --num_calibration_prompts 128 \
        --per_layer_direction \
        --writer_output_directions \
        --seed 42 \
        "${_SKIP_FLAGS[@]}" \
        "${_BENCH_FLAGS[@]}" \
        "${extra_args[@]}" \
        2>&1 | tee "$log_path"
    local rc=$?
    set -e

    if [[ $rc -eq 0 ]]; then
        echo "   [OK] → $csv_path" | tee -a "$MASTER_LOG"
    else
        echo "   [WARN] exited $rc — see $log_path" | tee -a "$MASTER_LOG"
    fi
}

# Llama — forced 10 layers at ε=0.025
run_config "utility_llama_eps0_025_layers10" \
    "meta-llama/Meta-Llama-3-8B-Instruct" \
    --epsilon 0.025 \
    --num_pertinent_layers 10

# Qwen — auto layers (7) at ε=0.05
run_config "utility_qwen_eps0_05_layers7" \
    "Qwen/Qwen3-8B" \
    --epsilon 0.05

# Gemma — forced 10 layers at ε=0.01
run_config "utility_gemma_eps0_01_layers10" \
    "google/gemma-2-9b-it" \
    --epsilon 0.01 \
    --num_pertinent_layers 10

echo "" | tee -a "$MASTER_LOG"
echo "── Aggregating results ──────────────────────────────────────" | tee -a "$MASTER_LOG"

OUTPUT_BASE="$OUTPUT_BASE" python3 - <<'PYEOF' 2>&1 | tee -a "$MASTER_LOG"
import csv, glob, os

output_base = os.environ["OUTPUT_BASE"]
all_rows, all_keys = [], []

for csv_path in sorted(glob.glob(os.path.join(output_base, "utility_*.csv"))):
    if os.path.basename(csv_path) == "utility_results.csv":
        continue
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
    out = os.path.join(output_base, "utility_results.csv")
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
echo " Done — $(date)" | tee -a "$MASTER_LOG"
echo " Log     : $MASTER_LOG" | tee -a "$MASTER_LOG"
echo " Results : $OUTPUT_BASE/utility_results.csv" | tee -a "$MASTER_LOG"
echo "================================================================" | tee -a "$MASTER_LOG"
