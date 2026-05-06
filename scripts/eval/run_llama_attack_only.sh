#!/usr/bin/env bash
set -euo pipefail

# Run low-GPU-utilization attack loops separately from the GH200 direct-safety pass.
# Intended for a cheaper GPU box. Produces a CSV whose attack columns can be
# merged with the direct-safety CSV by (model, defense_type, config).

RUN_DIR="${1:-results/llama_attack_only_$(date +%Y%m%d_%H%M%S)}"
mkdir -p "$RUN_DIR/logs"
CSV="$RUN_DIR/all_results.csv"

COMMON=(
  --model_path meta-llama/Meta-Llama-3-8B-Instruct
  --save_csv "$CSV"
  --skip_direction_extraction
  --gcg --gcg_n_behaviors 16 --gcg_steps 200 --gcg_batch_size 512
  --autodan --autodan_n_behaviors 16
  --pair --pair_n_behaviors 16
  --renellm
  --softopt --softopt_limit 16
  --skip_harmbench
  --skip_integrity_eval
  --skip_adaptive_attacks
  --skip_xstest
  --skip_lm_harness
  --skip_alpacaeval
  --skip_heretic
  --ce_loss_n_batches 0
)

run_row() {
  local tag="$1"; shift
  echo "[llama-attack-only] start $tag $(date -Is)"
  python3 run_obfuscation_pipeline.py "${COMMON[@]}" --artifact_subdir "attack_only_${tag}" "$@" \
    > "$RUN_DIR/logs/llama_${tag}.log" 2>&1
  local rc=$?
  echo "[llama-attack-only] done $tag rc=$rc $(date -Is)"
  return "$rc"
}

run_row none --defense_type none
run_row aprs --defense_type obfuscation --projection_mode full --epsilon 0.025 --num_pertinent_layers 20 --per_layer_direction --writer_output_directions --num_writer_directions 1 --num_reader_directions 8
run_row surgical --defense_type surgical
run_row cast --defense_type cast
run_row circuit_breakers --defense_type circuit_breakers
run_row alphasteer --defense_type alphasteer

echo "[llama-attack-only] PASS $(date -Is)"
