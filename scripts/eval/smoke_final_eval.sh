#!/usr/bin/env bash
# =============================================================================
# smoke_final_eval.sh — tiny full-stack smoke for the final-eval launcher.
#
# This keeps the same final comparison grid as run_final_eval.sh, but clamps
# every expensive evaluation to minimal sample/step counts.  It is meant to
# verify that the planned final-run plumbing executes end-to-end:
#
#   - undefended row
#   - APRS scalar / APRS full / APRS writer-only
#   - surgical / CAST / Circuit Breakers / AlphaSteer
#   - HarmBench, XSTest, LlamaGuard
#   - Arditi, adaptive attacks, LEACE
#   - GCG, AutoDAN, Jailbroken, PAIR, ReNeLLM, SoftOpt
#   - Heretic on weight-patched configs
#   - CE loss, GSM8k/MATH500/MMLU, AlpacaEval completions
#
# CipherChat remains excluded, matching the planned headline run. AlpacaEval
# judging is skipped; completions are still generated.
#
# Environment variables:
#   PYTHON_BIN  Python command for run_obfuscation_pipeline.py.
#               Default: python3. Use "uv run python" for the uv environment.
# =============================================================================

set -euo pipefail

REPO_DIR="$(cd "$(dirname "$0")/../.." && pwd)"
MODEL_ID="${MODEL_ID:-meta-llama/Meta-Llama-3-8B-Instruct}"
OUTPUT_DIR="${OUTPUT_DIR:-$REPO_DIR/results/final_eval_smoke}"
PYTHON_BIN="${PYTHON_BIN:-python3}"

while [[ $# -gt 0 ]]; do
  case "$1" in
    --model)      MODEL_ID="$2";   shift 2 ;;
    --output_dir) OUTPUT_DIR="$2"; shift 2 ;;
    -h|--help)
      sed -n '2,45p' "$0"
      exit 0
      ;;
    *) echo "Unknown argument: $1"; exit 1 ;;
  esac
done

SMOKE_FLAGS=(
  # Final utility stack, tiny sizes.
  --ce_loss_batch_size 1
  --ce_loss_n_batches 1
  --lm_harness_tasks gsm8k,math500,mmlu
  --lm_harness_n 2
  --lm_harness_batch_size 1
  --alpacaeval_n 2
  --alpacaeval_max_new_tokens 64
  --alpacaeval_skip_judge

  # Safety evals, tiny sizes.
  --harmbench_n 2
  --gcg_n_behaviors 1
  --gcg_steps 2
  --gcg_topk 16
  --gcg_batch_size 8
  --autodan_n_behaviors 1
  --autodan_steps 2
  --autodan_population 4
  --jailbroken_n_behaviors 1
  --jailbroken_templates roleplay
  --pair_n_behaviors 1
  --pair_streams 1
  --pair_iterations 1
  --renellm_strategies 1
  --renellm_attempts 1
  --softopt_limit 1
  --softopt_steps 2

  # Nonlinear probe, minimal epochs.
  --nonlinear_probe
  --nlprobe_epochs 5

  # Keep learned baselines cheap in the final grid.
  --cb_steps 2
  --cb_batch_size 1

  # Heretic smoke sizing.
  --heretic_trials 1
  --heretic_train_samples 4
  --heretic_eval_samples 2
  --heretic_max_response_length 16
)

if [[ -n "${APRS_EXTRA_FLAGS:-}" ]]; then
  export APRS_EXTRA_FLAGS="${SMOKE_FLAGS[*]} ${APRS_EXTRA_FLAGS}"
else
  export APRS_EXTRA_FLAGS="${SMOKE_FLAGS[*]}"
fi

echo "[smoke-final] model      : $MODEL_ID"
echo "[smoke-final] output dir : $OUTPUT_DIR"
echo "[smoke-final] python     : $PYTHON_BIN"
echo "[smoke-final] extra flags: $APRS_EXTRA_FLAGS"

export PYTHON_BIN
export APRS_INCLUDE_HERETIC=1

bash "$REPO_DIR/scripts/eval/run_final_eval.sh" \
  --model "$MODEL_ID" \
  --output_dir "$OUTPUT_DIR"
