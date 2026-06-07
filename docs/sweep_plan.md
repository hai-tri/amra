# APRS Optuna Sweep Plan

## Goal

Find APRS hyperparameters that preserve utility while retaining refusal after
mechanistic attacks.

## Models

- Llama
- Gemma
- Qwen

## Search Space

- `epsilon`: log-uniform from `0.001` to `0.75`
- `k_w`: `{1, 2, 4, 8, 16, 32, 64}`
- `k_r`: `{1, 2, 4, 8, 16, 32, 64}`
- `layer_budget`: `{4, 8, 12, 16, 20, 24, 30, 32}`

## Fixed Layer Bounds

- `min_layer`: default `0`
- `max_layer`: default `-1`, meaning the model's final layer

By default, each trial can select from the full model depth. To restrict the
eligible layer range for a targeted ablation, pass explicit CLI values such as
`--min_layer 8 --max_layer 24`.

## Layer Selection

For each trial, select the top causal layers within the fixed eligible layer
range. Invalid layer budgets are pruned when the requested `layer_budget`
exceeds the number of eligible layers.

## Rank Constraints

`k_w` and `k_r` may go up to `64`. Trials are pruned if either rank exceeds the
number of calibration prompts. For `k_r`, the script increases the probe prompt
count up to `k_r`, capped by `num_calibration_prompts`.

## Objectives

Multi-objective Optuna minimizes:

- `arditi_gap = clean baseline refusal - defended post-Arditi refusal`
- `leace_gap = clean baseline refusal - defended post-LEACE refusal`
- `pca8_gap = undefended PCA-8 refusal - defended PCA-8 refusal`
- `utility_loss = normalized BPB increase + normalized MMLU drop + normalized MATH500 drop`

## Stage 1 Command Template

Run one command per model:

```bash
python scripts/sweeps/sweep_optuna.py \
  --model llama \
  --n_trials 100 \
  --objective multi \
  --sampler tpe \
  --epsilon_min 0.001 \
  --epsilon_max 0.75 \
  --min_layer 0 \
  --max_layer -1 \
  --n 20 \
  --num_calibration_prompts 64 \
  --attack_batch_size 64 \
  --forward_batch_size 64 \
  --bpb_batches 8 \
  --mmlu_n 100 \
  --math500_n 100 \
  --utility_batch_size 8 \
  --output_dir results/optuna_sweep_v2/llama
```

Repeat with `--model qwen` and `--model gemma`, changing `--output_dir`
accordingly.

## Smoke Test Command

Before launching the full hyperparameter sweep, run all planned models on a
smaller CUDA GPU:

```bash
scripts/sweeps/smoke_optuna_all.sh
```

This executes `llama`, `qwen`, and `gemma` sequentially with `--smoke`. Smoke
mode uses tiny data splits, two refusal-only trials per model, small batches,
and skips LEACE / utility objectives. Passing this smoke test means the sweep
can load each model, extract a direction, apply APRS, restore snapshots, write
CSV/SQLite artifacts, and run the lightweight Arditi/PCA checks. It does not
validate hyperparameter quality.

## Notes

- The run is seeded with `--seed 42` by default.
- GSM8K is not part of the current sweep objective.
- MMLU `n=100` and MATH500 `n=100` are cheap-search proxies, not final paper
  numbers.
- Final configs should be validated later with larger utility samples and direct
  safety metrics.
