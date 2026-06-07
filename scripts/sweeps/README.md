# Sweep Scripts

## Primary Sweep

Use `sweep_optuna.py` for the current APRS hyperparameter search.

It sweeps:

- `epsilon`: log-uniform between CLI bounds
- `k_w`: writer rank
- `k_r`: reader rank
- `layer_budget`: number of causal layers patched

The default layer range is the full model:

```bash
--min_layer 0 --max_layer -1
```

where `-1` means the final layer.

## Objectives

`--objective multi` minimizes:

- `arditi_gap`
- `leace_gap`
- `pca8_gap`
- `utility_loss`

See `../../docs/sweep_plan.md` for the full command template and intended
NeurIPS-level sweep settings.

## Smoke Test

Before renting larger GPUs, run the three-model smoke launcher on a smaller
CUDA box:

```bash
scripts/sweeps/smoke_optuna_all.sh
```

The launcher runs `llama`, `qwen`, and `gemma` sequentially with
`sweep_optuna.py --smoke`: two tiny refusal-only trials per model, small
direction-extraction splits, small batches, and no LEACE or utility objective.
It is intended to catch model loading, tokenization, direction extraction,
APRS application, snapshot restore, CSV/SQLite output, and Arditi/PCA attack
plumbing issues. It is not a quality signal for final hyperparameters.

## Legacy Scripts

The other scripts in this directory are older targeted sweeps or ablations.
They are kept for reproducibility and reference, but new broad searches should
start from `sweep_optuna.py`.
