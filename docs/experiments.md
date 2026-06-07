# Experiment Organization

Raw experiment outputs live under `results/` and are treated as uncurated
artifacts. Paper-facing exports live under `paper/results/` and
`paper/figures/`.

## Current Curated Outputs

- `paper/results/`: tables for utility, direct safety, and robustness.
- `paper/figures/`: selected figures suitable for paper drafts.
- `results/optuna_sweep_v2/SWEEP_PLAN.txt`: current sweep-only plan.

## Current Sweep

The main sweep entrypoint is:

```bash
python scripts/sweeps/sweep_optuna.py
```

It sweeps epsilon, writer rank, reader rank, and layer budget. It optimizes
Arditi, PCA-8, LEACE, and utility-loss objectives.

## Notes

- Raw `results/` directories are not deleted by cleanup.
- Qwen token-level refusal scores have known saturation issues in the current
  paper tables.
- AlpacaEval completions exist in raw results, but judged scores were not
  available in the curated tables.
- XSTest was not run for the current curated tables.

