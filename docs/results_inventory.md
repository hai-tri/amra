# Results Inventory

This document lists the result files that are currently easiest to interpret.

## Curated Paper Tables

- `paper/results/utility.csv`
- `paper/results/direct_safety.csv`
- `paper/results/robustness.csv`
- `paper/results/all_metrics.csv`
- `paper/results/tables_main.tex`
- `paper/results/safety_robustness_merged.tex`

## Curated Figures

- `paper/figures/drift_eps_sweep_combined.png`
- `paper/figures/llama_pareto_post_arditi.pdf`
- `paper/figures/llama_pareto_post_arditi.png`

## Raw Source Locations

- `results/paper_final/`: source for current curated paper tables.
- `results/optuna_sweep/`: older Optuna/Pareto outputs.
- `results/optuna_sweep_v2/`: intended output root for the updated sweep.

## Caveats

- Qwen token-level refusal scores saturate in the current tables; use behavioral
  safety metrics for Qwen.
- AlpacaEval score columns are blank because judging did not complete.
- XSTest is absent from the current curated result set.
- LEACE is retained in robustness CSVs but excluded from the current main
  merged table to keep the main paper framing focused on abliteration.

