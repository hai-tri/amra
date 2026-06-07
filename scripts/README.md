# Scripts

This directory contains experiment entrypoints, sweep drivers, analysis helpers,
and machine setup scripts.

## Canonical Entrypoints

- `../run_obfuscation_pipeline.py`: full defense/evaluation pipeline.
- `sweeps/sweep_optuna.py`: current APRS hyperparameter sweep.
- `sweeps/smoke_optuna_all.sh`: tiny three-model smoke for the Optuna sweep.
- `eval/smoke_final_eval.sh`: tiny full-stack smoke for the final-eval grid.
- `eval/smoke_final_eval_all.sh`: tiny three-model full-stack final-eval smoke.
- `eval/smoke_heretic_all.sh`: focused three-model Heretic integration smoke.
- `eval/run_defense_utility.py`: utility-only evaluation for defenses.
- `eval/run_llama_attack_only.sh`: attack-only launcher for Llama.

## Subdirectories

- `sweeps/`: hyperparameter sweeps. `sweep_optuna.py` is the current primary
  sweep; older sweep scripts are retained for reference.
- `eval/`: utility, validation, smoke, and attack-only evaluation scripts.
- `analysis/`: plotting and matrix-analysis utilities.
- `tpu/`: TPU/GPU launcher utilities and parity checks.
- `setup/`: environment setup scripts for different machines.

## Notes

Raw outputs should go under `results/`, which is ignored by git. Curated
paper-facing exports should go under `paper/results/` and `paper/figures/`.
