# Paper Assets

This directory contains curated assets for paper drafts.

## Structure

- `results/`: paper-facing CSV, Markdown, and LaTeX tables.
- `figures/`: selected figures suitable for drafts.

Raw experiment artifacts remain under `../results/` and are ignored by git.
Do not treat raw `results/` directories as reviewer-facing outputs.

## Current Notes

- The main curated safety table focuses on Arditi-style abliteration rather
  than LEACE, though LEACE values remain available in `results/robustness.csv`.
- Qwen token-level refusal scores have known saturation issues in the current
  raw tables.
- AlpacaEval completions were generated in raw artifacts, but judged scores were
  not available in the curated tables.
- XSTest was not run for the current curated tables.

