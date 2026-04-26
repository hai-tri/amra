# APRS — NeurIPS Experiment Plan

Plan of record for the final sweep and paper. Update this file as decisions change.

---

## 1. Compute

- **2× H100/A100 80GB** in parallel for paper results (Lambda / Modal / RunPod, ~$2.50/hr each).
- **A10 is smoke-only.** Use it to catch integration failures before renting the expensive boxes; do not use A10 runs as paper numbers.
- Model pairing: box A = (Llama-3-8B, Qwen3-8B), box B = (Gemma-2-9B, Mistral-7B-v0.3).
- Wall clock: ~48h. **Revised budget: ~$420** (see §10).

## 2. Models

Pin exact Hugging Face IDs in `scripts/setup_h100.sh` and `scripts/run_final_eval.sh` for reproducibility.

| Model | HF ID | Arch notes |
|---|---|---|
| Llama-3-8B-Instruct | `meta-llama/Meta-Llama-3-8B-Instruct` | Primary baseline; matches smoke/final script defaults |
| Qwen3-8B | `Qwen/Qwen3-8B` | 2025 release; works out of the box |
| Gemma-2-9B-it | `google/gemma-2-9b-it` | Requires `pre_feedforward_layernorm` dispatch (done) |
| Mistral-7B-Instruct-v0.3 | `mistralai/Mistral-7B-Instruct-v0.3` | Different family |

Alternative for Gemma slot: `google/gemma-3-9b-it` (also supported via the LN dispatch fix).

## 3. Defense configs (per model)

9 result rows. `run_final_eval.sh` now runs the undefended baseline first, which also generates the cached refusal direction; defended configs can reuse it via `--skip_direction_extraction`.

```
1. undefended                                                             (baseline)
2. APRS full ε=0.025     --projection_mode full --epsilon 0.025 --per_layer_direction --writer_output_directions
3. APRS hadamard ε=0.3   --projection_mode hadamard --epsilon 0.3 --per_layer_direction --writer_output_directions
4. APRS scalar ε=0.3     --projection_mode scalar_projection --epsilon 0.3 --per_layer_direction --writer_output_directions
5. APRS full ε=0.025 writer-only  (ablation: shows reader patches matter)
6. surgical              --defense_type surgical
7. cast                  --defense_type cast
8. circuit_breakers      --defense_type circuit_breakers
9. alphasteer            --defense_type alphasteer
```

APRS writer patches now use **per-writer output refusal directions**: each
`W_O^l` is patched with a harmful-minus-harmless direction measured at that
layer's attention output, and each `W_down^l` is patched with the corresponding
MLP-output direction. The older block-input residual directions remain as the
fallback and as the source for causal layer selection.

The CSV also records writer-output direction leakage diagnostics:
`writer_attn_avg_cos_sim`, `writer_mlp_avg_cos_sim`, and
`writer_output_avg_cos_sim` compare post-defense writer-output DIM directions
against the pre-defense writer-output directions on validation prompts.

## 4. Attacks

Fixed flag set passed to every defended run:

```
--llamaguard
--gcg            --gcg_n_behaviors 25 --gcg_steps 500
--autodan        --autodan_n_behaviors 25
--pair           --pair_n_behaviors 25
--renellm
--softopt        --softopt_limit 25
```

**CipherChat** is dropped from the main sweep — text-transform jailbreaks are orthogonal to an activation-steering defense's mechanism, and the five attacks above already span white-box (GCG, SoftOpt), genetic (AutoDAN), LLM-driven (PAIR), and template-rewriting (ReNeLLM) categories. Re-enable with `--cipherchat` if budget allows.

## 5. Utility suite

Ratcheted up from the smoke defaults for the final tables.

| Benchmark | Sample size | Metric |
|---|---|---|
| Pile perplexity | 1024 sequences (≈260k tokens) | PPL + **BPB** |
| Alpaca perplexity | 1024 sequences | PPL + **BPB** |
| GSM8k | 100 | exact_match |
| MATH500 | 500 (full set) | exact_match |
| MMLU | 500 | accuracy |
| XSTest | 250 safe prompts (full) | over-refusal rate |
| AlpacaEval | 805 (full) | length-controlled win rate (GPT-4 judge) |

**Set `OPENAI_API_KEY`** on the H100 box before launch — without it AlpacaEval skips judging and only saves completions.

Pipeline flag changes vs current defaults:
```
--ce_loss_n_batches 256
--lm_harness_n 500
--alpacaeval_n 805
```

## 6. Statistics

- **Single seed (42)** across the full 9-row × 4-model grid.
- **Optional if time remains:** three seeds (42, 123, 2024) on the headline row only: APRS hadamard ε=0.3. Launch as separate output dirs rather than blocking the main sweep.

## 7. Pre-launch checklist (before renting H100s)

- [x] `scripts/setup_h100.sh` — idempotent bootstrap: torch, requirements, HF weight prefetch (4 models × ~16 GB + HarmBench judge + LlamaGuard), HarmBench clone.
- [x] `scripts/run_final_eval.sh` — takes `--model`, iterates configs, resumes from cached CSVs, forwards attack flags, includes undefended row, writes per-config artifact dirs, uses final utility sample sizes, exits nonzero on config failure, and omits CipherChat/Heretic by default.
- [x] `scripts/make_tables.py` — pivots `all_results.csv` to LaTeX + markdown with BPB columns and writer-only handling.
- [x] Decoder bug fixed: HarmBench, XSTest, and AlpacaEval decode with the explicit tokenizer.
- [x] Hook-based baseline bug fixed: integrity, abliteration, adaptive, LEACE, and SoftOpt receive defense hooks.
- [x] APRS writer-output direction extraction added: `--writer_output_directions`.
- [ ] Commit all outstanding local changes (`run_obfuscation_pipeline.py`, `apply_obfuscation.py`, `obfuscation_utils.py`, `obfuscation_config.py`, `evaluate_loss.py`, `evaluate_alpacaeval.py`, `scripts/smoke_run.sh`, `PLAN.md`).
- [ ] `lm_eval` version pinned in `requirements.txt`.
- [ ] Verify `scripts/smoke_run.sh` passes on a cheap A10 (~20–45 min realistic) using Llama-3-8B.
- [ ] **Gemma smoke gate** — a second smoke run on a cheap Gemma-2-9B container to validate `pre_feedforward_layernorm` dispatch + empirical pollution probe on real weights before burning H100 time.

## 8. Execution order

1. **Today (no GPU)**: finish syntax/static checks, commit, and run table renderer on existing CSVs.
2. **Cheap GPU (~$1–$2)**: Llama-3-8B smoke via `scripts/smoke_run.sh`.
3. **Cheap GPU (~$2)**: Gemma-2-9B smoke (architecture-specific gate).
4. **H100 × 2 for ~2 days**: launch the full sweep via `run_final_eval.sh`.
5. **Local post-run**: rsync CSVs back, run `make_tables.py`, generate figures, write-up.

## 9. Failure-mode watchlist

| Risk | Mitigation |
|---|---|
| GCG OOM on 8B at batch=256 | Drop `--gcg_topk` to 128 |
| PAIR attacker context overflow | Clamp attacker history |
| HarmBench judge GPU contention | Load judge on separate device when possible |
| lm-eval version mismatch | Pin in `requirements.txt` |
| Gemma LN misrouting on real weights | Covered by the Gemma smoke gate (§7) |
| AlpacaEval judge API failure mid-run | Pipeline falls back to completions-only; re-judge offline |
| Writer-output directions underperform residual directions | Run one Llama residual-direction fallback row by omitting `--writer_output_directions` if smoke ASR/utility looks off |

## 10. Revised budget

| Item | Cost |
|---|---|
| Base 9-row × 4-model sweep, single seed | ~$270 |
| Multi-seed on headline config (3 seeds × 4 models) | ~$90 |
| Bumped utility sample sizes (PPL, MMLU, MATH, AlpacaEval) | ~$80 |
| AlpacaEval GPT-4 judging (4 models × 805) | ~$8 |
| Smoke tests (A10 Llama + A10 Gemma) | ~$3 |
| **Total** | **~$451** |

## 11. Tables

`scripts/make_tables.py` pivots `all_results.csv` into LaTeX + markdown. Each
table has one job — don't cram multiple stories into one.

### Main text (3 tables)

**Table 1 — ε sweep** *(§4 Experimental Setup, near the hyperparameter-selection paragraph)*
Narrow, 4 columns. Single model (Llama-3-8B). Shows the Pareto knee that
justifies ε=0.025.

| ε | Pile BPB↓ | MMLU↑ | Arditi ASR↓ |

ε values: {0.01, 0.025, 0.05, 0.1, 0.15, 0.2, 0.3} + undefended row.
Source: `scripts/run_sweep.sh` Sweep 1.

**Table 2 — Main utility** *(§5 Results)*
Wide. Utility preservation at the chosen ε=0.025, across all 4 models × all
defenses × full benchmark suite.

| Model × Defense | Pile BPB↓ | Alpaca BPB↓ | GSM8k↑ | MATH500↑ | MMLU↑ | AlpacaEval LC-WR↑ | XSTest over-refusal↓ |

Rows: 4 models × {undefended, APRS (full ε=0.025), APRS (hadamard ε=0.3), APRS
(scalar ε=0.3), APRS writer-only, surgical, CAST, circuit_breakers, alphasteer}.
Source: `run_final_eval.sh`.

**Table 3 — Main security (headline ASR)** *(§5 Results)*
Wide. Attack resistance across all 4 models × all defenses × all attacks.

| Model × Defense | Arditi↓ | PCA↓ | per-layer↓ | LEACE↓ | GCG↓ | AutoDAN↓ | PAIR↓ | ReNeLLM↓ | SoftOpt↓ |

Same rows as Table 2. This is the paper's single most-read artifact.
Source: `run_final_eval.sh`.

### Appendix tables

| Table | What it shows |
|---|---|
| Calibration-set-size sweep | `num_calibration_prompts` ∈ {32, 64, 128, 256, 512} × 3 modes. From Sweep 2. |
| Per-layer vs global refusal direction | ε=0.1 anchor × 3 modes. From Sweep 3. |
| Writer-only ablation | Same as APRS full ε=0.025 but with `--obfuscation_writer_only`. Demonstrates reader patches are necessary. |
| LlamaGuard per-category ASR breakdown | Fine-grained safety breakdown if LlamaGuard results warrant it. |

## 12. Figures

All figures read from the same source-of-truth `all_results.csv` emitted by the
sweep. `scripts/make_figures.py` should render them in one pass so no figure
ever disagrees with a table number.

### Main text (7 figures)

| # | Figure | What it shows | Source |
|---|---|---|---|
| 1 | Method schematic | Writer patch (W_O / W_down injects alias) + reader patch (Q/K/V/gate/up compensates) in a transformer block. | Hand-drawn / TikZ |
| 2 | Residual-stream drift under the patch | 1×3 panel: L2 / variance / cosine vs layer index for an example writer patch. Complements the ε-sweep table. | `scripts/plot_drift_combined.py` (already exists) |
| 3 | Main ASR grid (headline) | Grouped bars: attack × defense × model. Headline comparison. | `all_results.csv` |
| 4 | Security–utility Pareto | Scatter: utility retention (BPB ratio or AlpacaEval LC-win-rate) vs attack resistance. One marker per (defense, model, ε). | `all_results.csv` |
| 5 | Per-architecture defense generalization | Horizontal bars: Arditi ASR × 4 models × {undefended, APRS, top baseline}. Shows "this is not a Llama-specific trick." | `all_results.csv` |
| 6 | Refusal-direction recoverability | Cosine similarity between attacker-recovered refusal direction (diff-in-means on the *defended* model) and true pre-defense direction, across layers. APRS-specific money plot. | Post-hoc probe on saved defended models |
| 7 | Writer-output direction recoverability | Cosine similarity between defended `W_O` / `W_down` output DIM directions and their pre-defense counterparts. Directly validates the per-writer direction patch. | `writer_output_cosine.json` + `all_results.csv` |

### Appendix / supplementary

| Figure | What it shows | Why |
|---|---|---|
| Per-layer ablation scores | Pertinent layer selection | Justifies §2 Layer Selection |
| Null-space overlap heatmap | Readers' null spaces across layers overlap near chance | Justifies why per-layer reader patches are independent (`results/reader_nullspace_overlap.csv` already exists) |
| Per-benchmark utility bars (full grid) | Defended vs undefended on every benchmark × model | Detail behind Fig 4 |
| XSTest over-refusal examples | Qualitative: defense doesn't break safe prompts | Reviewer "does it over-refuse?" check |
| AlpacaEval head-to-head examples | 3–4 side-by-side completions | Humanizes the generation-quality numbers |

### Skip

- Negative-space figures (defended ≈ undefended). Use a table.
- Per-attack × per-model × per-defense line plots. Redundant with Fig 3.
- LlamaGuard confusion matrices. Use a table.

## 13. Open questions

- Gemma-2 vs Gemma-3 for the Gemma slot. Default to Gemma-2-9B (more established baseline, same LN structure).
- Whether to include CipherChat results as a supplementary table even if excluded from the main sweep. Cheap to run post-hoc once weights are cached.

---

*Last updated: 2026-04-24.
Major code changes this week: explicit tokenizer decoding, hook-aware baseline evaluation, empirical pollution tracking, Gemma `pre_feedforward_layernorm` dispatch, writer-output refusal directions, AlpacaEval integration, bits-per-byte in loss_evals, smoke/final scripts, and table rendering.*
