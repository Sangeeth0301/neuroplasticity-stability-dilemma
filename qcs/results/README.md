# `results/` — the final experiment outputs

Everything that the final report and presentation quote comes from here. The raw files are written by
`scripts/experiments/`; the summaries by `scripts/figures/make_final_figures.py` and `make_evidence_figures.py`; the written
results file by `scripts/report/finalize_results.py`.

| Path | What it is |
|---|---|
| **`RESULTS_FINAL.md`** | **Start here.** The scorecard, the paired bootstrap, the learning-rate fairness check, the direct evidence table, and a plain list of what is and is not supported. |
| `final_summary.json` | Mean and standard deviation over seeds for each model: test MRR and rank-1, stale-above, backward transfer, stability, plasticity, combined score, Fisher-trace ratio. |
| `evidence_summary.json` | The same for the direct evidence: old facts, updated facts, facts older than 400 turns, mean over all past tasks, score on unseen tasks. |
| `raw/final/` | `final_eval.py` output, 12 files (4 models × 3 seeds). |
| `raw/evidence/` | `evidence_eval.py` output, 12 files. |
| `raw/fairness/` | `plasticity_fairness.py` output, 9 files (3 models × 3 seeds). |
| `exploration/` | The search that led to the final circuit (see its README). |

## What is inside a raw file

**`raw/final/MODEL_sSEED.json`**: `train` and `test` (ranking metrics: `mrr`, `rank1`, `recall@k`, `auc`, `stale_above_current`), `bwt`,
`final_avg`, `plasticity_index`, `trF` (Fisher trace per task), `acc_diag`, `retention_early` (MRR on the first 30 tasks at tasks
0, 29, 59, 89, 119), `reversal` (reversed-rule MRR after 0, 10, 20, 40 steps at the same checkpoints), `final_profile` (the final model on every
task), `A` (the transfer matrix), `test_per_probe` (per-thread test scores for the bootstrap).

**`raw/evidence/MODEL_sSEED.json`**: `pre` and `post` (score on each task before and after training on it), `matrix` (every past task
re-tested after every 10th task), `by_distance` (test MRR by distance band, with the last-100-turns baseline), `bands`, `split`
(`updated` against `unchanged` probes).

**`raw/fairness/MODEL_sSEED.json`**: `before`, `by_lr` (reversed-rule MRR after 40 and 80 steps for each of six learning rates), `best40`, `best80`.

All results are deterministic: re-running a script with the same model and seed reproduces its file exactly.
