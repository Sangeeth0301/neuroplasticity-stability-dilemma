# `scripts/pipeline/` — the build, in order

Each script is one build phase. Each one writes a run folder in `runs/` (its configuration, environment, metrics and
figures) and the evidence figures `V01` to `V46` into `figures/phases/`.

| Script | Phase | What it does | Main options | Main outputs |
|---|---|---|---|---|
| `gen_dataset.py` | 1 | Generates the corpus, validates it (rules V1 to V7), draws V01 to V04. | `--config`, `--tag` | `data/raw/*.jsonl` |
| `fit_encoder.py` | 2 | Segments the training corpus, embeds it, fits whitening on held-out threads, trains the projection with InfoNCE, freezes everything, draws V05 to V08 and V13. | `--epochs 30` | `data/processed/whiten.npz`, `projection.npz` |
| `study_whitening.py` | 2 | The shrinkage sweep behind `whiten_eps = 0.1` (figure V06b). | `--tag` | `runs/.../whitening_study.json` |
| `build_memory.py` | 3 | Writes the corpus into the append-only database, builds the entity graph, salience and summaries, measures retrieval recall. | `--threads 8` | `data/processed/qcs.db`, V09 to V12 |
| `study_trainability.py` | 4 | Checks the quantum core on real data: encoding quality, gradient variance against depth, entanglement, gradient agreement. The depth is chosen from this evidence. | `--threads`, `--probes`, `--samples` | V14 to V20 |
| `tune_lr.py` | 5 | Gives every model **its own best learning rate** before any comparison, by training on a prefix of the stream and scoring on later tasks. | `--arms`, `--lrs`, `--train-tasks`, `--eval-tasks` | `configs/tuned.yaml` |
| `build_scoring_cache.py` | 6 | Builds the cached feature files used by every training and evaluation script. | `--train-threads 12`, `--test-threads 6`, `--candidates 64` | `data/processed/scoring_train.npz`, `scoring_test.npz` |
| `train_continual.py` | 5 to 6 | Trains each model through the task stream and records the full instrumentation. | `--arms`, `--probes-per-task`, `--seed`, `--rebuild` | V21 to V34, checkpoints |
| `run_assembly.py` | 7 | Budgeted prompt assembly with the stand-in reader and the budget sweep. | `--checkpoints`, `--budgets` | V41 to V45 |
| `run_eval.py` | 8 | Seeds, ablations, depolarising noise, the sealed test split, significance, failure analysis. | `--seeds`, `--arms`, `--tasks`, `--ablations-only` | V35 to V40, V46 |

Notes

- `train_continual.py` and `run_eval.py` accept the new arm names `hybrid_quantum` and `hybrid_linear` (the failed similarity-prior idea). The final model, `MultiReadoutQRE`, is run by `experiments/final_eval.py`.
- `run_eval.py` is the original single-readout evaluation. Its headline numbers are kept in `MASTER_DOC.md`; the final comparison is in `results/RESULTS_FINAL.md`.
