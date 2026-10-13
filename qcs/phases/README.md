# `phases/` — the ten build phases

One folder per phase. Each `objective.md` states the objective, the scripts that run it, the exit criterion (which has to pass before
the next phase starts) and the measured result. The narrative of each phase is in `../MASTER_DOC.md`.

| Phase | Folder | Exit criterion in one line | Status |
|---|---|---|---|
| 0 | `00_foundations` | Reproducible substrate: one config, fixed seeds, run logger | Done |
| 1 | `01_dataset_generation` | A corpus with exact ground truth; oracle 1.0, random near chance | Done |
| 2 | `02_encoding` | Text becomes a unit vector of 1,024 numbers; recall preserved | Done |
| 3 | `03_memory_manifold` | Append-only memory enforced by the database; retrieval recall ceiling measured | Done |
| 4 | `04_quantum_core` | Verified circuit: simulator matches PennyLane; gradients agree; depth chosen by evidence | Done |
| 5 | `05_baselines` | Each model at its own best learning rate | Done |
| 6 | `06_continual_training` | 120 tasks with full instrumentation | Done |
| 7 | `07_assembly_generation` | Prompt never exceeds the budget; no unlabelled stale fact served | Done |
| 8 | `08_evaluation` | Seeds, ablations, noise, sealed split, significance | Done |
| 9 | `09_hardening` | Long run, hardware parity, real language model, one-command reproduction | **Not started** |

The final experiments of this project (the 10-qubit readout and the stability/plasticity measurements) were done after phase 8 and
are described in `../results/RESULTS_FINAL.md`.
