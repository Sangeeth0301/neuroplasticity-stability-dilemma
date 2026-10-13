# `configs/` — every tunable number

No algorithm in `src/` contains a tunable constant. They all come from here (enforced by `tests/test_config.py`).

- **`base.yaml`**: the defaults.
- **`tuned.yaml`**: an overlay that holds the per-model learning rates found by `scripts/pipeline/tune_lr.py`.
  `Config.load("tuned")` reads `base.yaml` first and then applies `tuned.yaml`.

| Section | Key settings (value) | Meaning |
|---|---|---|
| `seed` | 7 | Master random seed. |
| `data` | 24 train and 6 test threads; 600 turns; 5 users; 60 facts per thread; probe distances 10, 50, 200, 500; filler 0.55; distractor 0.25; `supersede_ratio` 0.35 | Corpus generation. The update ratio applies to adversarial threads only. |
| `encode` | MiniLM (384); `proj_dim` 256; `whiten_components` 256; `whiten_eps` 0.1; `feature_dim` 1024; block norm targets 0.35, 0.35, 0.20, 0.10 | The frozen encoding stack. `feature_dim` must equal 2^n_qubits. |
| `memory` | `candidate_M` 256; `consolidate_every` 50; `salience_alpha` 0.05 | Retrieval size, summary window, salience smoothing. (Training uses 64 candidates, set when the cache is built.) |
| `quantum` | 10 qubits; `depth` 8; brickwall; gate form `exp`; readout qubit 9; `local_z`; `beta_init` 4; `init_scale` 0.05 | The circuit. `depth` must be at least n_qubits ÷ 2. |
| `train` | 3 epochs per task; batch 32; replay capacity 8,000 and ratio 0.5; `lambda_omega` 0.01; `trust_region_delta` 0.35; `drift_threshold` 0.25; 8 negatives per probe; `lr_by_arm` | The continual loop. |
| `assemble` | 3,072-token budget; reserves 256 (system), 256 (pinned), 768 (recent); redundancy 0.35; value prior fact 1.0, summary 0.7, filler 0.35 | Prompt assembly. |

`tuned.yaml` currently sets the learning rates: circuit 0.02, linear 0.1, MLP 0.02, and the same values for the
two hybrid models of the discarded similarity-prior experiment.

A run's configuration is saved with it (`runs/<run>/config.yaml`) and its content hash is part of the run folder name.
