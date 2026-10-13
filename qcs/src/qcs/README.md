# `qcs` — the Python package

This is the system itself. Everything in `scripts/` is a thin command-line wrapper around
the code here; everything in `tests/` checks it.

## How the pieces connect

```
datagen ──► ingest ──► encode ──► memory ──► train/dataset ──► quantum or classical scorer
 (make        (split     (text →    (store,      (candidates →      (relevance score r)
  chats)       into       1024       retrieve)     feature rows)          │
               units)     numbers)                                        ▼
                                                              train (replay, anchor) ──► eval
                                                                          │
                                                                          ▼
                                                                  assemble (prompt)
```

| Sub-package | One-line role | Build phase |
|---|---|---|
| [`datagen/`](datagen/README.md) | Generate the synthetic chat corpus with exact ground truth | 1 |
| [`ingest/`](ingest/README.md) | Load the corpus and cut turns into short "semantic units" | 1–2 |
| [`encode/`](encode/README.md) | Text → 1,024-number unit vector (frozen encoder, whitening, projection, features) | 2 |
| [`memory/`](memory/README.md) | The append-only SQLite store, entity graph, salience, retrieval | 3 |
| [`quantum/`](quantum/README.md) | The 10-qubit circuit: gates, encoding, readout, Fisher information, noise | 4 |
| [`classical/`](classical/README.md) | The classical comparison models behind the same interface | 5 |
| [`train/`](train/README.md) | Scorer dataset, replay buffer, anchor, trust region, the continual loop | 6 |
| [`assemble/`](assemble/README.md) | Token budget, knapsack selection, prompt rendering, stand-in reader | 7 |
| [`eval/`](eval/README.md) | Metrics, significance test, failure analysis, headroom | 8 |
| [`viz/`](viz/README.md) | One plotting module per phase, one shared style | all |

## Top-level modules

| File | What it does | Logic worth knowing |
|---|---|---|
| `config.py` | Typed configuration (`Config`, one dataclass per area). `Config.load("tuned")` reads `configs/base.yaml` and overlays the named files. | `validate()` enforces the rules the design depends on: `feature_dim == 2**n_qubits` (lossless amplitude encoding), `depth >= n_qubits // 2` (light-cone floor), block norm shares sum to 1, prompt reserves leave room, replay ratio < 1. Unknown YAML keys are rejected, and `hash()` gives every run a content address. |
| `paths.py` | The only place that knows where `data/`, `runs/`, `figures/` and `configs/` live. | Everything else imports `paths.ROOT`, `paths.RUNS`, and so on, so the project works from any working directory. |
| `seeding.py` | `seed_everything(seed)` and `child_rng(seed, *tags)`. | `child_rng` derives a reproducible random stream from `sha256(seed, tags)`. Python's built-in `hash()` is salted per process, so it is deliberately not used. Thread 17 can be regenerated without replaying threads 0 to 16. |
| `logging_.py` | `RunLogger` (one directory per run, one JSON line per task), `read_jsonl`, `write_jsonl`. | `METRIC_KEYS` is a whitelist: a typo in a metric name raises an error instead of silently producing an empty plot. `RunLogger.create(cfg, tag, root=...)` accepts an optional output root; the experiment scripts pass `runs/scratch` so `runs/` stays a clean record. |
| `types.py` | The records that flow through the pipeline: `Fact`, `Turn`, `SemanticUnit`, `Triple`, `ThreadPlan`. | Everything is a dataclass that converts to plain JSON, so any stage can be dumped to disk and reloaded. |

## Conventions used throughout

- **Nothing tunable is hard-coded.** Numbers live in `configs/*.yaml` (enforced by `tests/test_config.py`).
- **Frozen means frozen.** The encoder, whitening and projection cannot be refitted once set (`Whitener.fit` refuses to run twice).
- **Causality.** Retrieval only ever sees turns strictly before the question (`ANNIndex.search(mask=...)`).
- **The learning part is isolated.** Only `quantum/circuit.py` (and the classical twins in `classical/scorers.py`) have trainable parameters, and only `train/loop.py` updates them.
