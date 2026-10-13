# `classical/` — the models the circuit is compared with

Every scorer has the same interface as the circuit: it takes a batch of 1,024-number unit vectors and
returns one relevance score in [0, 1] per row. Because the interface is identical, a comparison between
two models changes exactly one component and nothing else in the pipeline.

| Class (in `scorers.py`) | Parameters | What it is |
|---|---|---|
| `LinearScorer` | 1,025 | One linear layer plus sigmoid. The closest classical match to the circuit's size (1,080 angles). |
| `MLPScorer` | 65,665 | 1,024 → 64 → 1 with ReLU. A larger capacity reference, always reported as such. |
| `HybridLinearScorer` | 1,026 | The control for the failed "similarity prior" idea: the same fixed similarity term plus a linear residual. |
| `CosineScorer` | 0 | No learning. Sums the interaction block of the feature vector, which is the cosine similarity. |
| `RecencyScorer` | 0 | No learning. "Just use the most recent turns", the truncation floor. |
| `RandomScorer`, `OracleScorer` | 0 | The chance floor and the ceiling (knows the labels). |

## Logic worth knowing

- `BaseScorer` gives all models the same helper methods (`score`, `theta_norm`, `grad_norm`, `wrap_angles`).
  `wrap_angles` does **nothing** for the classical models: their weights are free to grow without limit, and that
  difference is the experiment.
- Each learning model exposes its weights as `theta`, so the replay, anchor and Fisher code in `train/` works on
  every model without special cases.
- The cosine scorer uses a sigmoid gain of 6; a gain of 20 saturated to exactly 1.0 for most candidates and produced
  ties that made AUC meaningless.
- `build_scorer(name, cfg, seed)` is the factory. The circuit models are built separately (they need the full config).

## Tests

`tests/test_train_assemble.py`: every scorer returns probabilities; the linear model is parameter-matched to the circuit;
classical models do not wrap angles.
