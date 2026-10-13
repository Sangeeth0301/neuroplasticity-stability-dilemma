# `tests/` — 88 automated tests

Run from `qcs/`:

```
python -m pytest
```

The tests do not check style. Each one guards a property that the design depends on, so that a regression
shows up as a failing test and not as a quietly wrong result.

| File | Tests | What it protects |
|---|---|---|
| `test_config.py` | 7 | The configuration loads and validates; unknown keys are rejected; the hash is content-addressed; the light-cone depth floor and `feature_dim == 2**n_qubits` are enforced; the seeded random streams are stable across processes. |
| `test_datagen.py` | 8 | The generated corpus is valid; an oracle scores 1.0 and random scores near chance; every probe is answerable; no probe targets a superseded fact; adversarial threads really contain updates; regeneration is deterministic; probe distance bands are balanced; the validator catches a deliberately broken corpus. |
| `test_encode_shapes.py` | 13 | Segmentation is lossless and deterministic; the feature vector is 1,024-dimensional and unit length; block norm shares hit their targets (and unbalanced blocks would collapse participation); relational features are bounded; recency is monotone and saturating; the whitener refuses to refit and flattens the spectrum. |
| `test_memory.py` | 11 | The database rejects `UPDATE` and `DELETE`; rewriting a unit changes nothing; consolidation only adds and is idempotent; retrieval never returns a future candidate; the graph marks superseded facts; graph features are bounded; salience separates users. |
| `test_quantum.py` | 23 | Gates are unitary and equal the identity at zero angles; norm is preserved at every layer; amplitude encoding is lossless and rejects non-states; parameter count is 1,080; light-cone depth matches the config floor; entanglement grows with depth; angle wrapping bounds the norm; autodiff and parameter-shift agree for the product form and disagree for the exponential form; gradients vanish below the light cone; the simulator matches PennyLane. |
| `test_train_assemble.py` | 26 | Every scorer returns probabilities; the linear model is size-matched to the circuit; classical models do not wrap angles; ranking metrics on hand-computed cases; replay is uniform over history; the anchor, trust region and drift detector behave as specified; the knapsack respects the budget and suppresses near-duplicates; the reader ignores outdated facts. |

Tests use small synthetic inputs and temporary folders.
