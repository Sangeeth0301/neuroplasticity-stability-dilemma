# `eval/` — how results are measured

All metrics are defined once and computed identically for every model.

| File | What it does | Logic worth knowing |
|---|---|---|
| `metrics.py` | `evaluate_probes(score_fn, examples)` and the individual metrics. | Everything is computed **per probe and then averaged over probes**, never pooled over candidates, so probes with many candidates cannot dominate. MRR = 1 / rank of the first correct candidate. `auc` handles ties by averaging ranks (a saturated scorer produces many exact ties). `stale_above_current` is 1 for a probe when a superseded fact is ranked above the current one. `persona_bleed` counts top candidates written by someone other than the fact's owner. |
| `significance.py` | `paired_bootstrap`, `per_thread_metric`. | The bootstrap resamples **threads**, not probes. Probes in one thread share facts, participants and templates, so resampling probes would shrink the interval and invent significance. Returns the mean difference, a 95% interval and a two-sided p-value. |
| `failure_analysis.py` | `analyse` / `categorise`. | Sorts every wrong answer into the first applicable layer: `retrieval` (the fact was not even a candidate), `ranking` (present but not first), `stale` (an outdated version outranked it), `eviction` (ranked well but the knapsack dropped it). This points the next fix at the layer that is actually failing. |
| `headroom.py` | `cosine_headroom`. | Measures what plain cosine similarity cannot do (by regime, by distance band, and on update cases), which is the room a learned scorer has to earn. |

## Tests

`tests/test_train_assemble.py`: ranking metrics on hand-computed cases, AUC with ties, stale-above-current detection, averaging over probes.
