# `viz/` — plotting code for the per-phase evidence figures

One module per build phase, all sharing one style. The figures are named `V01` to `V46`, and each one
answers a question about its phase. They are saved in `figures/phases/`, and also next to each run in `runs/<run>/figs/`.

| File | Phase | Figures |
|---|---|---|
| `style.py` | all | The single source of truth for colours, fonts and saving (`apply_style`, `save`, `panel`, `note`). Teal means classical, magenta means quantum, slate means frozen. |
| `dataset_viz.py` | 1 | V01 thread structure, V02 probe distances, V03 fact ownership, V04 corpus statistics |
| `embedding_viz.py` | 2 | V05 embedding geometry, V06 whitening spectrum, V07 projection retention, V08 cosine separation, V13 feature-block balance |
| `memory_viz.py` | 3 | V09 memory growth, V10 retrieval recall against M, V11 consolidation map, V12 entity graph and salience separation |
| `circuit_viz.py` | 4 | V14 to V16 encoding, V17 gradient variance against depth, V18 entanglement, V19 gradient agreement, V20 readout separation |
| `training_viz.py` | 6 | V21 to V25 training curves, V26 to V29 stability machinery, V31 to V33 continual-learning summary, V34 metrics table |
| `assemble_viz.py` | 7 | V41 budget composition, V42 selection waterfall, V43 and V44 saturation, V45 latency |
| `eval_viz.py` | 8 | V35 ablations, V36 and V37 regimes and stale rate, V38 noise, V39 failure taxonomy, V40 significance, V46 headline table |

The plotting rule of the project: **a figure must be regenerable from logged numbers.** If a plot needs a number, it is logged
in `metrics.jsonl`; it is not computed inside plotting code.

The figures made for the final report and presentation (the circuit, the architecture, the stability and plasticity charts)
are produced by the scripts in `scripts/figures/`, not by this package.
