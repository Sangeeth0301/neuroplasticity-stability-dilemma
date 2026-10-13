# `figures/`

| Folder | What is in it | Made by |
|---|---|---|
| **`final/`** | **The figures of the final report and presentation:** `architecture`, `circuit` (the complete circuit), `forgetting_matrix`, `memory_distance`, `learning_new`, `plasticity`, `dilemma` (stability against plasticity), `fairness`. | `scripts/figures/make_slide_diagrams.py`, `make_slide_figures.py` |
| `analysis/` | Analysis versions with more detail: `FIN1` stability, `FIN2` plasticity, `FIN3` combined, `FIN4` scorecard, `FIN5` fairness, `EV1` to `EV3` evidence. | `make_final_figures.py`, `make_evidence_figures.py`, `report/finalize_results.py` |
| `branding/` | The Amrita logo (SVG as downloaded from amrita.edu, and a PNG render) used on the report cover. | n/a |
| `architecture/`, `circuit/`, `results/` | Version 1 figures of the **original single-readout circuit** (architecture diagrams, the circuit with its light cone, a PennyLane rendering of the real model, result figures RES1 to RES6). | `make_hero_diagram.py`, `make_diagrams.py`, `make_overall_circuit.py`, `make_circuit_figure.py`, `make_vqc_canonical.py`, `make_result_figures.py` |
| `phases/` | The per-phase evidence figures `V01` to `V46` of the build. | the phase scripts in `scripts/pipeline/` |

For the circuit of the final model use `final/circuit.png`. The `circuit/` and `architecture/` folders show the original design that
measured one qubit.
