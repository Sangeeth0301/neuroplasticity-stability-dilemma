# `scripts/figures/` — every figure script

Run from `qcs/`. Scripts marked **final** produce the figures used in the final report and presentation.

## Final charts and diagrams (read `results/`, write `figures/final/` and `figures/analysis/`)

| Script | What it draws | Output |
|---|---|---|
| `make_slide_diagrams.py` | **The architecture diagram and the complete circuit diagram** (amplitude encoding, 8 layers of 9 gates, a readout on all ten qubits, one gate opened up, what is trained). | `figures/final/architecture.*`, `circuit.*` |
| `make_slide_figures.py` | The result charts at presentation size: the forgetting matrices, recall by distance, learning new things, the reversed-rule test, the stability-plasticity plane, the learning-rate fairness check. | `figures/final/*.png` |
| `make_final_figures.py` | Analysis versions of stability and plasticity (more detail, smaller type) and the scorecard. **Also writes `results/final_summary.json`**, which the report and presentation read. | `figures/analysis/FIN1` to `FIN4` |
| `make_evidence_figures.py` | Analysis versions of the evidence charts. **Also writes `results/evidence_summary.json`.** | `figures/analysis/EV1` to `EV3` |

## Original figures (version 1, single-qubit readout)

| Script | What it draws |
|---|---|
| `make_hero_diagram.py`, `make_diagrams.py` | Architecture, pipeline and stack diagrams (`figures/architecture/`) |
| `make_overall_circuit.py`, `make_circuit_figure.py`, `make_vqc_canonical.py` | The original circuit figures, including a PennyLane rendering of the real model (`figures/circuit/`) |
| `make_result_figures.py` | The original result figures RES1 to RES6 (`figures/results/`) |

These describe the original circuit that measured one qubit. For the final circuit use `figures/final/circuit.png`.
