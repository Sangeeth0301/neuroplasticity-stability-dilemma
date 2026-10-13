# Dual-Manifold ContextSync

**Keeping stability and plasticity together in long, multi-user chat threads, with a classical memory and a 10-qubit circuit.**

Course project, *Quantum Computing and Algorithm* (Team_14).

A model that follows a long group chat must remember facts stated hundreds of messages ago, and also accept them when someone changes
them. In an ordinary network both needs live in the same weights, so they compete. Here they are kept apart:

- an **append-only memory** holds what must never be lost (nothing in it is trained, so training cannot erase it);
- a **10-qubit variational circuit**, simulated exactly, learns which stored message is relevant to a new question.

The language model is frozen. Everything lives in [`qcs/`](qcs/); start with **[`qcs/README.md`](qcs/README.md)**.

---

## Result

Three seeds, 120 tasks, sealed test split.

| Model | Test MRR | Stability | Plasticity | Combined |
|---|---|---|---|---|
| **Quantum, 10-qubit readout (final)** | 0.974 | 0.967 | 0.835 | **0.896** |
| Quantum, 1-qubit readout (original) | 0.861 | 0.873 | 0.846 | 0.859 |
| Linear (same size) | 0.974 | 0.982 | 0.194 | 0.323 |
| MLP (65k parameters) | 0.995 | 0.998 | 0.225 | 0.316 |

Reading out all ten qubits instead of one closes the ranking gap to the linear model (a tie, not a win), every model remembers old facts
through the memory, and the circuit keeps more of its ability to learn a changed rule, although classical models recover part of that
gap when their learning rate is tuned. The study covers 120 tasks, a simulated circuit and 6 test threads. Details and limits:
[`qcs/results/RESULTS_FINAL.md`](qcs/results/RESULTS_FINAL.md).

---

## Where things are

| Path | What it is |
|---|---|
| [`qcs/deliverables/`](qcs/deliverables/) | The project report (Word and PDF) and the presentation |
| [`qcs/src/qcs/`](qcs/src/qcs/README.md) | The code: one package per stage, each with a README that explains its logic file by file |
| [`qcs/scripts/`](qcs/scripts/README.md) | Command-line entry points: `pipeline/`, `experiments/`, `figures/`, `report/` |
| [`qcs/results/`](qcs/results/README.md) | The final experiment outputs |
| [`qcs/tests/`](qcs/tests/README.md) | 88 automated tests |
| [`qcs/MASTER_DOC.md`](qcs/MASTER_DOC.md) | The build record, phase by phase |

## Run it

```bash
cd qcs
python -m pip install -r requirements.txt
python -m pip install -e .
python -m pytest
```

The full rebuild and the final experiments are described in [`qcs/README.md`](qcs/README.md) and [`qcs/scripts/README.md`](qcs/scripts/README.md).

## Not in this repository

`.gitignore` excludes the cached feature files (`qcs/data/processed/scoring_*.npz`, rebuilt by
`scripts/pipeline/build_scoring_cache.py`), the sentence-embedding cache, the third-party reference papers and local archives.

## Reference

Yu-Qin Chen and Shi-Xin Zhang, *Intrinsic Preservation of Plasticity in Continual Quantum Learning*, PRX Quantum **7**, 033003 (2026).
