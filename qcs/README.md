# Dual-Manifold ContextSync

**A classical memory plus a 10-qubit circuit, for the stability-plasticity problem in long, multi-user chat threads.**

A model that follows a long group chat has to remember facts stated hundreds of messages ago, and also accept those facts when someone
changes them. In an ordinary network both needs live in the same weights, so they compete. This project keeps them apart:

- **Stability** comes from an **append-only memory** (SQLite, with triggers that reject edits and deletes). Nothing in it is trained, so training cannot erase it.
- **Plasticity** comes from a **10-qubit variational circuit** (simulated exactly) that learns which stored message is relevant to a new question. Its angles wrap around a circle and its gates are unitary, so its parameters cannot grow without limit.
- The language model is frozen. The two sides meet at one point: a feature vector goes in and one relevance score comes out.

Reference paper: Chen and Zhang, *Intrinsic Preservation of Plasticity in Continual Quantum Learning*, PRX Quantum 7, 033003 (2026).

---

## Final results

Three seeds, 120 sequential tasks, each model at its own tuned learning rate. Test = the sealed test split (6 threads, 360 probes). Full detail: [`results/RESULTS_FINAL.md`](results/RESULTS_FINAL.md).

| Model | Parameters | Test MRR | Stability | Plasticity | Combined |
|---|---|---|---|---|---|
| **Quantum, 10-qubit readout (final)** | 1,080 + 12 | 0.974 ± 0.005 | 0.967 | 0.835 | **0.896** |
| Quantum, 1-qubit readout (original) | 1,080 + 2 | 0.861 ± 0.028 | 0.873 | 0.846 | 0.859 |
| Linear | 1,025 | 0.974 ± 0.001 | 0.982 | 0.194 | 0.323 |
| MLP | 65,665 | 0.995 ± 0.001 | 0.998 | 0.225 | 0.316 |

*Stability* = score on the first 30 tasks after all 120 are learned. *Plasticity* = score after 40 steps on a reversed rule (the outdated fact becomes the correct answer). *Combined* = harmonic mean.

What the evidence supports, stated plainly:

- Reading out **all ten qubits** instead of one raised test MRR from 0.861 to 0.974 (+0.114, significant). It **ties** the linear model (difference 0.001, interval −0.007 to +0.009); the larger MLP is more accurate.
- Every model **remembers the past**: facts older than 400 turns are still found (0.97 for the circuit), while a model that sees only the last 100 turns scores 0 beyond 200 turns. This comes from the memory.
- After 120 tasks the circuit **keeps its ability to learn a changed rule** better than linear and MLP at their normal learning rates. When every model gets its best learning rate, the classical models recover part of the gap (linear 0.72, MLP 0.71, circuit 0.97 after 80 steps), so the honest claim is *faster and steadier adaptation*, not that classical models cannot learn.
- Not shown: anything beyond 120 tasks, real quantum hardware, or a real language model (a stand-in reader is used). The test split has only 6 threads.

---

## Repository layout

```
qcs/
├── README.md              you are here
├── MASTER_DOC.md          the build record: every phase, what it measured, what changed
├── configs/               base.yaml + tuned.yaml: every tunable number
├── data/                  the generated corpus (raw/) and frozen artefacts (processed/)
├── src/qcs/               the system: datagen, ingest, encode, memory, quantum, classical, train, assemble, eval, viz
├── scripts/               command-line entry points
│   ├── pipeline/            the build, phase by phase
│   ├── experiments/         the final experiments (3 seeds x 4 models)
│   ├── figures/             every figure script
│   └── report/              builds results, presentation and report
├── tests/                 88 automated tests
├── phases/                one folder per build phase: objective, exit criterion, result
├── runs/                  the record of the build phases (config, environment, metrics, figures)
├── results/               the final experiment outputs and RESULTS_FINAL.md
├── figures/               final/ (report figures), analysis/, branding/, and the version 1 figures
└── deliverables/          the Team_14 report and presentation (+ archive_v1/)
```

**Every code folder has its own README that explains the important logic file by file.** Start with
[`src/qcs/README.md`](src/qcs/README.md).

---

## Quick start

Python 3.11 or newer (verified on 3.14, Windows 11, CPU only).

```bash
cd qcs
python -m pip install -r requirements.txt
python -m pip install -e .
python -m pytest                                   # 88 tests

# the cached feature files are not stored in git; rebuild them once
python scripts/pipeline/build_scoring_cache.py

# one final experiment: MODEL in quantum | multi | linear | mlp, SEED in 7 | 17 | 27
python scripts/experiments/final_eval.py multi 7
python scripts/experiments/evidence_eval.py multi 7
python scripts/experiments/plasticity_fairness.py multi 7

# rebuild summaries, figures and the written results
python scripts/figures/make_final_figures.py
python scripts/figures/make_evidence_figures.py
python scripts/report/finalize_results.py
```

The full rebuild from an empty folder (dataset, encoder, memory, quantum core, learning rates, training) is listed in
[`scripts/README.md`](scripts/README.md).

## How one question is answered

1. The chat is cut into short **units** and each unit is stored word for word in the **memory**.
2. For a new question, the memory returns the 64 best earlier candidates (never a later one).
3. Each (question, candidate) pair becomes **1,024 numbers** of unit length: question, candidate, their product, and relational features (age, speaker, graph links, salience).
4. The numbers become the amplitudes of a **10-qubit state** (2^10 = 1,024). Eight layers of two-qubit gates (72 gates, 1,080 angles) act on it.
5. All ten qubits are measured; a weighted sum and a sigmoid give one **relevance score**.
6. Candidates are chosen by value per token within a 3,072-token budget; outdated facts are dropped or labelled.
7. During training only the angles and readout weights change, protected by replay, a Fisher anchor and a trust region.

## Environment

| Package | Version | Role |
|---|---|---|
| torch | 2.10 (CPU) | autodiff, the classical models, the state-vector simulator |
| pennylane | 0.44 | independent cross-check of the circuit |
| sentence-transformers | 5.2 | the frozen encoder |
| faiss-cpu | 1.15 | candidate retrieval |
| numpy / scipy / matplotlib | 2.4 / 1.17 / 3.10 | core |
| qiskit, tensorcircuit-ng | optional | cross-checks and the hardware-parity path (not started) |

## Status

Build phases 0 to 8 are complete. Phase 9 (the 1,000+ task run, hardware parity with finite shots, a scaling study, a real frozen
language model, a one-command reproduction kit) has not been started.
