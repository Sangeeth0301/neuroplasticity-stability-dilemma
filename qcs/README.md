# Dual-Manifold ContextSync

**A quantum–classical hybrid architecture for the stability–plasticity dilemma in long,
multi-user LLM chat threads.**

Stability and plasticity are normally carried by the same parameter vector, so they
compete for the same coordinates. This project puts them on two different mathematical
objects: an **append-only classical memory** that nothing can gradient-erase, and a
**10-qubit variational circuit** whose parameters live on a compact manifold where
learning capacity provably cannot decay. The language model is frozen throughout.

Reference paper: Chen & Zhang, *Intrinsic Preservation of Plasticity in Continual Quantum
Learning*, PRX Quantum **7**, 033003 (2026) — `references/paper 4.pdf`.

---

## Start here

| If you want… | Open |
|---|---|
| The full technical report | `deliverables/Dual_Manifold_ContextSync_Report.docx` |
| The presentation | `deliverables/Dual_Manifold_ContextSync_Deck.pptx` |
| How the 10 phases build the dataset and satisfy both properties | `deliverables/Ten_Phases_Dataset_Stability_Plasticity.docx` |
| The architecture argument, in the browser | `deliverables/Architecture_Blueprint.html` |
| The repository and figure catalogue | `deliverables/Build_Atlas.html` |
| The running build record, phase by phase | `MASTER_DOC.md` |

---

## Headline result, stated honestly

Three seeds, 120 sequential tasks, every arm at its own tuned learning rate.

| Arm | \|θ\| | rank-1 | MRR | stale-above ↓ | Tr(F) after 120 tasks |
|---|---|---|---|---|---|
| quantum (QRE) | 1,080 | 0.928 ± 0.038 | 0.959 ± 0.022 | 0.015 | **4.75× — capacity grew** |
| **linear** (parameter-matched) | 1,025 | **0.979 ± 0.002** | **0.990 ± 0.001** | 0.000 | 0.28× — capacity lost |
| mlp (capacity reference) | 65,665 | 0.996 ± 0.002 | 0.998 ± 0.001 | 0.000 | 0.06× — capacity collapsed |
| cosine (no learning) | 0 | 0.894 | 0.943 | 0.078 | — |

Two things follow, and the project reports both:

- **The quantum arm loses on ranking quality**, significantly: Δ −0.056 MRR against its
  parameter-matched classical twin, 95% CI [−0.070, −0.043], p = 0.000 by paired bootstrap
  over threads — at roughly 135× the compute. The design required this comparison to be
  falsifiable, and it falsified in the classical arm's favour at this scale.
- **On the metric the reference paper is actually about, the predicted signature appears.**
  Effective learning capacity rises for the quantum arm and collapses for both classical
  arms. That is an early indicator, not a proof: 120 tasks is two orders of magnitude short
  of the horizon at which the published effect emerges.

Architecture claims that *are* supported: zero retrieval failures and zero evictions on the
sealed test split, stale-fact errors down from 0.130 to 0.000–0.035, no unlabelled stale
fact ever reaching a prompt, and ranking unchanged under depolarising noise up to p = 1e-3.

---

## Layout

```
qcs/
├── README.md                 you are here
├── MASTER_DOC.md             the build record: every phase, what it measured, what changed
│
├── deliverables/             report · phases document · deck · two HTML documents
├── figures/
│   ├── architecture/         the system diagrams
│   ├── circuit/              the variational circuit, the gate, the PennyLane cross-check
│   ├── results/              RES1–RES6, the six result figures
│   └── phases/               V01–V46, the per-phase evidence figures
├── phases/                   00–09, one folder per phase with its objective and evidence
├── references/               the reference papers
│
├── src/qcs/                  the package — the actual system
│   ├── datagen/              corpus generation (the plan IS the ground truth)
│   ├── ingest/  encode/      segmentation and the frozen encoding stack
│   ├── memory/               the append-only store, graph, salience, retrieval
│   ├── quantum/              gates · encoding · ansatz · readout · circuit · fisher · noise
│   ├── classical/            the parameter-matched twin and the baselines
│   ├── train/                dataset · replay · anchor · the continual loop
│   ├── assemble/             budget · knapsack · prompt assembly
│   ├── eval/                 metrics · headroom · significance · failure analysis
│   └── viz/                  one style module, one figure module per phase
├── scripts/                  thin CLIs — argument parsing only
│   └── build_docs/           the report, phases document and deck generators
├── tests/                    88 tests
├── configs/                  base.yaml + tuned.yaml — every tunable number
├── data/raw/                 the generated corpus (immutable)
├── data/processed/           frozen artefacts: whitening, projection, the memory database
├── runs/                     one canonical run per phase: config, env, metrics, figures
└── archive/                  superseded V1 material, kept for provenance only
```

---

## Reproducing the results

```bash
python -m pip install -r requirements.txt
python -m pip install -e .

python scripts/gen_dataset.py             # corpus + validation + V01–V04
python scripts/fit_encoder.py             # frozen encoding stack + V05–V08, V13
python scripts/study_whitening.py         # the shrinkage sweep behind whiten_eps
python scripts/build_memory.py            # memory manifold + V09–V12, V10b
python scripts/study_trainability.py      # quantum core + V14–V20
python scripts/tune_lr.py                 # per-arm learning rates -> configs/tuned.yaml
python scripts/train_continual.py --config tuned
python scripts/run_assembly.py --checkpoints runs/<phase6-run>/checkpoints
python scripts/run_eval.py --seeds 7 17 27

python -m pytest                          # 88 tests
```

Figures and documents:

```bash
python scripts/make_hero_diagram.py       # figures/architecture
python scripts/make_diagrams.py
python scripts/make_overall_circuit.py
python scripts/make_circuit_figure.py     # figures/circuit
python scripts/make_vqc_canonical.py
python scripts/make_result_figures.py     # figures/results

npm install                               # docx + pptxgenjs, for the document builders
node scripts/build_docs/report_body.js    # -> deliverables/…Report.docx
node scripts/build_docs/build_phases_doc.js
node scripts/build_docs/build_deck.js     # -> deliverables/…Deck.pptx
```

Two caches were removed to keep the submission small; both regenerate automatically:
`data/interim/` (sentence-embedding cache, rebuilt by `fit_encoder.py`) and
`data/processed/scoring_train.npz` (feature tensors, rebuilt by
`train_continual.py --rebuild`).

---

## Environment

Verified on Python 3.14, Windows 11, CPU only.

| package | version | role |
|---|---|---|
| pennylane | 0.44.0 | independent cross-check of the circuit |
| qiskit | 2.3.0 | hardware-parity path (phase 9) |
| tensorcircuit-ng | 1.9.1 | cross-check against the reference paper's stack |
| torch | 2.10.0+cpu | autodiff, the classical arms, the statevector simulator |
| sentence-transformers | 5.2.3 | the frozen encoder |
| faiss-cpu | 1.15.0 | candidate prefilter |
| numpy / scipy / matplotlib | 2.4.0 / 1.17.0 / 3.10.8 | core |

The quantum simulator used for training is the project's own batched statevector code in
`src/qcs/quantum/` — differentiable end to end, and cross-checked against PennyLane at
fidelity 1.000 so the implementation cannot silently drift.

---

## Status

Phases 0–8 complete. Phase 9 (hardening) not started: the long-horizon run of 1,000–3,000
tasks that would settle the plasticity question, hardware parity with finite shots, a
scaling study across qubit counts, a real frozen LLM in place of the extractive reader,
and a one-command reproduction kit.
