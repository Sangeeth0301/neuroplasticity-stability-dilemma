# Dual-Manifold ContextSync — Master Document

**Living record of the build.** Updated at the end of every phase: what was built, how it
works, what the figures showed, what changed as a result, and what the next phase needs.

- **Blueprint (architecture):** `Q_ContextSync_V2_Blueprint.html` — https://claude.ai/code/artifact/28ba1ecc-05fb-4234-b272-735ed24c195e
- **Build Atlas (repo + figure catalogue):** `Q_ContextSync_Build_Atlas.html` — https://claude.ai/code/artifact/baf769c4-adbc-43e5-aed1-3da77d4a85b1
- **Reference paper:** Chen & Zhang, *Intrinsic Preservation of Plasticity in Continual Quantum Learning*, PRX Quantum **7**, 033003 (2026).

---

## The idea in one paragraph

The stability–plasticity dilemma is normally posed as one dial on one parameter vector, so
retention and adaptation fight over the same coordinates. This project refuses that framing
and puts them on two different mathematical objects. **Stability** lives on ℳ, an
append-only classical memory of every semantic unit ever said — nothing there is learned, so
nothing there can be gradient-erased, and forgetting degrades from an irreversible weight
update into a *retrieval miss*. **Plasticity** lives on 𝒰, a 10-qubit variational circuit
whose parameters sit on a compact periodic manifold where unitarity bounds gradient norms and
`Tr(F)` provably cannot collapse — so it keeps learning for an unbounded number of tasks. The
LLM is **frozen**: no gradient path reaches it, so catastrophic forgetting of its own
capabilities is absent rather than mitigated. Context saturation is handled where it actually
lives — a budgeted knapsack that decides what enters a fixed-size prompt. The quantum layer
buys plasticity and nothing else; stability comes from the architecture.

---

## Status board

| Phase | Name | Status | Exit criterion | Evidence |
|---|---|---|---|---|
| 0 | Foundations | ✅ done | identical seeds → identical output | `tests/test_config.py` (7 tests) |
| 1 | Dataset | ✅ done | oracle = 1.00, random ≈ chance | oracle **1.000**, random **0.049** |
| 2 | Ingest + encoding stack | ✅ done | unit-norm 1024-vectors for every triple | recall@20 **0.996** held-out, block shares on target |
| 3 | Memory manifold ℳ | ✅ done | append-only proven, ANN recall ≥ 0.95 @ M=256 | recall@256 **1.000**, 10 append-only tests |
| 4 | Quantum core | ✅ done | gradient agreement < 1e-6, depth chosen by evidence | agreement **8.6e-9**, PennyLane cross-check passes |
| 5 | Baselines + tuned twin | ✅ done | every arm at its own best lr | lr sweep changed the linear arm 0.063 → 0.965 MRR |
| 6 | Continual training | ✅ done (120 tasks, 1 seed) | full metric suite logged, both arms | quantum Tr(F) rises, classical collapses |
| 7 | Assembly + reader | ✅ done | budget never exceeded, curves produced | **0** unlabelled stale facts served |
| 8 | Evaluation | ✅ done | every claim has a figure | 3 seeds · 7 ablations · noise sweep · sealed test split |
| 9 | Hardening | ⬜ not started (stopped here on request) | clean-clone reproduction | — |

---

## Environment

Installed and verified on this machine (Python 3.14, Windows):

| package | version | role |
|---|---|---|
| pennylane | 0.44.0 | primary quantum framework |
| qiskit | 2.3.0 | hardware-parity cross-check (Phase 9) |
| tensorcircuit-ng | 1.9.1 | cross-check against the reference paper's stack |
| torch | 2.10.0+cpu | autodiff backend, classical arm |
| sentence-transformers | 5.2.3 | frozen encoder |
| faiss | 1.15.0 | ANN prefilter |
| umap-learn | 0.5.12 | embedding geometry figures |
| numpy / scipy / matplotlib / pandas | 2.4.0 / 1.17.0 / 3.10.8 / 2.3.3 | core |

Install the package itself in editable mode: `python -m pip install -e .`

---

## Phase 0 — Foundations ✅

**Objective.** Make every later experiment re-runnable by someone else, and make randomness a
controlled variable rather than an excuse.

**What was built**

| File | Role |
|---|---|
| `src/qcs/paths.py` | canonical directory layout; every module asks this one for paths |
| `src/qcs/config.py` | `Config` dataclass tree, YAML load/overlay, content hash, **invariant checks** |
| `src/qcs/seeding.py` | `seed_everything` + `child_rng(seed, *tags)` reproducible sub-streams |
| `src/qcs/logging_.py` | `RunLogger` — per-run directory, `metrics.jsonl` with a guarded schema, `env.json` |
| `src/qcs/types.py` | `Fact`, `Turn`, `SemanticUnit`, `Triple`, `ThreadPlan` — all JSON-serialisable |
| `src/qcs/viz/style.py` | the *only* place figure colours, fonts and save paths are defined |
| `configs/base.yaml` | every tunable number in the system |

**Two decisions worth recording**

1. `child_rng` hashes with `hashlib`, not Python's `hash()`. CPython salts string hashes per
   process, so `hash()`-derived seeds would silently produce a different corpus on every run
   — the exact class of bug that makes a project irreproducible without anyone noticing.
2. `Config.validate()` enforces the architecture's hard constraints as assertions, not
   comments: `feature_dim == 2**n_qubits` (lossless amplitude encoding) and
   `depth >= n_qubits//2` (the light-cone floor found in the pre-build trainability study).
   A config that violates them cannot be loaded.

**Exit criterion met.** `pytest tests/test_config.py` — 7 passed, including two that assert the
validator rejects bad configs.

---

## Phase 1 — Dataset ✅

**Objective.** A multi-user thread corpus in which the correct answer to *"what should have
been retrieved?"* is known by construction. This comes before any modelling, because a corpus
without ground truth makes every later result unfalsifiable.

**How it works.** Generation is a two-step process, and the separation is the whole point:

```
   plan  ─────────────────────────────►  surface text
   (structured, IS the ground truth)     (templates, may not add or remove facts)
```

`thread_planner.plan_thread()` decides what happens on every turn *before any prose exists*:
which turn introduces which fact, which turn supersedes it, which turn probes it and at what
distance. `surface_realizer` then writes the words. Because the plan is authoritative, the set
of facts required to answer any probe is exact.

**Turn kinds**

| kind | meaning |
|---|---|
| `fact` | a participant states a new (entity, attribute, value) fact they own |
| `update` | a participant supersedes an earlier fact (adversarial regime only) |
| `probe` | a participant asks a question answerable only from stated facts |
| `distractor` | mentions a **real entity** with no usable value — lexically close noise |
| `filler` | ordinary conversation |

**Three regimes**, cycled across threads:

- `single_owner` — all facts from one participant.
- `multi_owner` — facts spread across participants, probes asked by *someone else* (this is
  what makes persona-bleed measurable).
- `adversarial` — as multi-owner, plus supersession chains: a fact is overwritten mid-thread
  and the system must return the new value, never the stale one.

**Corpus generated** (`configs/base.yaml`, seed 7):

| quantity | value |
|---|---|
| threads | 24 train / 6 test (sealed) |
| turns | 14,400 train / 3,600 test |
| probes | 1,440 train / 360 test |
| facts | 1,871 (incl. supersession replacements) |
| probe distance bands | 10 / 50 / 200 / 500 turns |
| **oracle recall** | **1.000** |
| **random recall** | **0.049** |

**Figures**

| Id | File | Read |
|---|---|---|
| V01 | `V01_thread_structure_train-{000,001,002}.png` | one per regime: turn map by speaker with fact→probe arcs |
| V02 | `V02_probe_distance_hist.png` | all four bands populated, long tail present |
| V03 | `V03_fact_ownership.png` | ownership by regime, attribute mix, supersession counts |
| V04 | `V04_corpus_stats.png` | turn kinds, word counts, speakers, thread sizes |

**What V01 caught on the first run.** The initial planner placed fact-introduction turns first
and then chose probe distances, which forced every fact into the opening ~120 turns of a
600-turn thread — recency alone would have separated facts from filler, and the retrieval task
would have been fake. The fix inverts the order: **assign the probe distance band first**
(round-robin, so bands stay balanced), then place the fact anywhere that still leaves room for
its probe. `tests/test_datagen.py::test_probe_bands_are_balanced` locks the property in. This
is the visualization layer paying for itself on day one.

**Validation** (`datagen/validate.py`) — each assertion maps to a way the corpus could quietly
stop being a valid instrument:

| id | invariant |
|---|---|
| V1 | every probe's required facts appear strictly earlier in the same thread |
| V2 | no probe requires a superseded fact (otherwise "correct" is ambiguous) |
| V3 | fact ids unique corpus-wide |
| V4 | contiguous turn indices, speakers inside the participant set |
| V5 | probe distances actually cover the configured bands |
| V6 | no train/test thread overlap |
| V7 | oracle scores 1.0, random scores near chance |

**Exit criterion met.** `pytest` — 15 passed, including a test that feeds the validator a
deliberately broken corpus and asserts it raises (a validator that never fails is not a
validator).

**Commands**

```bash
python scripts/gen_dataset.py
python -m pytest -q
```

---

## Phase 2 — Ingest and encoding stack ✅

**Objective.** A deterministic, frozen, well-conditioned map from text to a unit-norm
1024-dimensional vector — the classical half of the quantum interface.

**What was built**

| File | Role |
|---|---|
| `ingest/segment.py` | turn → semantic units; **lossless** (concatenation reproduces the turn) |
| `ingest/loader.py` | JSONL → typed records → SUs, grouped by thread |
| `encode/encoder.py` | frozen MiniLM (384-d), `requires_grad_(False)` enforced, hashing fallback if offline |
| `encode/cache.py` | disk cache keyed by `sha256(text)` — survives corpus regeneration |
| `encode/whiten.py` | PCA whitening, **fit-once** (refit raises), shrinkage `eps` |
| `encode/projection.py` | InfoNCE 256→256 projection with hard in-thread negatives, then frozen |
| `encode/relational.py` | the ρ block; every feature bounded to [−1,1] *before* normalisation |
| `encode/assemble.py` | the 1024-d assembly with **per-block norm balancing** |
| `encode/pipeline.py` | the whole stack as one frozen, shareable object |
| `viz/embedding_viz.py` | V05–V08, V13 |

**The encoding path, with shapes**

```
text ──► MiniLM ──► whiten ──► P ──► [ p_q | p_c | p_q⊙p_c | ρ ] ──► L2 ──► |ψ₀⟩
         (384)      (256)     (256)   256   256      256     256      1024   10 qubits
```

**Results (held-out threads, never seen by the projection)**

| representation | recall@20 |
|---|---|
| raw MiniLM | 0.879 |
| whitened | 0.887 |
| **projected (frozen P)** | **0.996** |

Block norm shares landed exactly on target — `p_q` 0.350, `p_c` 0.350, interaction 0.200,
ρ 0.100 — and participation ratio rose to **189 / 1024** (18.5% of the register) from the
**61 / 1024** the pre-build demo produced without balancing.

**Two findings that changed the design**

1. **Whitening with a tiny epsilon destroys retrieval.** The first run showed whitened
   recall@20 collapsing from 0.879 to 0.633. `scripts/study_whitening.py` (figure V06b)
   swept rank × shrinkage and found the cause: with `eps = 1e-6`, low-variance noise
   directions get rescaled to the same magnitude as signal directions. At rank 128 the
   damage is worst (recall 0.338). Shrinkage fixes it without giving up rank —
   `whiten_eps: 0.1` is now in `base.yaml`, chosen from the sweep rather than defaulted,
   and projected recall went 0.887 → **0.996**.
2. **Retention must be measured off the projection's own training pairs.** The first
   measurement used the InfoNCE anchors and reported a meaningless 1.000 rank-1
   accuracy. Retention is now evaluated only on the held-out threads.

**Exit criterion met.** 28 tests pass, including: segmentation round-trip, unit-norm
assertions, batch/single equivalence, ρ bounded under absurd inputs (gap = 100,000 turns),
block targets hit even when input blocks arrive at 10⁶× different scales, and a test
showing that *unbalanced* blocks collapse the participation ratio — the failure this
component exists to prevent.

**Commands**

```bash
python scripts/fit_encoder.py --epochs 20
python scripts/study_whitening.py
```

---

## Phase 3 — Memory manifold ℳ ✅

**Objective.** A store where writes never destroy, retrieval is fast, and provenance
survives into the prompt.

**What was built**

| File | Role |
|---|---|
| `memory/schema.sql` | tables + **SQLite triggers that abort any UPDATE or DELETE** on `semantic_units` |
| `memory/store.py` | `MemoryStore`: writes, causal reads, salience blobs, selection log |
| `memory/ann_index.py` | FAISS prefilter, exact inner product, with a **causality mask** |
| `memory/consolidate.py` | additive summaries — sources verified byte-identical afterwards |
| `memory/graph.py` | entity / supersedes / contradicts DAG; fills the 32 graph dims of ρ |
| `memory/salience.py` | per-participant EMA vectors, deterministic 256→64 folding |
| `eval/headroom.py` | what cosine similarity *cannot* do — the scorer's reason to exist |
| `viz/memory_viz.py` | V09–V12 plus V10b |

**Append-only is enforced by the database, not by convention.** Two triggers raise
`RAISE(ABORT, 'semantic_units is append-only')`. Three tests attempt an `UPDATE`, a
`DELETE`, and a silent overwrite-by-reinsert; all three are rejected. This is the
mechanical foundation of the stability claim — knowledge that cannot be overwritten
cannot be catastrophically forgotten.

**Store built** (12 threads): 7,200 semantic units · 1,920 facts · 144 summary nodes ·
9,989 edges · salience for every participant.

**Results**

| measurement | value | read |
|---|---|---|
| ANN recall @ M=256 | **1.000** | exit criterion (≥0.95) — PASS |
| ANN recall @ M=8 | 1.000 | the prefilter is not the bottleneck |
| salience separation | 0.892 | participants are distinguishable in ρ |
| consolidation sources intact | true | summaries added, nothing consumed |

**The finding that shaped the rest of the project.** Perfect prefilter recall is a
*ceiling*, not a score, so the real question is what ranking still gets wrong. `V10b`
measures it:

| measurement | value |
|---|---|
| cosine rank-1 accuracy | 0.894 → **10.6% headroom** |
| rank-1, adversarial regime | 0.871 (worst of the three) |
| supersession probes | 49 |
| **stale fact outranks its replacement** | **28.6%** |

A superseded fact and its replacement are near-identical text about the same entity, so
semantic similarity is *structurally blind* between them. Only the ρ block — recency plus
the supersession edges built in this phase — can break that tie. That gap is exactly what
the learned scorer has to earn, and it is why the ρ block is in the architecture at all.

**Two corpus changes forced by measurement**

1. **One fact per entity made retrieval trivial.** The first build hit cosine recall
   1.000 at M=8 because each fact had a unique entity name, so the entity string was a
   primary key. `fact_bank.py` now issues ~3 facts per entity with *different*
   attributes, so the retriever must resolve the attribute too.
2. **Supersession density raised 0.20 → 0.35.** The first measurement found only 16
   supersession probes across 6 threads — too few to train or evaluate on. Now 49 across
   12 threads.

**Exit criterion met.** 39 tests pass, including causality (a future candidate is
unreachable through the masked index), consolidation idempotence, and graph features
defaulting to *unknown* (zeros) rather than to a false positive claim.

**Commands**

```bash
python scripts/build_memory.py --threads 12
```

---

## Phase 4 — Quantum core ✅

**Objective.** A correct, tested, differentiable quantum scorer whose depth is chosen by
evidence rather than taste.

**What was built**

| File | Role |
|---|---|
| `quantum/gates.py` | Pauli basis, **two** SU(4) forms, batched 2-qubit application |
| `quantum/encoding.py` | amplitude encoding, participation ratio, entanglement entropy |
| `quantum/ansatz.py` | brickwall / ladder / ring, `n_params`, **`light_cone_depth`** |
| `quantum/readout.py` | local ⟨Z_q⟩, true parity ⟨Z⊗…⊗Z⟩, shot sampling |
| `quantum/circuit.py` | `QRE` — the plasticity manifold 𝒰, an `nn.Module` |
| `quantum/gradients.py` | autodiff, parameter shift, agreement, gradient-variance sweep |
| `quantum/fisher.py` | Tr(F) empirical/model, FIM spectrum, effective rank, QFIM diagonal |
| `train/dataset.py` | real probe→candidate feature tensors, causality asserted twice |
| `viz/circuit_viz.py` | V14–V20 |

**The circuit.** N=10 qubits, brickwall SU(4), depth 8 → **1,080 parameters**
(`|θ| = L × 9 × 15`). Identity-block initialisation (`θ ≈ 0 ⇒ U ≈ I`). Readout is a
*local* ⟨Z₉⟩ through `σ(β⟨Z⟩+b)`. All M candidates are scored in one batched tensor
contraction, not a Python loop.

**Verification results**

| check | result |
|---|---|
| gate unitarity, `V(0) = I` | pass, both forms |
| amplitude encoding vs input | max abs error < 1e-7 |
| norm preserved at every layer | pass (1e-4) |
| **PennyLane cross-check** (state + ⟨Z⟩) | **fidelity 1.000, ⟨Z⟩ match < 1e-5** |
| autodiff vs parameter shift (product form) | **8.6e-9** ✅ |
| entanglement, mid-cut | 3.91 → 4.26 bits (Page value 5) |
| untrained readout separation | Cohen's d = −0.13 (correctly ≈ 0) |

**Finding 1 — the parameter-shift rule is invalid for the paper's gate form.**
`V(θ) = exp(−i/2 Σₖ θₖGₖ)` has **non-commuting** generators, so shifting one angle does
not isolate its derivative. Measured discrepancy against autodiff: **3.3e-3**, i.e. ~25%
relative. The fix was not to abandon the form but to add a second one:
`su4_product` = Πₖ exp(−i/2 θₖGₖ), where each parameter has its own involutory generator
and the two-term rule is exact (**8.6e-9**). `gate_form` is now a config knob: `exp`
stays the default for comparability with the reference paper (autodiff is exact
regardless), `product` is what a hardware run must use. A regression test asserts the
exponential form *disagrees* — so nobody later "fixes" it by silently changing the
training path.

**Finding 2 — the light cone, measured on real data.**
Gradient variance of a parameter on qubits (0,1), 25 random initialisations per depth:

| depth L | 1 | 2 | 4 | **6** | 8 | 12 | 16 |
|---|---|---|---|---|---|---|---|
| local ⟨Z₉⟩ | 3.6e-16 | 2.0e-16 | 4.8e-16 | **1.1e-5** | 5.9e-5 | 3.7e-5 | 4.7e-5 |

Exactly zero below depth 5 (`light_cone_depth(10, 9) = 5`), then live and flat.
Below the cone those parameters are not badly conditioned — they are *physically
disconnected* from the readout. `Config.validate()` enforces `depth ≥ n_qubits//2`.

**Finding 3 — CORRECTION to an earlier claim.** The pre-build demo reported the global
observable as a barren plateau collapsing fifteen orders of magnitude. **That was wrong.**
The demo computed `Π_q ⟨Z_q⟩` — a product of marginals, which decays exponentially in N
by construction — not the global observable. With the correct parity expectation
`⟨Z⊗…⊗Z⟩ = Σ_j (−1)^popcount(j)|ψ_j|²`, variance stays ≈2e-4 across depths 1–16: **no
plateau at this scale.** Both the standalone demo and figure V17 now agree. The local
readout remains the default, justified by the light cone and by standard barren-plateau
theory for deep global-observable circuits — not by our own measurement. The demo script
and the published Build Atlas have both been corrected.

**Exit criterion met.** 62 tests pass, including the PennyLane cross-check and a test
that gradients vanish below the light cone while remaining alive above it.

**Commands**

```bash
python scripts/study_trainability.py --threads 4 --probes 30 --samples 25
python -m pytest tests/test_quantum.py -q
```

---

## Phase 5 — Baselines and the tuned classical twin ✅

**Objective.** Know exactly what must be beaten, with classical arms built with equal care.

**The arms** (all implement the identical `score(X) → r` interface):

| arm | what it is | |θ| |
|---|---|---|
| `quantum` | the QRE, depth 8 | 1,080 |
| `linear` | **parameter-matched twin**, 1024→1 | 1,025 |
| `mlp` | capacity reference, 1024→64→1 | 65,665 |
| `cosine` | B1, no learning | 0 |
| `recency` | B0 truncation floor | 0 |
| `random` / `oracle` | chance floor / ceiling | 0 |

A classical net consuming a 1024-dim input needs ≥1024 weights for one linear layer, so
the honest parameter-matched twin is the *linear* scorer. The wider MLP is reported too,
clearly labelled — hiding it would produce exactly the handicapped baseline that makes
quantum results unconvincing.

**Finding — the first comparison was unfair, and measuring it proved it.** The reference
paper's experiment 1 uses lr 0.03 for the QNN and 1e-3 for the MLP. Running that way, the
linear arm scored MRR **0.063**. `scripts/tune_lr.py` trains each arm on a task prefix and
scores it on held-out later tasks:

| arm | 0.001 | 0.005 | 0.02 | 0.05 | 0.1 | best |
|---|---|---|---|---|---|---|
| quantum | 0.29 | 0.44 | **0.62** | 0.51 | 0.30 | 0.02 |
| linear | 0.03 | 0.04 | 0.72 | 0.94 | **0.97** | 0.1 |
| mlp | 0.04 | 0.92 | **0.97** | 0.97 | 0.86 | 0.02 |

At its own best learning rate the linear arm goes **0.063 → 0.965**. Every comparison
after this point uses `configs/tuned.yaml`, and any result reported at mismatched learning
rates should be treated as invalid.

---

## Phase 6 — Continual training ✅ *(120 tasks, 1 seed — an interim run)*

**What was built:** `train/replay.py` (reservoir, uniform over history) ·
`train/anchor.py` (QFIM-weighted anchor, trust region, PSI drift detector) ·
`train/loop.py` (the only module that updates parameters) · `viz/training_viz.py`.

**Results, 120 tasks × 6 probes, all arms at their tuned learning rate**

| arm | rank-1 | MRR | AUC | stale-above ↓ | BWT | Tr(F) trend | wall clock |
|---|---|---|---|---|---|---|---|
| cosine (B1) | 0.894 | 0.943 | 0.932 | 0.078 | — | — | — |
| recency (B0) | 0.025 | 0.124 | 0.532 | 0.521 | — | — | — |
| **quantum** | 0.911 | 0.950 | 0.998 | **0.005** | +0.028 | **↑ 2.5×** | 406 s |
| **linear** (matched) | **0.981** | **0.990** | 1.000 | **0.000** | +0.046 | **↓ 0.30×** | 3 s |
| mlp (65k) | 0.999 | 0.999 | 1.000 | 0.000 | +0.015 | **↓ 0.06×** | 20 s |
| oracle | 1.000 | 1.000 | 1.000 | 0.000 | — | — | — |

**Read this honestly, in three parts.**

1. **On ranking quality the quantum arm loses.** The parameter-matched linear scorer beats
   it (0.981 vs 0.911 rank-1) at **135× less wall clock**. At this scale and horizon there
   is no quantum advantage in accuracy, and the blueprint never claimed one.
2. **On the metric the reference paper is actually about, the predicted signature is
   already visible at 120 tasks.** Normalised `Tr(F)` — effective learning capacity — rises
   to ≈2.5× for the quantum arm and holds, while the linear arm decays to 0.30× and the MLP
   collapses to **0.06×** after task 80. Gradient norms tell the same story (quantum 1.2×,
   classical 0.2–0.35×), and the unwrapped parameter norm of the linear arm grows to 5×
   and is still climbing while the quantum arm plateaus — the compact-manifold mechanism,
   drawn.
3. **What that does and does not mean.** Losing `Tr(F)` is losing the *ability to keep
   learning*, not current accuracy. Both classical arms are still near ceiling because 120
   tasks is far too short for capacity loss to show up as error — the paper needed
   1,000–3,000 tasks. So the headline plasticity claim is **not yet tested**; what we have
   is an early indicator pointing the way the paper predicts. The long run is Phase 8 work,
   and the quantum arm's 3.4 s/task is the binding constraint on it.

**Also validated:** all three learned arms drive the stale-fact error from cosine's 0.078
to ≈0, which is the ρ-block architecture paying off — and that result is independent of
whether the scorer is quantum.

---

## Phase 7 — Budgeted assembly and the frozen reader ✅

**What was built:** `assemble/budget.py` (exact tokenizer counts, reserved sub-budgets) ·
`assemble/knapsack.py` (greedy submodular selection, (1−1/e) guarantee, plus an explicit
top-k baseline to beat) · `assemble/pipeline.py` (supersession filter, provenance-tagged
template, `ExtractiveReader`) · `viz/assemble_viz.py`.

**Answer accuracy vs token budget** (80 probes, reserves scaled with B):

| arm | B=128 | 256 | 512 | 1024 | 2048 |
|---|---|---|---|---|---|
| quantum | 0.975 | 0.975 | 0.975 | 0.975 | 0.975 |
| linear | 1.000 | 1.000 | 1.000 | 1.000 | 1.000 |
| mlp | 1.000 | 1.000 | 1.000 | 1.000 | 1.000 |
| cosine | 0.938 | 0.988 | 1.000 | 1.000 | 1.000 |
| recency | 0.925 | 0.988 | 1.000 | 1.000 | 1.000 |

**Unlabelled stale facts served: 0 for every arm.** The supersession rule is a hard filter
— a superseded statement is either dropped or shown explicitly as `OUTDATED:` beside its
`CURRENT:` replacement — and the reader is tested to never answer from an OUTDATED line.

The learned arms reach their ceiling at **B=128** where the unlearned baselines need
**B=512** — a 4× budget reduction for the same answer quality, which is the "constant
context length independent of thread length" claim in its weakest measurable form.

**The honest limitation.** There is no LLM API wired in, so the frozen generator is
`ExtractiveReader`, which answers only if the required fact reached the prompt. That
measures precisely what this architecture is responsible for, and it cannot flatter the
system — but it also **cannot degrade when given too much context**, so the saturation
*turnover* (V44) is not observable with it. Demonstrating attention dilution needs a real
frozen LLM; until then, Phase 7 shows the budget is respected and selection is efficient,
not that saturation was defeated.

---

## Phase 8 — Full evaluation ✅

**Objective.** Produce results that survive an adversarial reader — including the ones
that did not go our way.

**What was built:** `quantum/noise.py` (depolarising noise by trajectory sampling) ·
`eval/significance.py` (paired bootstrap over **threads**, not probes) ·
`eval/failure_analysis.py` (retrieval / ranking / stale / eviction taxonomy) ·
`viz/eval_viz.py` (V35–V40, V46) · `scripts/run_eval.py`. Total runtime 40.6 min.

### Headline — 3 seeds (7, 17, 27), 120 tasks, each arm at its tuned learning rate

| arm | rank-1 | MRR | AUC | stale-above ↓ | |θ| |
|---|---|---|---|---|---|
| quantum | 0.928 ± 0.038 | 0.959 ± 0.022 | 0.998 ± 0.001 | 0.015 ± 0.008 | 1,080 |
| **linear** (matched) | **0.979 ± 0.002** | **0.990 ± 0.001** | 1.000 | **0.000** | 1,025 |
| mlp (capacity ref.) | 0.996 ± 0.002 | 0.998 ± 0.001 | 1.000 | 0.000 | 65,665 |

The quantum arm's seed-to-seed spread is **19× larger** than the linear arm's
(±0.038 vs ±0.002) — a stability cost, not just an accuracy one.

### Sealed test split — opened once, here, for the first time

| arm | MRR | rank-1 | AUC | stale-above ↓ |
|---|---|---|---|---|
| quantum | 0.919 | 0.856 | 0.996 | 0.035 |
| linear | 0.975 | 0.953 | 0.999 | 0.000 |
| mlp | 0.993 | 0.986 | 1.000 | 0.000 |
| cosine (B1) | 0.910 | 0.831 | 0.928 | **0.130** |
| recency (B0) | 0.128 | 0.028 | 0.530 | 0.478 |

Held-out numbers track the training-stream numbers closely (quantum 0.959 → 0.919), so the
result is not an artefact of evaluating where we trained. Per regime, every arm is within
±0.04 across single-owner / multi-owner / adversarial — no persona-bleed collapse.

### Significance — paired bootstrap over threads (5,000 resamples)

| comparison | Δ MRR | 95% CI | p | verdict |
|---|---|---|---|---|
| quantum − linear | −0.056 | [−0.070, −0.043] | 0.000 | **significant** |
| quantum − mlp | −0.074 | [−0.086, −0.060] | 0.000 | significant |
| linear − mlp | −0.018 | [−0.023, −0.012] | 0.000 | significant |

**The quantum arm is significantly worse than its parameter-matched classical twin on
ranking quality.** That is the honest headline of Phase 8, and it is reported as the
result rather than buried: the blueprint's claim 4 said this comparison had to be
falsifiable, and it falsified in the classical arm's favour at this scale and horizon.

### Ablations — quantum arm, 60 tasks, each variant retrained (V35)

| variant | MRR | Δ |
|---|---|---|
| **no ρ block at all** | 0.572 | **−0.324** |
| no replay | 0.733 | −0.162 |
| no graph/supersession block | 0.765 | −0.130 |
| global observable ⟨Z⊗…⊗Z⟩ | 0.804 | −0.092 |
| no trust region | 0.896 | ±0.000 |
| no salience block | 0.897 | +0.001 |
| no anchor (λ_Ω = 0) | 0.917 | **+0.022** |
| *full system* | *0.896* | — |

Four things this settles:

1. **The ρ block is the architecture.** Removing it costs 32 MRR points — more than every
   other component combined. The relational features (recency, supersession, graph) are
   what the scorer is actually using, exactly as the Phase 3 headroom analysis predicted.
2. **Replay earns its place** (−0.162), and the supersession/graph features earn theirs
   (−0.130).
3. **The local readout is validated on task performance**, not just on trainability
   theory: switching to the global observable costs 0.092.
4. **The QFIM anchor currently *hurts* slightly** (+0.022 when removed). At 60 tasks there
   is nothing to protect yet, so a Fisher-weighted penalty is pure drag. Its value can only
   appear over horizons long enough for forgetting to occur — which this run does not
   reach. Reported as-is rather than tuned away.

*(This table is the corrected one. The first ablation run was invalid: tagging runs
`abl_*` broke the arm-name parsing in `run_continual`, so every ablation trained the
quantum circuit at the **classical** learning rate. `kind` is now an explicit argument and
the fragile string parsing is gone.)*

### Noise robustness (V38) — trained circuit, depolarising noise by trajectory sampling

| p per gate | 0 | 1e-4 | 5e-4 | 1e-3 | 5e-3 |
|---|---|---|---|---|---|
| MRR | 0.908 | 0.908 | 0.904 | 0.904 | 0.887 |
| rank-1 | 0.850 | 0.850 | 0.850 | 0.850 | 0.825 |

Ranking is essentially unchanged through **p = 1e-3**, an order of magnitude beyond the
5e-4 the reference paper reports, and loses only 2 points at 5e-3. This is the one
quantum-side claim that came through cleanly.

### Failure taxonomy (V39, test split)

| arm | accuracy | ranking | stale | retrieval | eviction |
|---|---|---|---|---|---|
| mlp | 0.986 | 5 | 0 | 0 | 0 |
| linear | 0.953 | 17 | 0 | 0 | 0 |
| quantum | 0.856 | 48 | 4 | 0 | 0 |
| cosine | 0.831 | 47 | **14** | 0 | 0 |

**Zero retrieval failures and zero evictions for every arm** — ℳ and the knapsack are not
the bottleneck; every remaining loss is a ranking decision. Cosine's losses are
disproportionately stale-fact errors (14 of 61), which the learned arms almost eliminate.

### What Phase 8 establishes, stated plainly

- **Architecture claims: supported.** Append-only memory + prefilter + budgeted assembly
  produce zero retrieval and zero eviction failures; the ρ block is worth 0.32 MRR; stale
  facts drop from 0.130 (cosine) to 0.000–0.035; no unlabelled stale fact ever reaches a
  prompt.
- **Quantum ranking advantage: refuted at this scale.** Significantly worse than a
  1,025-parameter linear model, with 19× the seed variance and ~30× the wall clock.
- **Quantum plasticity advantage: still untested.** The Phase 6 Tr(F) divergence is a real
  early indicator, but 120 tasks is two orders of magnitude short of where the reference
  paper's effect appears. Nothing here confirms or refutes it.
- **Noise robustness: supported.**

---

## Open questions and risks

| item | status |
|---|---|
| Anchor (λ_Ω) is a net negative at 60–120 tasks | measured in Phase 8; expected to invert only at long horizons — keep, but do not claim value for it yet |
| Quantum arm is ~30× slower per task | the binding constraint on running the 1,000+ task experiment that would actually test plasticity |
| `ExtractiveReader` cannot show attention dilution | a real frozen LLM is needed before any saturation-turnover claim |
| Embedding dim is 384 (MiniLM), not 768 — projection is 384→256 | resolved; recorded in `configs/base.yaml` |
| Block-dominance at S6 | fix designed (per-block norm targets), to be verified by V13 in Phase 2 |
| Light-cone floor L ≥ 6 for N=10 | enforced in `Config.validate()` |
| ~~Global observable is a barren plateau~~ | **corrected in Phase 4**: the earlier measurement used a product of marginals. No plateau at N=10, L≤16. Local readout kept for the light-cone reason. |
| Template-generated prose overstates probe cleanliness | accepted for now; `threads_human.jsonl` external-validity set planned for Phase 8 |
