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
| 5 | Baselines + harness | ⬜ next | B0–B3 run end to end | — |
| 6 | Continual training | ⬜ | 500-task run, both arms, 3 seeds | — |
| 7 | Assembly + LLM | ⬜ | budget never exceeded, saturation curves | — |
| 8 | Evaluation | ⬜ | every claim has a figure | — |
| 9 | Hardening | ⬜ | clean-clone reproduction | — |

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

## Phase 5 — Baselines, classical twin, and the training harness ⬜ next

**Objective.** Know exactly what must be beaten, with a parameter-matched classical
scorer built with equal care.

**Planned work.** `classical/mlp_scorer.py` (identical interface, |θ| matched to 1,080) ·
`cosine_scorer.py` · `random_scorer.py` · `oracle_scorer.py` · the metric suite
(Recall@B, MRR, stale-fact rate, persona bleed) · a single `score(X) → r` interface every
arm implements · figures V21 onwards.

---

## Open questions and risks

| item | status |
|---|---|
| Embedding dim is 384 (MiniLM), not 768 — projection is 384→256 | resolved; recorded in `configs/base.yaml` |
| Block-dominance at S6 | fix designed (per-block norm targets), to be verified by V13 in Phase 2 |
| Light-cone floor L ≥ 6 for N=10 | enforced in `Config.validate()` |
| ~~Global observable is a barren plateau~~ | **corrected in Phase 4**: the earlier measurement used a product of marginals. No plateau at N=10, L≤16. Local readout kept for the light-cone reason. |
| Template-generated prose overstates probe cleanliness | accepted for now; `threads_human.jsonl` external-validity set planned for Phase 8 |
