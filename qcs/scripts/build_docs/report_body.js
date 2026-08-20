/**
 * Part 2 of the report builder: the layer-by-layer walkthrough, the dataset,
 * the build log, the results, the stack, and document assembly.
 *   node docs/report_body.js
 */
const B = require("./build_report.js");
const {
  P, H, BULLET, NUM, CODE, TABLE, IMG, CAP, RULEP, BREAK, CALLOUT, RICH, RUNIMG,
  portraitProps, landscapeProps, footer, cover, toc, part1, part2, diagA,
  CONTENT_P, CONTENT_L, INK, MUTE, STAB, PLAS, OKC, WARN,
  Document, Packer, HeadingLevel, LevelFormat, AlignmentType, fs, path,
} = B;

// ============================================================ PART 3
const layer = (id, title, purpose, io, steps, invariants, tech, result) => [
  H(`${id}  ${title}`, HeadingLevel.HEADING_2),
  RICH([{ t: "Purpose. ", b: true, c: STAB }, { t: purpose }]),
  TABLE(["", "Object", "Shape / type"], io, [16, 54, 30]),
  RICH([{ t: "What happens, step by step", b: true, c: INK }], { after: 60 }),
  ...steps.map(s => NUM(s)),
  RICH([{ t: "Invariants enforced here", b: true, c: INK }], { after: 60 }),
  ...invariants.map(s => BULLET(s)),
  RICH([{ t: "Technology. ", b: true, c: STAB }, { t: tech }]),
  RICH([{ t: "Measured result. ", b: true, c: OKC }, { t: result }]),
  RULEP(),
];

const part3 = [
  H("3  The architecture, layer by layer", HeadingLevel.HEADING_1),
  P("This section walks the full path of a single question through the system. Each layer is described by what enters it, exactly what it does to that object, what leaves, which invariants it is responsible for, the technology it runs on, and what was actually measured at that point."),
  P("The reference configuration throughout: N = 10 qubits, Hilbert dimension d = 2¹⁰ = 1024, encoder output 384, projection 256, candidate set M = 64 (training) or 256 (serving), token budget B = 3072."),

  ...layer("3.1", "L0 · Ingest and segmentation",
    "Turn a stream of chat events into the atomic unit of memory — the semantic unit (SU) — without losing a single character.",
    [["in", "raw thread event", "{thread_id, user_id, ts, text, kind}"],
     ["out", "semantic units", "list[SemanticUnit], 20–60 tokens each"]],
    ["Normalise whitespace. This is the only text mutation permitted anywhere in the pipeline.",
     "Split into sentences on terminal punctuation with a deterministic regex.",
     "Greedily merge sentences into units up to a target of 40 tokens, so a unit is 1–3 sentences.",
     "Attach provenance to every unit: speaker, turn index, timestamp, kind, and the fact id if the generator declared one."],
    ["Losslessness: joining a turn's units with single spaces reproduces the normalised turn exactly. Asserted by test_segment_roundtrip.",
     "Determinism: the same turn always yields the same units — no randomness, no model call."],
    "Pure Python with a compiled regex; records are typed dataclasses that are JSON-serialisable by construction, so any stage boundary can be dumped and reloaded without a custom codec.",
    "14,400 training turns → 4,800–7,200 semantic units per configuration, with round-trip equality holding on every one."),

  ...layer("3.2", "L1 · The frozen encoder",
    "Map text to a fixed semantic vector that never changes for the life of the project.",
    [["in", "unit text", "str"], ["out", "embedding", "ℝ³⁸⁴ (float32)"]],
    ["Look the text up in a disk cache keyed by sha256(text). Encoding is the most repeated expensive operation in the project and the cache survives corpus regeneration, reordering and partial reruns.",
     "On a miss, encode with sentence-transformers MiniLM-L6-v2 inside torch.no_grad().",
     "requires_grad_(False) is called on every parameter at load time, so a gradient reaching this module is impossible rather than merely unintended.",
     "Record the backend and revision in the run's env.json."],
    ["The encoder is never fine-tuned. Thread knowledge belongs in ℳ, not in encoder weights — this is failure mode F1, closed at the source.",
     "A deterministic hashing fallback exists so that structural work never blocks on a model download; which backend ran is always recorded."],
    "sentence-transformers 5.2 on torch 2.10 (CPU is sufficient), with a compressed .npz cache.",
    "384-dimensional embeddings; cache hit rate approaches 1.0 after the first pass over the corpus."),

  ...layer("3.3", "L1b · Whitening and the learned projection",
    "Condition the embedding geometry for amplitude encoding, then compress it to 256 dimensions — the last classical component that learns anything about semantics.",
    [["in", "embedding", "ℝ³⁸⁴"], ["out", "projection p", "ℝ²⁵⁶, frozen after Phase 2"]],
    ["Fit PCA whitening ONCE, on held-out threads only. Refitting mid-run would be a silent, untracked distribution shift, so the object raises if fit is called twice.",
     "Apply ẽ = W(e − μ) with shrinkage ε = 0.1 on the eigenvalues.",
     "Train a linear projection P: 256 → 256 with an InfoNCE objective — positives are (probe, required fact) pairs, negatives are hard in-thread distractors that mention the same entity but carry no usable value.",
     "Freeze P permanently. Everything that adapts after this point lives on the quantum manifold."],
    ["Fit-once, on held-out data, never online.",
     "Retention is measured on threads the projection never saw; measuring on its own training pairs reports a meaningless 1.000."],
    "numpy/scipy SVD for whitening; a torch InfoNCE loop for the projection; both persisted as .npz alongside the run.",
    "Held-out recall@20: raw 0.879 → whitened 0.887 → projected 0.996. Whitening with ε = 1e-6 instead of 0.1 collapses recall to 0.633 — the shrinkage value was chosen from a measured sweep, not defaulted."),

  ...layer("3.4", "L2 · The memory manifold ℳ",
    "Store everything, forever, in a way that cannot destroy information — the entire stability argument rests on this layer.",
    [["in", "semantic units + facts", "typed records"],
     ["out", "queryable memory", "SQLite tables + FAISS index"]],
    ["Write every unit into the episodic tier: verbatim text, speaker, turn, timestamp, token count, frozen embedding.",
     "Every K = 50 turns, run consolidation: emit summary nodes that point at their source units through summarises edges. The sources are untouched — a summary is an added index level, not a compression.",
     "Build the entity / supersession DAG: nodes for units, edges for supersedes, contradicts and shares_entity.",
     "Maintain a 64-dimensional salience vector per (thread, participant) by exponential moving average over that participant's own turns.",
     "Log every selection decision the assembler later makes, which feeds the usage features in ρ."],
    ["Append-only, enforced by two SQL triggers that abort any UPDATE or DELETE on semantic_units.",
     "Consolidation is additive: a test captures every source's text before consolidating and asserts byte-equality afterwards.",
     "Contradiction is represented, not resolved. 'Ada said March 14' and 'Ada later said June 3' both survive, joined by an edge."],
    "SQLite in WAL mode for the store, FAISS IndexFlatIP for vectors, both versioned with the run.",
    "12 threads → 7,200 units, 1,920 facts, 144 summary nodes, 9,989 edges. Three tests attempt an UPDATE, a DELETE and a silent overwrite-by-reinsert; all three are rejected."),

  ...layer("3.5", "L2b · The candidate prefilter",
    "Hand the scorer a short list. This layer sets a hard ceiling on the entire system — a fact that never enters the candidate set cannot be selected however good the scorer is.",
    [["in", "query projection", "ℝ²⁵⁶"], ["out", "candidate set", "M units, causally valid"]],
    ["Cosine search over the projected unit vectors.",
     "Apply a causality mask so only units strictly earlier than the query turn are reachable. This is enforced twice — by the mask and by an assertion — because a single leaked future candidate would silently inflate every result in the project.",
     "Return the top M with their similarities."],
    ["No future candidate is ever reachable.",
     "Exact inner-product search is used deliberately; an approximate index would confound 'the scorer missed it' with 'the index never showed it'."],
    "FAISS IndexFlatIP with an explicit boolean row mask; an IVF path exists for the Phase 9 scaling study.",
    "Ground-truth fact recall is 1.000 at M = 256 and also at M = 8. The prefilter is therefore not the bottleneck — a finding that redirected the whole evaluation toward ranking quality."),

  BREAK(),

  ...layer("3.6", "L3 · Feature assembly — the classical half of the interface",
    "Build the single 1024-dimensional vector that will become a quantum state. This is where the architecture's actual intelligence about conversation lives.",
    [["in", "p_q, p_c, context", "two ℝ²⁵⁶ vectors + metadata"],
     ["out", "x", "ℝ¹⁰²⁴ with ‖x‖₂ = 1"]],
    ["Normalise the query and candidate projections to direction only — the projection's magnitude carries no meaning here.",
     "Form the elementwise interaction p_q ⊙ p_c, which gives the circuit a per-dimension agreement signal rather than a single dot product.",
     "Build the relational block ρ (256 dims) from four families: 64 dims of participant salience, 16 of recency, 32 of graph/supersession features, 16 of speaker relations, 8 of prior-usage counts, and 120 reserved zeros.",
     "Scale each of the four blocks to a designed share of the total norm (0.35 / 0.35 / 0.20 / 0.10).",
     "Concatenate and L2-normalise onto the unit sphere S¹⁰²³."],
    ["Every ρ feature is bounded to [−1,1] BEFORE normalisation. Normalisation destroys magnitude, so a raw count or a raw age in seconds would not survive the trip into Hilbert space; magnitudes are encoded as bounded ratios or as sin/cos phases of a log-scaled quantity.",
     "Block-norm balancing is mandatory. Without it, the block that happens to arrive at the largest scale absorbs the amplitude mass.",
     "1024 = 2¹⁰ exactly. Config.validate() refuses to load any configuration where feature_dim ≠ 2^n_qubits."],
    "numpy, with hard assertions on shape and norm at the boundary; the ρ layout is documented dimension by dimension in encode/relational.py.",
    "Realised block shares land on target to four decimals (0.350 / 0.350 / 0.200 / 0.100). Participation ratio rose from 61/1024 without balancing to 189/1024 with it — a threefold increase in how much of the qubit register carries signal."),

  ...layer("3.7", "L4 · Amplitude encoding — crossing into Hilbert space",
    "Turn 1024 real numbers into the amplitudes of a ten-qubit quantum state, losslessly.",
    [["in", "x", "ℝ¹⁰²⁴, unit norm"], ["out", "|ψ₀⟩", "ℂ¹⁰²⁴ statevector"]],
    ["Verify the input is unit-norm to 1e-4 and of exactly dimension 2ᴺ; both are hard errors, not warnings.",
     "Load |ψ₀⟩ = Σⱼ xⱼ |j⟩ — amplitude j of the state is literally feature j of the vector.",
     "On hardware this is a state-preparation circuit of O(2ᴺ) depth, which is the real bottleneck for scaling beyond simulation; in simulation it is a type conversion."],
    ["Lossless: the simulated statevector equals the input to 1e-7, verified by test.",
     "A non-unit-norm input is rejected — it is not a quantum state and silently renormalising would hide an upstream bug."],
    "A custom batched statevector representation in torch, so that all M candidates are encoded and evolved in one tensor operation rather than a Python loop.",
    "Encoding error < 1e-7; total probability 1.000000; typical participation ratio 189 of 1024 basis states."),

  ...layer("3.8", "L5 · The variational quantum circuit",
    "The only component that learns online. Its geometry is the entire reason the project exists.",
    [["in", "|ψ₀⟩", "ℂ¹⁰²⁴"], ["out", "|ψ_L⟩", "ℂ¹⁰²⁴, ‖ψ‖ ≡ 1"]],
    ["Apply L = 8 brickwall layers. Each layer applies a general two-qubit gate to even nearest-neighbour pairs (0,1)(2,3)…(8,9), then to odd pairs (1,2)(3,4)…(7,8) — nine gates per layer.",
     "Each gate is the full SU(4): V(θ) = exp(−(i/2) Σ θ_αβ σ_α ⊗ σ_β) over the 15 non-identity Pauli products. Fifteen parameters per gate, 135 per layer, 1,080 in total.",
     "Every gate application is a permute–reshape–matmul on the batched statevector: one dense 4×4 by 4×(B·2^(N−2)) product.",
     "After each task, wrap every angle back into (−π, π]."],
    ["Unitarity is preserved exactly: ‖ψ‖ = 1 after every layer, verified numerically to 1e-4.",
     "Depth ≥ ⌈N/2⌉ is mandatory and is enforced by Config.validate(). A gate on qubits (0,1) cannot influence qubit 9 in fewer than five brickwall layers, and below that depth its gradient is not small but exactly zero.",
     "Angle wrapping keeps the parameter norm bounded — without it the logged weight norm would drift upward and hide the very property being demonstrated."],
    "Custom torch simulator (differentiable, batched), cross-checked against PennyLane; Qiskit and TensorCircuit-NG are installed for the hardware-parity and reference-stack comparisons.",
    "PennyLane cross-check: statevector fidelity 1.000, ⟨Z⟩ agreement < 1e-5. Gradient variance at a parameter on qubits (0,1): 3.6e-16 at depths 1–4 (exactly zero), 1.1e-5 at depth 6, flat thereafter — the light cone, measured."),

  ...layer("3.9", "L6 · Measurement and readout",
    "Collapse the quantum state to the single classical number the rest of the system consumes.",
    [["in", "|ψ_L⟩", "ℂ¹⁰²⁴"], ["out", "relevance r", "scalar in [0,1]"]],
    ["Compute the marginal ⟨Z₉⟩ = ⟨ψ_L| Z₉ |ψ_L⟩ by summing the probability tensor over all other qubits.",
     "Pass it through a learned affine sigmoid: r = σ(β⟨Z₉⟩ + b), where β and b are the only non-circuit parameters that learn.",
     "For hardware realism a shot-sampling path estimates the same quantity from a finite number of measurements."],
    ["The observable is local — a single qubit, never a global product.",
     "The readout is the decoherence boundary of the design: everything before it is a quantum amplitude, everything after it is a classical score."],
    "torch tensor reductions; a binomial sampler for the shot-noise path.",
    "Switching to the global observable ⟨Z⊗…⊗Z⟩ costs 0.092 MRR in ablation. At initialisation the readout separation between relevant and irrelevant candidates is Cohen's d ≈ −0.13, i.e. correctly nothing — training has to create it."),

  ...layer("3.10", "L7 · The stability controller",
    "Keep the plastic core from being rewritten by whatever the conversation is currently about. The quantum geometry buys plasticity and nothing else.",
    [["in", "θ, task batch", "parameters + examples"], ["out", "θ′", "updated parameters"]],
    ["Maintain a reservoir replay buffer sampled uniformly over the whole history, not a recent window — a FIFO buffer would protect against nothing.",
     "At each consolidation checkpoint, compute the diagonal of the quantum Fisher information by finite differences of state fidelity, and store θ*.",
     "Add the anchor penalty Ω = ½ Σ F_k (θ_k − θ*_k)² to the loss. On a curved parameter manifold, Euclidean EWC would measure distance in the wrong geometry.",
     "Clip per-task movement with a trust region ‖Δθ‖∞ ≤ δ and log the clip rate.",
     "Watch the score distribution with a population-stability index; when it shifts, re-checkpoint θ*."],
    ["The λ_Ω = 0 ablation exists precisely so the anchor's value is tested rather than assumed.",
     "The QFIM diagonal is subsampled (96 of 1,080 entries, mean-imputed) because the exact diagonal costs |θ| circuit evaluations and dominated the task loop."],
    "torch; the QFIM by finite-difference fidelity; PSI on binned score histograms.",
    "At 60 tasks the anchor is a small net negative (+0.022 MRR when removed) — there is nothing to protect yet, so the penalty is pure drag. Reported as measured rather than tuned away."),

  ...layer("3.11", "L8 · The context assembler",
    "Convert scores into the best possible prompt under a hard token ceiling. This is the layer that actually addresses saturation.",
    [["in", "scores r, candidates", "ℝ^M + texts"], ["out", "prompt C", "≤ B tokens"]],
    ["Count tokens exactly with the target model's own tokenizer. Estimated counts are how budgets silently overrun.",
     "Allocate reserved sub-budgets: system prompt, pinned constraints (never evictable), recent verbatim turns, and whatever remains for retrieved memory.",
     "Select greedily by marginal gain per token: gain(c|S) = r_c·v_c − λ·max_{c′∈S} cos(p_c, p_c′). The objective is monotone submodular, so greedy carries the (1 − 1/e) guarantee against the optimal selection.",
     "Apply the supersession filter: a stale unit is either dropped, or included explicitly labelled OUTDATED beside its CURRENT replacement. It is never served silently.",
     "Render with provenance tags, chronologically ordered within each block."],
    ["Σ tokens ≤ B always — asserted inside the knapsack and verified on 10,000 random cases.",
     "Context length is constant in thread length. A thread of 600 turns and a thread of 60,000 produce the same prompt size.",
     "No unlabelled stale fact ever reaches the model."],
    "Hugging Face tokenizer for exact counts; pure-numpy greedy selection; versioned prompt templates (a template change is a new experiment, not a patch).",
    "Budget never exceeded; zero unlabelled stale facts served across every arm; learned arms reach their quality ceiling at B = 128 where the unlearned baselines need B = 512."),

  ...layer("3.12", "L9 · The frozen language model",
    "Generate the answer, and change nothing.",
    [["in", "prompt C", "≤ B tokens"], ["out", "answer y", "text"]],
    ["Condition on the assembled context and generate.",
     "Decode at temperature 0 for evaluation so results are reproducible; a sampled path exists for qualitative review only."],
    ["No gradient path to these weights exists in the codebase — the training loop cannot even import the generator.",
     "The model has no privileged access to memory: if a fact is not in the prompt, it cannot be used."],
    "Any frozen generator. In this build the stand-in is ExtractiveReader, which answers only if the required fact's statement is present in the assembled context — an honest measure of what the architecture is responsible for, and one that cannot flatter it.",
    "Answer accuracy tracks context quality exactly, by construction. The limitation is that the stand-in cannot degrade when given too much context, so the saturation turnover is not observable with it."),

  ...layer("3.13", "L10 · Feedback and evaluation",
    "Close the loop and make every claim measurable.",
    [["in", "answers + ground truth", "text + fact ids"], ["out", "labels, metrics, gradients", "JSONL rows"]],
    ["Derive a binary label per candidate from the probe's required-fact set.",
     "Compute the loss ℒ = BCE(r, ℓ) + λ_Ω·Ω(θ, θ*) and backpropagate into θ, β and b only.",
     "Evaluate the current task and a sample of every past task, filling the transfer matrix A[i][j].",
     "Log one schema-guarded JSONL row per task: accuracy, loss, Tr(F), gradient norm, wrapped and unwrapped parameter norms, anchor pull, drift PSI, clip rate."],
    ["Every figure in the project is regenerable from metrics.jsonl alone. If a plot needs a number, the number is logged rather than recomputed inside plotting code.",
     "The Fisher probe batch is fixed once and never changes, because Tr(F) is only comparable across tasks and arms if the probe set is identical."],
    "JSONL with a key whitelist that raises on an unknown metric name — a typo fails loudly instead of producing an empty plot.",
    "Full metric suite logged for every task of every arm across three seeds; 26 MB of run artefacts, all regenerable."),
  BREAK(),
];

// ============================================================ PART 4
const part4 = [
  H("4  The data pipeline, end to end", HeadingLevel.HEADING_1),
  P("Figure 2 gives the whole journey in one page. What follows is the same path expressed as a worked example: a single probe turn from the corpus, followed from raw JSON to a generated answer."),
  IMG("DIAG_B_pipeline.png", CONTENT_P),
  CAP(2, "Every transformation from a raw chat turn to a generated answer, with the exact object at each stage. Colour indicates which manifold owns the stage."),
  BREAK(),

  H("4.1  A worked example", HeadingLevel.HEADING_2),
  P("Take thread train-002, turn 431. Ada asks: “Remind me — what deadline did we set for the Kestrel migration?” The required fact was stated by Bo at turn 231, and superseded by Bo at turn back-dated later in the thread."),
  CODE([
    "S0   {'thread_id':'train-002','turn_idx':431,'user_id':'Ada','kind':'probe',",
    "      'text':'Remind me — what deadline did we set for the Kestrel migration?',",
    "      'required_facts':['train-002:f014u'],'answer':'June 3'}",
    "",
    "S1   1 semantic unit  (12 tokens)",
    "S2   e   = MiniLM(text)                       -> float32[384]",
    "S3   ẽ   = W(e − μ)                           -> float32[256]",
    "S4   p_q = P ẽ                                -> float32[256]",
    "",
    "     prefilter: 64 causal candidates from 431 earlier units",
    "     the gold unit (turn 231) is present at cosine rank 3",
    "",
    "S5   for each candidate c:",
    "       x_raw = [p_q | p_c | p_q⊙p_c | ρ]      -> float32[1024]",
    "       ρ = [salience(Ada) | recency(200 turns) | graph(supersedes=1) | ...]",
    "S6   x = x_raw/‖x_raw‖₂                       -> ‖x‖₂ = 1.000000",
    "S7   |ψ₀⟩                                      -> complex64[64, 1024]",
    "S8   |ψ_L⟩ = U(θ)|ψ₀⟩   (8 layers, 72 gates)  -> complex64[64, 1024]",
    "S9   ⟨Z₉⟩                                      -> float32[64]",
    "S10  r = σ(β⟨Z₉⟩+b)                            -> float32[64] ∈ [0,1]",
    "",
    "     the superseded unit (turn 231) scores 0.11",
    "     the current unit    (turn 388) scores 0.94",
    "",
    "S11  knapsack under B=3072: 41 items selected, 2,986 tokens",
    "S12  prompt: system | pinned | selected (chronological) | recent 6 | query",
    "S13  answer: 'June 3'                          -> correct",
    "S14  labels for the next update",
    "S15  ℒ = BCE + λ_Ω Ω  ->  θ, β, b",
  ]),
  CALLOUT("What this example demonstrates",
    "Cosine similarity cannot distinguish the superseded statement from its replacement — both are near-identical text about the same entity. The 0.11 versus 0.94 gap comes entirely from the ρ block: recency plus the supersedes edge built in L2. Removing ρ costs 0.324 MRR in ablation, more than every other component combined.", OKC),
  BREAK(),
];

// ============================================================ PART 5
const part5 = [
  H("5  The dataset", HeadingLevel.HEADING_1),
  P("The corpus is synthetic and deliberately so: it must contain exact ground truth about which memories are required to answer each question. Without that, no claim about memory is falsifiable. This phase was built before any modelling, because a corpus without ground truth makes every later result unverifiable."),

  H("5.1  Generation is two steps, and the separation is the point", HeadingLevel.HEADING_2),
  CODE(["   plan  ───────────────────────────►  surface text",
        "   (structured; IS the ground truth)   (templates; may not add or remove facts)"]),
  P("The planner decides what happens on every turn before any prose exists: which turn introduces which fact, which turn supersedes it, which turn probes it, and at what distance. The realiser then writes the words. Because the plan is authoritative, the set of facts required to answer any probe is exact rather than inferred. An LLM is a drop-in replacement for the realiser, but it must be given the plan and forbidden from inventing facts."),

  H("5.2  Structure", HeadingLevel.HEADING_2),
  TABLE(["Turn kind", "Meaning", "Role in evaluation"], [
    ["fact", "a participant states an (entity, attribute, value) fact they own", "the retrieval target"],
    ["update", "a participant supersedes an earlier fact", "creates the stale/current trap"],
    ["probe", "a participant asks a question answerable only from stated facts", "the measurement point"],
    ["distractor", "mentions a REAL entity with no usable value", "lexically close noise — defeats keyword matching"],
    ["filler", "ordinary conversation", "dilution"],
  ], [16, 48, 36]),
  P("Three regimes cycle across threads: single_owner (all facts from one participant), multi_owner (facts spread across participants and probes asked by someone else, which is what makes persona bleed measurable), and adversarial (multi_owner plus supersession chains)."),

  H("5.3  The corpus actually generated", HeadingLevel.HEADING_2),
  TABLE(["Quantity", "Value"], [
    ["threads", "24 train / 6 test (test sealed until Phase 8)"],
    ["turns", "14,400 train / 3,600 test"],
    ["probes", "1,440 train / 360 test"],
    ["facts", "1,920 including supersession replacements"],
    ["participants per thread", "5"],
    ["probe distance bands", "10 / 50 / 200 / 500 turns"],
    ["supersession density", "0.35 of facts in adversarial threads"],
    ["oracle recall", "1.000"],
    ["random recall", "0.042"],
  ], [40, 60], { boldCol: 0 }),

  H("5.4  Validation — seven invariants", HeadingLevel.HEADING_2),
  P("These are not style checks. Each corresponds to a way the corpus could silently stop being a valid measuring instrument."),
  TABLE(["Id", "Invariant"], [
    ["V1", "every probe's required facts appear strictly earlier in the same thread"],
    ["V2", "no probe requires a superseded fact — otherwise 'correct' is ambiguous"],
    ["V3", "fact ids are unique corpus-wide"],
    ["V4", "turn indices are contiguous; speakers are inside the participant set"],
    ["V5", "probe distances actually cover the configured bands"],
    ["V6", "no thread appears in both splits"],
    ["V7", "an oracle retriever scores 1.0 and a random one scores near chance"],
  ], [10, 90]),
  P("A test feeds the validator a deliberately broken corpus and asserts that it raises, because a validator that never fails is not a validator."),

  H("5.5  Two corpus changes that measurement forced", HeadingLevel.HEADING_2),
  ...[
    "One fact per entity made retrieval trivial. The first build hit cosine recall 1.000 at M = 8 because each fact had a unique entity name, so the entity string was effectively a primary key. Facts now share entities (about three attributes each), so the retriever must resolve the attribute as well.",
    "Fact-introduction turns were crushed into the opening 120 turns of a 600-turn thread, because the planner placed facts first and chose probe distances afterwards. Recency alone would have separated facts from filler. The planner now assigns the probe distance band first, then places the fact anywhere that still leaves room for its probe.",
  ].map(t => BULLET(t)),
  BREAK(),
];

// ============================================================ PART 6
const part6 = [
  H("6  Build log — nine phases, and what each one measured", HeadingLevel.HEADING_1),
  P("Each phase has an exit criterion that the next phase depends on. Nothing later was started while an earlier criterion was red."),
  TABLE(["Phase", "Objective", "Exit criterion", "Result"], [
    ["0 · Foundations", "make every experiment reproducible by someone else", "identical seeds → identical output", "7 tests; config validates architecture invariants as assertions"],
    ["1 · Dataset", "ground truth by construction", "oracle 1.00, random ≈ chance", "1.000 / 0.042 — instrument calibrated"],
    ["2 · Encoding", "frozen, well-conditioned text → ℝ¹⁰²⁴", "unit-norm vectors for every triple", "held-out recall@20 = 0.996; block shares on target"],
    ["3 · Memory ℳ", "storage that cannot destroy", "append-only proven; recall ≥ 0.95 at M=256", "recall 1.000; 3 mutation attempts rejected by the DB"],
    ["4 · Quantum core", "a correct, differentiable circuit", "gradient agreement < 1e-6", "8.6e-9; PennyLane cross-check passes"],
    ["5 · Baselines", "know what must be beaten", "each arm at its own best lr", "tuning moved the linear arm 0.063 → 0.965 MRR"],
    ["6 · Continual training", "the actual experiment", "full metric suite, both arms", "120 tasks × 3 arms; Tr(F) diverges as predicted"],
    ["7 · Assembly", "budgeted context", "budget never exceeded", "0 unlabelled stale facts; 4× budget reduction"],
    ["8 · Evaluation", "results that survive review", "every claim has a figure", "3 seeds, 7 ablations, noise sweep, sealed test split"],
    ["9 · Hardening", "clean-clone reproduction", "a colleague reproduces the table", "not started"],
  ], [16, 26, 26, 32]),

  H("6.1  Findings that changed the design", HeadingLevel.HEADING_2),
  P("Seven measurements altered the architecture rather than confirming it. They are listed because they are the most useful part of the record."),
  ...[
    "Python's hash() is salted per process, so hash-derived seeds would have produced a different corpus on every run. Switched to hashlib — the class of bug that makes a project irreproducible without anyone noticing.",
    "Whitening with ε = 1e-6 destroyed retrieval (recall 0.879 → 0.633) by amplifying low-variance noise directions. Shrinkage ε = 0.1 fixed it and lifted projected recall to 0.996.",
    "Unbalanced feature blocks let the relational block absorb the amplitude mass; participation ratio was 61 of 1024 basis states. Per-block norm targets raised it to 189.",
    "The two-term parameter-shift rule is invalid for the exponential gate form, because the 15 generators do not commute — measured discrepancy 3.3e-3. A product gate form was added where the rule is exact (8.6e-9).",
    "An earlier claim that the global observable is a barren plateau was wrong: the code had computed a product of marginals. With the true parity observable there is no plateau at N = 10 and L ≤ 16. The local readout is still correct, but for the light-cone reason.",
    "Running the quantum arm at lr 0.03 and the classical arm at 1e-3 — the reference paper's protocol — is a handicap, not a comparison. Per-arm tuning moved the classical twin from 0.063 to 0.965 MRR.",
    "Tagging ablation runs broke arm-name parsing, so every ablation trained the quantum circuit at the classical learning rate and the first ablation table was invalid. Model family is now an explicit argument.",
  ].map(t => BULLET(t)),
  BREAK(),
];

// ============================================================ PART 7
const part7 = [
  H("7  Results", HeadingLevel.HEADING_1),
  P("All numbers below come from the accompanying code. Training and evaluation used three seeds (7, 17, 27) over a 120-task stream, with every arm at its own tuned learning rate."),

  H("7.1  Headline comparison", HeadingLevel.HEADING_2),
  TABLE(["Arm", "|θ|", "rank-1", "MRR", "AUC", "stale-above ↓", "wall clock"], [
    ["quantum (QRE)", "1,080", "0.928 ± 0.038", "0.959 ± 0.022", "0.998", "0.015", "406 s"],
    ["linear (parameter-matched)", "1,025", "0.979 ± 0.002", "0.990 ± 0.001", "1.000", "0.000", "3 s"],
    ["mlp (capacity reference)", "65,665", "0.996 ± 0.002", "0.998 ± 0.001", "1.000", "0.000", "20 s"],
    ["cosine (B1, no learning)", "0", "0.894", "0.943", "0.932", "0.078", "—"],
    ["recency (B0, truncation)", "0", "0.025", "0.124", "0.532", "0.521", "—"],
    ["oracle (ceiling)", "—", "1.000", "1.000", "1.000", "0.000", "—"],
  ], [26, 10, 14, 14, 10, 14, 12], { boldCol: 0 }),
  RUNIMG("V46", CONTENT_P),
  CAP(3, "Headline metrics with standard deviation across three seeds. The quantum arm's spread is roughly nineteen times the linear arm's."),

  H("7.2  The sealed test split", HeadingLevel.HEADING_2),
  P("Six threads were held out from the moment of generation and opened once, at this point, for the first time."),
  TABLE(["Arm", "MRR", "rank-1", "AUC", "stale-above ↓"], [
    ["quantum", "0.919", "0.856", "0.996", "0.035"],
    ["linear", "0.975", "0.953", "0.999", "0.000"],
    ["mlp", "0.993", "0.986", "1.000", "0.000"],
    ["cosine", "0.910", "0.831", "0.928", "0.130"],
    ["recency", "0.128", "0.028", "0.530", "0.478"],
  ], [30, 18, 18, 17, 17], { boldCol: 0 }),
  P("Held-out numbers track the training-stream numbers closely, so the ranking is not an artefact of evaluating where the models trained. Per regime every arm stays within ±0.04 across single-owner, multi-owner and adversarial threads — no persona-bleed collapse."),

  H("7.3  Significance", HeadingLevel.HEADING_2),
  P("Paired bootstrap over threads, 5,000 resamples. Threads are the independent unit; resampling individual probes would shrink the interval by roughly the square root of probes-per-thread and manufacture significance."),
  TABLE(["Comparison", "Δ MRR", "95% CI", "p", "Verdict"], [
    ["quantum − linear", "−0.056", "[−0.070, −0.043]", "0.000", "significant"],
    ["quantum − mlp", "−0.074", "[−0.086, −0.060]", "0.000", "significant"],
    ["linear − mlp", "−0.018", "[−0.023, −0.012]", "0.000", "significant"],
  ], [28, 14, 26, 12, 20], { boldCol: 0 }),
  CALLOUT("The honest headline",
    "The quantum arm is significantly worse than its parameter-matched classical twin on ranking quality, at roughly 135× the compute and 19× the seed-to-seed variance. The blueprint required this comparison to be falsifiable; it falsified in the classical arm's favour at this scale and horizon. That is reported as the result rather than buried.", PLAS),
  RUNIMG("V40", Math.round(CONTENT_P * 0.78)),
  CAP(4, "Paired bootstrap over threads. A confidence interval that excludes zero is a real difference; all three pairs are significant."),
  BREAK(),

  H("7.4  Ablations — what each component is actually worth", HeadingLevel.HEADING_2),
  P("Quantum arm, 60 tasks, seed 7, every variant retrained from scratch at the tuned learning rate."),
  TABLE(["Variant", "MRR", "Δ from full system"], [
    ["no ρ block at all", "0.572", "−0.324"],
    ["no replay", "0.733", "−0.162"],
    ["no graph / supersession block", "0.765", "−0.130"],
    ["global observable ⟨Z⊗…⊗Z⟩", "0.804", "−0.092"],
    ["no trust region", "0.896", "±0.000"],
    ["no salience block", "0.897", "+0.001"],
    ["no anchor (λ_Ω = 0)", "0.917", "+0.022"],
    ["full system", "0.896", "—"],
  ], [50, 22, 28], { boldCol: 0 }),
  RUNIMG("V35", CONTENT_P),
  CAP(5, "Ablation effects, sorted by size. The relational block is worth more than every other component combined."),
  P("Four conclusions follow. First, the ρ block is the architecture — removing it costs 32 MRR points, and the relational features are what the scorer is actually using. Second, replay and the supersession graph both earn their place. Third, the local readout is validated on task performance, not merely on trainability theory. Fourth, the QFIM anchor currently hurts slightly: at 60 tasks there is nothing to protect, so a Fisher-weighted penalty is pure drag. Its value can only appear over horizons long enough for forgetting to occur, which this run does not reach."),

  H("7.5  Plasticity — the metric the reference paper is about", HeadingLevel.HEADING_2),
  P("Normalised Tr(F) is effective learning capacity. Over 120 tasks the quantum arm rises to roughly 2.5× its initial value and holds; the linear arm decays to 0.30× and the MLP collapses to 0.06× after task 80. Gradient norms agree (quantum 1.2×, classical 0.2–0.35×), and the unwrapped parameter norm of the linear arm grows to 5× and is still climbing while the quantum arm plateaus."),
  RUNIMG("V21", CONTENT_P),
  CAP(6, "Training curves. Top right is the decisive panel: capacity preserved on the compact manifold, capacity lost in Euclidean space."),
  CALLOUT("What this does and does not mean",
    "Losing Tr(F) is losing the ability to keep learning, not current accuracy — both classical arms are still near ceiling because 120 tasks is far too short for capacity loss to surface as error. The reference paper needed 1,000–3,000 tasks. The plasticity claim is therefore neither confirmed nor refuted here; this is an early indicator pointing the way the paper predicts, and the quantum arm's ~4 s per task is the binding constraint on running the experiment that would settle it.", STAB),
  BREAK(),

  H("7.6  Noise robustness", HeadingLevel.HEADING_2),
  P("Depolarising noise applied per gate by trajectory sampling on the trained circuit."),
  TABLE(["p per gate", "0", "1e-4", "5e-4", "1e-3", "5e-3"], [
    ["MRR", "0.908", "0.908", "0.904", "0.904", "0.887"],
    ["rank-1", "0.850", "0.850", "0.850", "0.850", "0.825"],
    ["AUC", "0.994", "0.994", "0.993", "0.993", "0.991"],
  ], [22, 16, 16, 16, 15, 15], { boldCol: 0 }),
  P("Ranking is essentially unchanged through p = 1e-3, an order of magnitude beyond the 5e-4 the reference paper reports, and loses two points at 5e-3. This is the one quantum-side claim that came through cleanly."),

  H("7.7  Failure taxonomy", HeadingLevel.HEADING_2),
  P("Every loss on the test split, categorised in pipeline order — the first category that applies wins, because a retrieval miss makes every downstream question moot."),
  TABLE(["Arm", "accuracy", "retrieval", "ranking", "stale", "eviction"], [
    ["mlp", "0.986", "0", "5", "0", "0"],
    ["linear", "0.953", "0", "17", "0", "0"],
    ["quantum", "0.856", "0", "48", "4", "0"],
    ["cosine", "0.831", "0", "47", "14", "0"],
  ], [22, 18, 15, 15, 15, 15], { boldCol: 0 }),
  P("Zero retrieval failures and zero evictions for every arm: the memory manifold and the knapsack are not the bottleneck, and every remaining loss is a ranking decision. Cosine's losses are disproportionately stale-fact errors, which the learned arms almost eliminate."),
  BREAK(),
];

// ============================================================ PART 8-9
const part8 = [
  H("8  The technology stack", HeadingLevel.HEADING_1),
  P("Figure 7 lists what every layer runs on and, more usefully, why that choice rather than the obvious default."),
  IMG("DIAG_D_stack.png", CONTENT_P),
  CAP(7, "The stack, layer by layer, with the reasoning behind each choice."),

  H("8.1  Repository layout", HeadingLevel.HEADING_2),
  CODE([
    "qcs/",
    "├── configs/         base.yaml + tuned.yaml — every tunable number",
    "├── data/            raw (immutable) · interim (caches) · processed (frozen artefacts)",
    "├── runs/            one directory per experiment: config, env, metrics.jsonl, figs",
    "├── docs/            this report and its build scripts",
    "├── figs/            architecture diagrams",
    "├── scripts/         thin CLIs — argument parsing only",
    "│     gen_dataset · fit_encoder · study_whitening · build_memory",
    "│     study_trainability · tune_lr · train_continual · run_assembly · run_eval",
    "├── src/qcs/",
    "│   ├── config.py seeding.py logging_.py types.py paths.py",
    "│   ├── datagen/    fact_bank · thread_planner · surface_realizer · validate · stats",
    "│   ├── ingest/     segment · loader",
    "│   ├── encode/     encoder · cache · whiten · projection · relational · assemble · pipeline",
    "│   ├── memory/     schema.sql · store · ann_index · consolidate · graph · salience",
    "│   ├── quantum/    gates · encoding · ansatz · readout · circuit · gradients · fisher · noise",
    "│   ├── classical/  scorers (linear · mlp · cosine · recency · random · oracle)",
    "│   ├── train/      dataset · replay · anchor · loop",
    "│   ├── assemble/   budget · knapsack · pipeline",
    "│   ├── eval/       metrics · headroom · significance · failure_analysis",
    "│   └── viz/        style + one module per phase (45 figure ids)",
    "└── tests/           88 tests across 5 files",
  ]),

  H("8.2  Reproducing the results", HeadingLevel.HEADING_2),
  CODE([
    "python -m pip install -r requirements.txt",
    "python -m pip install -e .",
    "",
    "python scripts/gen_dataset.py            # corpus + V01–V04",
    "python scripts/fit_encoder.py            # frozen encoding stack + V05–V08, V13",
    "python scripts/build_memory.py           # memory manifold + V09–V12",
    "python scripts/study_trainability.py     # quantum core + V14–V20",
    "python scripts/tune_lr.py                # per-arm learning rates -> configs/tuned.yaml",
    "python scripts/train_continual.py --config tuned",
    "python scripts/run_assembly.py --checkpoints runs/<phase6>/checkpoints",
    "python scripts/run_eval.py --seeds 7 17 27",
    "python -m pytest                          # 88 tests",
  ]),
  BREAK(),
];

const part9 = [
  H("9  Limitations, and what would settle the open question", HeadingLevel.HEADING_1),
  P("Stated plainly, because a report that hides these is not useful to the person who has to continue the work."),
  TABLE(["Limitation", "Consequence", "What would resolve it"], [
    ["120 tasks is two orders of magnitude short of the reference paper's horizon", "the plasticity claim is untested — neither confirmed nor refuted", "a 1,000–3,000 task run; roughly one hour per 1,000 tasks per seed at current speed"],
    ["The quantum arm is ~135× slower per task than its classical twin", "it is the binding constraint on the long-horizon experiment", "batched multi-candidate state preparation; GPU backend; or a shallower ansatz"],
    ["The generator is an extractive stand-in, not a real frozen LLM", "attention dilution cannot occur, so the saturation turnover is unobservable", "wire in any frozen chat model and re-run the budget sweep"],
    ["The corpus is template-generated", "probe cleanliness is overstated relative to real threads", "the planned human-written thread set for external validity"],
    ["Amplitude encoding is O(2ᴺ) deep on hardware", "the encoding, not the ansatz, is the hardware bottleneck", "angle encoding with data re-uploading for the hardware path"],
    ["Simulation is exact only to ~20 qubits", "the design cannot be scaled arbitrarily on a laptop", "the Phase 9 scaling study across N ∈ {8, 10, 12}"],
  ], [30, 32, 38]),

  H("9.1  What this project does establish", HeadingLevel.HEADING_2),
  ...[
    "Architecture claims are supported: append-only memory plus a prefilter plus budgeted assembly produce zero retrieval failures and zero evictions; the relational block is worth 0.324 MRR; stale-fact errors fall from 0.130 under cosine to 0.000–0.035 with a learned scorer; no unlabelled stale fact ever reaches a prompt.",
    "The quantum ranking advantage is refuted at this scale, significantly and reproducibly across three seeds.",
    "Noise robustness is supported: ranking survives depolarising noise an order of magnitude beyond the level the reference paper reports.",
    "The plasticity question remains genuinely open, with a measured early indicator in its favour and a clear, costed experiment that would settle it.",
  ].map(t => BULLET(t)),

  H("9.2  Phase 9, the remaining work", HeadingLevel.HEADING_2),
  ...[
    "The long-horizon run: 1,000+ tasks, three seeds, both arms, with Tr(F) and the plasticity index as primary endpoints.",
    "Hardware parity: run the trained circuit shot-based and on a real backend; report the shot count at which score ranking becomes stable.",
    "A scaling study across N ∈ {8, 10, 12} qubits with wall-clock reported honestly.",
    "A real frozen LLM in place of the extractive reader, and the saturation sweep repeated.",
    "A reproduction kit: make reproduce regenerating every figure from raw logs.",
  ].map(t => NUM(t)),
  RULEP(),
  P("Report generated from the repository state at the end of Phase 8. Every table and figure in §7 is regenerable from runs/*/metrics.jsonl by scripts/make_figures.py and scripts/run_eval.py.", { italics: true, color: MUTE, size: 19 }),
];

// ============================================================ ASSEMBLE
const numbering = {
  config: [
    {
      reference: "bullets",
      levels: [
        { level: 0, format: LevelFormat.BULLET, text: "•", alignment: AlignmentType.LEFT,
          style: { paragraph: { indent: { left: 360, hanging: 200 } } } },
        { level: 1, format: LevelFormat.BULLET, text: "–", alignment: AlignmentType.LEFT,
          style: { paragraph: { indent: { left: 720, hanging: 200 } } } },
      ],
    },
    {
      reference: "numbers",
      levels: [
        { level: 0, format: LevelFormat.DECIMAL, text: "%1.", alignment: AlignmentType.LEFT,
          style: { paragraph: { indent: { left: 360, hanging: 240 } } } },
      ],
    },
  ],
};

const styles = {
  default: {
    document: { run: { font: "Calibri", size: 21, color: INK } },
    heading1: { run: { font: "Calibri Light", size: 34, bold: true, color: INK },
                paragraph: { spacing: { before: 400, after: 160 } } },
    heading2: { run: { font: "Calibri Light", size: 26, bold: true, color: STAB },
                paragraph: { spacing: { before: 300, after: 120 } } },
    heading3: { run: { font: "Calibri", size: 22, bold: true, color: MUTE },
                paragraph: { spacing: { before: 220, after: 100 } } },
  },
};

const doc = new Document({
  creator: "Dual-Manifold ContextSync",
  title: "Dual-Manifold ContextSync — Technical Report",
  description: "Quantum-classical continual learning for long multi-user chat threads",
  numbering, styles,
  sections: [
    { properties: portraitProps, footers: { default: footer },
      children: [...cover, ...toc, ...part1, ...part2] },
    { properties: landscapeProps, footers: { default: footer }, children: diagA },
    { properties: portraitProps, footers: { default: footer }, children: part3 },
    { properties: landscapeProps, footers: { default: footer },
      children: [IMG("DIAG_C_circuit.png", CONTENT_L),
                 CAP(8, "The variational circuit in detail: amplitude encoding, eight brickwall layers of general SU(4) gates, and a local readout on qubit 9.")] },
    { properties: portraitProps, footers: { default: footer },
      children: [...part4, ...part5, ...part6, ...part7, ...part8, ...part9] },
  ],
});

Packer.toBuffer(doc).then(buf => {
  // the previous file may be open in Word or mid-sync; never fail on a lock
  const base = process.argv[2] || "Dual_Manifold_ContextSync_Report.docx";
  let out = path.join(__dirname, "..", "..", "deliverables", base);
  try { fs.writeFileSync(out, buf); }
  catch (e) {
    out = out.replace(/\.docx$/, "_v2.docx");
    fs.writeFileSync(out, buf);
    console.log("(original was locked — wrote a new copy)");
  }
  console.log("wrote", out, (buf.length / 1024).toFixed(0) + " KB");
  return;
});
