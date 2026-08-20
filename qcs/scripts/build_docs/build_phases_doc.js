/**
 * "Ten Phases: from raw dataset to a stable, plastic system"
 *   node docs/build_phases_doc.js
 */
const B = require("./build_report.js");
const {
  P, H, BULLET, NUM, CODE, TABLE, IMG, CAP, RULEP, BREAK, CALLOUT, RICH, RUNIMG,
  portraitProps, landscapeProps, footer,
  CONTENT_P, CONTENT_L, INK, MUTE, STAB, PLAS, OKC, WARN,
  Document, Packer, Paragraph, TextRun, HeadingLevel, LevelFormat, AlignmentType, fs, path,
} = B;

// ---------------------------------------------------------------- phase block
const phase = (n, title, objective, dataIn, dataOut, work, artefacts, result, stability, plasticity) => [
  H(`Phase ${n} · ${title}`, HeadingLevel.HEADING_2),
  RICH([{ t: "Objective. ", b: true, c: STAB }, { t: objective }]),
  TABLE(["Dataset entering this phase", "Dataset leaving this phase"],
    [[dataIn, dataOut]], [50, 50]),
  RICH([{ t: "What the phase does to the data", b: true }], { after: 60 }),
  ...work.map(w => NUM(w)),
  RICH([{ t: "Artefacts written to disk", b: true }], { after: 60 }),
  ...artefacts.map(a => BULLET(a)),
  RICH([{ t: "Measured outcome. ", b: true, c: OKC }, { t: result }]),
  TABLE(["Contribution to STABILITY", "Contribution to PLASTICITY"],
    [[stability, plasticity]], [50, 50]),
  RULEP(),
];

// ================================================================ front matter
const cover = [
  new Paragraph({ spacing: { before: 2100, after: 0 },
    children: [new TextRun({ text: "COMPANION VOLUME", size: 20, color: PLAS, bold: true, font: "Calibri" })] }),
  new Paragraph({ spacing: { before: 200, after: 0 },
    children: [new TextRun({ text: "Ten Phases", size: 62, bold: true, color: INK, font: "Calibri Light" })] }),
  new Paragraph({ spacing: { before: 120, after: 240 },
    children: [new TextRun({ text: "How the dataset is built, transformed and turned into an answer — and exactly where stability and plasticity come from",
      size: 26, color: MUTE, font: "Calibri Light" })] }),
  RULEP(),
  P("Each phase is described twice: once as a transformation of the dataset, and once as a contribution to the two properties the whole project exists to reconcile. Every measured number comes from the accompanying code.", { size: 21 }),
  BREAK(),
];

const toc = [
  H("Contents", HeadingLevel.HEADING_1),
  P("If the list below is empty, click it and press F9 — Word builds it on first open.",
    { italics: true, color: MUTE, size: 19 }),
  new (require("docx").TableOfContents)("Contents", { hyperlink: true, headingStyleRange: "1-2" }),
  BREAK(),
];

// ================================================================ part 1
const part1 = [
  H("1  The two properties, defined so they can be measured", HeadingLevel.HEADING_1),
  P("Stability and plasticity are usually discussed as intuitions. Nothing in this project relies on an intuition: both are given an operational definition, an enforcement mechanism, and an instrument that measures them."),

  H("1.1  Stability", HeadingLevel.HEADING_2),
  RICH([{ t: "Definition used here. ", b: true },
        { t: "A fact stated at turn t remains answerable at every later turn, for the life of the thread." }]),
  P("Note what this does NOT say. It does not say the model's weights must not move. It says the information must survive. That distinction is the whole architecture: knowledge is held outside the learned parameters, so learning cannot destroy it."),
  TABLE(["Enforced by", "Instrument", "Failure would look like"], [
    ["An append-only store; two SQL triggers abort any UPDATE or DELETE", "Memory growth curve (V09) — must be monotone non-decreasing", "A downward step in the growth curve"],
    ["A frozen generator with no gradient path in the codebase", "Capability benchmark before and after a long run", "A drop in general ability after training"],
    ["Reservoir replay + a Fisher-weighted anchor on the scorer", "Backward transfer (BWT) over the task sequence", "BWT strongly negative"],
    ["Consolidation that adds summaries beside sources, never replacing them", "Byte-equality test on every source unit after consolidation", "An orphaned summary, or a modified source"],
  ], [34, 33, 33]),

  H("1.2  Plasticity", HeadingLevel.HEADING_2),
  RICH([{ t: "Definition used here. ", b: true },
        { t: "The system's capacity to learn from task 900 is the same as its capacity to learn from task 1." }]),
  P("Again, note what this does not say. It is not about current accuracy. A model can be highly accurate and have already lost its plasticity — the parameters still exist but have become functionally dead, and the loss only becomes visible much later when the data shifts. The instrument therefore measures capacity, not performance."),
  TABLE(["Enforced by", "Instrument", "Failure would look like"], [
    ["Parameters on a compact periodic manifold — angles wrap into (−π, π]", "Unwrapped parameter norm over time", "A norm that grows without bound"],
    ["Unitary evolution: ‖U(θ)‖₂ = 1 for every θ", "Gradient norm, normalised to the first tasks", "Gradient norm decaying toward zero"],
    ["A Fisher trace bounded above and below (Haar average)", "Tr(F) on a fixed probe batch — the primary endpoint", "Tr(F) collapsing over the sequence"],
    ["Depth above the light-cone floor, so no parameter is structurally dead", "Gradient variance versus depth (V17)", "Exactly-zero gradients on part of the circuit"],
  ], [34, 33, 33]),

  CALLOUT("Why the two need different machinery",
    "The reference paper (Chen & Zhang, PRX Quantum 7, 033003) shows that a quantum ansatz preserves plasticity intrinsically — and states explicitly that it does NOT solve stability. So the quantum manifold is used for the component that must keep learning, and classical append-only storage for the component that must never be lost. Any design that expects one mechanism to deliver both is unsound.", PLAS),
  BREAK(),
];

// ================================================================ part 2
const part2 = [
  H("2  The ten phases", HeadingLevel.HEADING_1),
  P("Phases run in order because each one's exit criterion is the next one's precondition. For every phase below: what the dataset looks like going in, what happens to it, what comes out, what was measured, and which of the two properties it serves."),

  ...phase(0, "Foundations",
    "Make randomness a controlled variable, so that every later measurement is repeatable by someone else.",
    "nothing — no data exists yet",
    "an empty but reproducible substrate: config, seeding, run logger, typed records",
    ["Define every tunable number in one YAML file; no algorithm anywhere is allowed a magic constant.",
     "Seed python, numpy and torch from a single call, and derive per-thread sub-streams with hashlib rather than Python's hash(), which is salted per process.",
     "Create the run logger: one directory per experiment holding the resolved config, the environment, and a schema-guarded metrics file.",
     "Encode the architecture's hard constraints as assertions inside Config.validate() so an invalid configuration cannot be loaded at all."],
    ["configs/base.yaml", "runs/<id>/config.yaml and env.json", "src/qcs/{config,seeding,logging_,types}.py"],
    "Two runs with the same seed produce identical output; 7 tests, including two that assert the validator rejects bad configurations.",
    "Provenance is recorded for every artefact, so a stored fact can always be traced to the run that produced it.",
    "The light-cone floor (depth ≥ ⌈N/2⌉) is enforced here, which is what prevents a circuit with structurally untrainable parameters."),

  ...phase(1, "Dataset generation",
    "Build a corpus in which the correct answer to “which memories were required?” is known exactly, not inferred.",
    "nothing — the corpus is generated",
    "18,000 turns across 30 threads, with 1,800 probes and 1,920 facts, plus a sealed test split",
    ["Build a fact bank: atomic (entity, attribute, value) facts with an owner, roughly three attributes per entity so the entity name alone cannot identify the answer.",
     "Plan every thread before any prose exists: assign each fact a probe-distance band first (10 / 50 / 200 / 500 turns), then place the fact anywhere that still leaves room for its probe.",
     "In adversarial threads, add supersession chains — a later turn overwrites an earlier fact, and only the current value counts as correct.",
     "Realise the surface text from the plan with deterministic templates, so the prose can never add or remove a fact.",
     "Interleave filler and distractors: distractors mention a real entity but carry no usable value, which defeats keyword matching.",
     "Validate seven structural invariants and calibrate the instrument with an oracle and a random retriever."],
    ["data/raw/fact_bank.jsonl", "data/raw/threads_train.jsonl", "data/raw/threads_test.jsonl (sealed until Phase 8)", "figures V01–V04"],
    "Oracle recall 1.000, random recall 0.042 — the measuring instrument is calibrated. A test feeds the validator a deliberately broken corpus and asserts that it raises.",
    "This phase creates the notion of a fact that must survive. Without exact ground truth, stability is an opinion; with it, forgetting becomes a countable event.",
    "The probe-distance bands and the supersession chains are what make the task non-stationary. A stream that never changes cannot exercise plasticity at all."),

  ...phase(2, "Encoding",
    "Turn text into a unit-norm 1024-dimensional vector — the classical half of the quantum interface.",
    "18,000 raw turns",
    "7,200 semantic units, each with a frozen 384-d embedding and a frozen 256-d projection",
    ["Segment each turn into semantic units of 20–60 tokens; concatenating a turn's units reproduces the turn exactly.",
     "Embed with a frozen MiniLM encoder under no_grad, cached by sha256 of the text.",
     "Fit PCA whitening once, on held-out threads only, with shrinkage ε = 0.1.",
     "Train a linear projection with InfoNCE — positives are (probe, required fact), negatives are hard in-thread distractors — then freeze it permanently.",
     "Assemble the four-block feature vector and balance each block to a designed share of the norm before the final L2 normalisation."],
    ["data/processed/whiten.npz", "data/processed/projection.npz", "the embedding cache", "figures V05–V08, V13"],
    "Held-out recall@20 rose 0.879 (raw) → 0.887 (whitened) → 0.996 (projected). Block shares landed on target to four decimals; participation ratio 189 of 1024 basis states, up from 61 without balancing.",
    "Every component here is frozen after this phase. A frozen encoder cannot drift, so a vector stored in month one still means the same thing in month six — retrieval stays valid indefinitely.",
    "Freezing the semantics is what leaves the quantum layer as the only thing that adapts. Plasticity is concentrated in 1,080 angles instead of being spread across millions of weights that would inflate."),

  ...phase(3, "The memory manifold",
    "Store everything, forever, in a way that cannot destroy information.",
    "7,200 encoded semantic units",
    "a queryable store: 7,200 units, 1,920 facts, 144 summaries, 9,989 graph edges, per-user salience",
    ["Write every unit into the episodic tier verbatim, with speaker, turn, timestamp and token count.",
     "Install two SQL triggers that abort any UPDATE or DELETE on the unit table.",
     "Every 50 turns, consolidate: emit summary nodes that point at their sources through edges — the sources are never touched.",
     "Build the entity graph with supersedes, contradicts and shares_entity edges, so contradiction is stored rather than resolved.",
     "Maintain a 64-dimensional salience vector per participant by exponential moving average over that person's own turns.",
     "Index the projections for cosine retrieval with a causality mask that makes future turns unreachable."],
    ["data/processed/qcs.db", "the FAISS index", "figures V09–V12, V10b"],
    "Prefilter recall of the required fact: 1.000 at M = 256 and also at M = 8. Three tests attempt to mutate the store — an UPDATE, a DELETE and a silent overwrite — and all three are rejected by the database.",
    "This is where stability physically lives. Forgetting is reduced from an irreversible gradient event to a retrieval miss, which is bounded, measurable and fixable without retraining anything.",
    "By absorbing the memory burden entirely, this phase frees the learned component to move as much as it needs to. The scorer never has to preserve knowledge, only ranking skill."),

  BREAK(),

  ...phase(4, "The quantum core",
    "Build a correct, differentiable circuit and choose its depth from evidence.",
    "unit-norm 1024-dimensional feature vectors",
    "a verified 10-qubit variational circuit with 1,080 parameters",
    ["Implement the general SU(4) gate as the exponential of the 15-generator Pauli sum, and a product form where the parameter-shift rule is exact.",
     "Amplitude-encode the feature vector: 1024 real numbers become 1024 amplitudes, losslessly, because 1024 = 2¹⁰.",
     "Arrange 8 brickwall layers of 9 gates and read out ⟨Z⟩ on a single qubit through a learned affine sigmoid.",
     "Cross-check the whole circuit against PennyLane, an independent simulator.",
     "Measure gradient variance versus depth to locate the light cone, and verify the two gradient paths agree."],
    ["src/qcs/quantum/*", "figures V14–V20", "figs/CIRCUIT_main.png and CIRCUIT_pennylane.png"],
    "PennyLane agreement: state fidelity 1.000, ⟨Z⟩ within 1e-5. Autodiff versus parameter shift: 8.6e-9 in the product form. Gradient variance 3.6e-16 at depths 1–4 (exactly zero) and 1.1e-5 from depth 6 — the light cone, measured.",
    "The readout is a single scalar, which keeps the quantum component off the generation path entirely. Nothing the circuit does can corrupt stored memory.",
    "This phase creates the compact manifold itself. Unitarity bounds the operator norm at 1 and the angles wrap, which is the structural reason capacity cannot decay."),

  ...phase(5, "Baselines and the classical twin",
    "Know exactly what must be beaten, with classical arms built with equal care.",
    "the encoded corpus",
    "six comparison arms behind one identical scoring interface",
    ["Implement a parameter-matched linear scorer (1,025 parameters against the circuit's 1,080), a wider MLP as a capacity reference, and cosine, recency, random and oracle baselines.",
     "Force every arm through the same interface: score(X) → r, so an A/B changes exactly one component.",
     "Tune the learning rate per arm on a task prefix, scored on later tasks the arm has not seen."],
    ["src/qcs/classical/scorers.py", "configs/tuned.yaml"],
    "The tuning step was decisive: at the reference paper's protocol (quantum 0.03, classical 1e-3) the linear arm scored 0.063 MRR; at its own best rate it scored 0.965. Every later comparison uses the tuned rates.",
    "Not applicable — this phase measures rather than stores.",
    "Not applicable directly, but it is what makes the plasticity claim testable rather than assumed. A handicapped baseline would have produced a flattering and worthless result."),

  ...phase(6, "Continual training",
    "Run the actual experiment: a long sequence of tasks, with both properties instrumented at every step.",
    "720 probes grouped into 120 sequential tasks",
    "trained scorers, a full transfer matrix, and a metric row per task per arm",
    ["Stream tasks in order. Train on each once, then evaluate on the current task and a sample of all past tasks, filling the matrix A[i][j].",
     "Mix each batch with samples from a reservoir buffer drawn uniformly over the whole history, not a recent window.",
     "At consolidation checkpoints, compute the quantum Fisher diagonal and anchor the parameters with Ω = ½ Σ Fₖ(θₖ − θ*ₖ)².",
     "Clip per-task movement with a trust region and watch the score distribution for drift.",
     "Wrap all angles back into (−π, π] after every task, and log both the wrapped and unwrapped norms."],
    ["runs/<id>/metrics.jsonl", "runs/<id>/cl_matrix_*.npy", "checkpoints", "figures V21–V34"],
    "Over 120 tasks: BWT positive for every arm (no catastrophic forgetting observed), and Tr(F) rising to 2.5× for the quantum arm while the linear arm decayed to 0.30× and the MLP collapsed to 0.06×.",
    "Replay plus the anchor keep the scorer accountable to material it saw hundreds of tasks earlier; BWT stayed positive, meaning later training did not damage earlier tasks.",
    "This phase produces the primary plasticity evidence. The divergence in Tr(F) between the compact and Euclidean arms is exactly the signature the reference paper predicts."),

  ...phase(7, "Assembly and generation",
    "Convert scores into the best possible prompt under a hard token ceiling, then generate.",
    "relevance scores over candidate memories",
    "a prompt of at most B tokens, and an answer",
    ["Count tokens exactly with the target model's tokenizer; reserve sub-budgets for the system prompt, pinned constraints and recent turns.",
     "Select greedily by marginal gain per token, with a redundancy penalty, which carries the (1 − 1/e) guarantee for a submodular objective.",
     "Apply the supersession filter: a stale unit is dropped, or shown explicitly as OUTDATED beside its CURRENT replacement — never served silently.",
     "Render with provenance tags in chronological order, and generate with a frozen model at temperature 0."],
    ["prompt templates (versioned)", "runs/<id>/phase7_summary.json", "figures V41–V45"],
    "The token ceiling was never exceeded; zero unlabelled stale facts were served by any arm; the learned arms reached their quality ceiling at B = 128 where the unlearned baselines needed B = 512.",
    "The supersession filter is the last line of defence for correctness: even if ranking fails, an outdated fact cannot be presented as current. Provenance tags make every answer auditable.",
    "The fixed budget is what allows the thread to grow without bound while the prompt stays constant — the system can keep absorbing new turns forever without the context inflating."),

  ...phase(8, "Evaluation",
    "Produce results that survive an adversarial reader, including the ones that did not go our way.",
    "trained arms plus the sealed test split, opened here for the first time",
    "a headline table over three seeds, seven ablations, a noise sweep, significance tests and a failure taxonomy",
    ["Repeat the full run across three seeds and report mean and standard deviation.",
     "Retrain the quantum arm with each component removed in turn, to price every part of the architecture.",
     "Apply depolarising noise by trajectory sampling to the trained circuit.",
     "Open the sealed test split and evaluate every arm on threads never seen.",
     "Test significance with a paired bootstrap over threads — the independent unit — rather than over probes.",
     "Categorise every remaining loss as retrieval, ranking, stale or eviction."],
    ["runs/<id>/phase8_summary.json", "figures V35–V40, V46"],
    "Headline: the parameter-matched classical twin beat the quantum arm significantly (Δ −0.056 MRR, CI [−0.070, −0.043], p = 0.000). Ablations: removing the relational block costs 0.324 MRR, more than every other component combined. Noise: ranking unchanged through p = 1e-3.",
    "Zero retrieval failures and zero evictions across every arm on the sealed split — the memory manifold and the assembler are not the bottleneck. Stale-fact errors fell from 0.130 (cosine) to 0.000–0.035.",
    "Plasticity was priced but not settled: 120 tasks is two orders of magnitude short of the horizon at which the reference effect appears. The Tr(F) divergence is an indicator, not a proof."),

  ...phase(9, "Hardening",
    "Make the result transferable to another person and another machine.",
    "a complete set of runs and artefacts",
    "a reproduction kit and a hardware-parity path",
    ["Run the trained circuit shot-based and on a real backend; report the shot count at which score ranking becomes stable.",
     "Scale across N ∈ {8, 10, 12} qubits and report wall clock honestly.",
     "Replace the extractive reader with a real frozen language model and repeat the budget sweep.",
     "Provide a single command that regenerates every figure from the raw logs."],
    ["planned: make reproduce, the scaling study, the hardware-parity report"],
    "Not started — the project was stopped at the end of Phase 8 by request.",
    "The reproduction kit is what makes the stability claim checkable by a third party rather than taken on trust.",
    "The long-horizon run belongs here, and it is the experiment that would actually settle the plasticity question."),
  BREAK(),
];

// ================================================================ part 3
const part3 = [
  H("3  The dataset's life, in one table", HeadingLevel.HEADING_1),
  P("The same corpus, viewed as it changes state through the ten phases."),
  TABLE(["After phase", "The dataset is now", "Size"], [
    ["0", "nothing — only the machinery that will make it reproducible", "—"],
    ["1", "raw chat turns with an exact ground-truth plan behind them", "18,000 turns · 1,800 probes · 1,920 facts"],
    ["2", "semantic units with frozen embeddings and frozen projections", "7,200 units · 384-d → 256-d"],
    ["3", "an append-only store, a graph, salience vectors and a vector index", "7,200 units · 9,989 edges · 144 summaries"],
    ["4", "unit-norm 1024-d feature vectors, i.e. valid quantum states", "1024 amplitudes per (query, candidate) pair"],
    ["5", "labelled training triples behind a single scoring interface", "720 probes × 64 candidates"],
    ["6", "a task stream, a transfer matrix and a metric row per task", "120 tasks × 3 arms × 3 seeds"],
    ["7", "prompts of at most B tokens, and generated answers", "≤ 3,072 tokens per prompt"],
    ["8", "a results table, ablations, noise curves and significance tests", "26 MB of run artefacts"],
    ["9", "a reproduction kit (planned)", "—"],
  ], [12, 58, 30], { boldCol: 0 }),
  BREAK(),
];

// ================================================================ part 4
const part4 = [
  H("4  How the design satisfies stability", HeadingLevel.HEADING_1),
  P("Stability is claimed on four mechanisms. Each is stated with the evidence that supports it and, where relevant, what remains unproven."),
  TABLE(["Mechanism", "Why it works", "Evidence"], [
    ["Knowledge is never stored in learned parameters", "A fact lives in an append-only row, not in a weight. No gradient step can reach it, so no gradient step can destroy it. Forgetting becomes a ranking error instead of an erasure.", "Zero retrieval failures across every arm on the sealed test split"],
    ["Immutability is enforced by the database", "Two triggers abort UPDATE and DELETE on the unit table. Immutability that depends on caller discipline is not immutability.", "Three mutation tests, all rejected"],
    ["Consolidation adds, never replaces", "Summaries are an extra index level pointing at their sources, so compression cannot lose a fact — unlike rolling summarisation, which discards what it summarises.", "Byte-equality asserted on every source after consolidation"],
    ["The generator is frozen", "No gradient path to the language model exists in the codebase; the training loop cannot even import it. Catastrophic forgetting of world knowledge is absent, not mitigated.", "Architectural, verified by inspection of the gradient scope"],
    ["The scorer itself is anchored", "Reservoir replay over the whole history plus a Fisher-weighted penalty keep ranking skill accountable to old material.", "BWT positive for every arm over 120 tasks"],
    ["Contradiction is represented, not resolved", "Superseded facts survive as graph nodes with edges, and the assembler labels them OUTDATED rather than dropping the history.", "Zero unlabelled stale facts served; stale-above rate 0.130 → 0.000–0.035"],
  ], [24, 46, 30], { boldCol: 0 }),
  CALLOUT("What is proven, and what is not",
    "Proven: information cannot be destroyed by learning, and the system did not forget over the horizon tested. Not proven: that BWT stays positive over thousands of tasks — the anchor is currently a small net negative at 60 tasks (+0.022 MRR when removed), because there is not yet anything to protect.", OKC),
  BREAK(),

  H("5  How the design satisfies plasticity", HeadingLevel.HEADING_1),
  P("Plasticity rests on the geometry of the parameter space rather than on any training heuristic. The argument has four steps, and the honest status of each is given."),
  TABLE(["Mechanism", "Why it works", "Evidence"], [
    ["Parameters live on a compact manifold", "The circuit's parameters are rotation angles: θ and θ + 2π are the same operation, and angles are wrapped after every task. There is no direction in which a weight can inflate.", "Unwrapped parameter norm plateaus at ~2.5× while the linear arm reaches 5× and is still climbing"],
    ["Evolution is unitary", "‖U(θ)‖₂ = 1 for every θ, so no layer can amplify a signal without bound and no activation can saturate.", "‖ψ‖ = 1 verified to 1e-4 at every layer; PennyLane cross-check at fidelity 1.000"],
    ["Learning capacity is bounded away from zero", "Haar-averaged Fisher information for a unitary ansatz is bounded both above and below, so effective capacity cannot decay to nothing — the mechanism the reference paper proves.", "Tr(F) rose to 2.5× over 120 tasks; the linear arm fell to 0.30× and the MLP to 0.06×"],
    ["No parameter is structurally dead", "Below the light-cone depth a parameter's gradient is exactly zero, not merely small. The configuration refuses to load a circuit shallower than ⌈N/2⌉.", "Gradient variance 3.6e-16 at depths 1–4, 1.1e-5 from depth 6"],
  ], [24, 46, 30], { boldCol: 0 }),
  RUNIMG("V21", CONTENT_P),
  CAP(3, "The plasticity evidence. Top right: effective learning capacity preserved on the compact manifold and lost in Euclidean space. Bottom right: the parameter norms that explain why."),
  CALLOUT("The honest status of the plasticity claim",
    "The mechanism is sound and the early indicator points the way the reference paper predicts. But losing Tr(F) is losing the ability to keep learning, not current accuracy — and at 120 tasks both classical arms are still near ceiling, so the loss has not yet surfaced as error. The reference paper needed 1,000–3,000 tasks. Until that run is done, plasticity is supported by mechanism and by a leading indicator, not by an outcome.", PLAS),
  BREAK(),
];

// ================================================================ part 6
const part6 = [
  H("6  Why the two do not interfere", HeadingLevel.HEADING_1),
  P("The reason both properties can be held at once is that they are carried by different objects with no shared coordinates."),
  TABLE(["", "Stability lives in", "Plasticity lives in"], [
    ["Object", "ℳ — an append-only classical store", "𝒰 — a compact unitary manifold"],
    ["What changes", "the store only grows; nothing is rewritten", "1,080 angles move freely, and wrap"],
    ["Failure mode if pushed", "retrieval slows; nothing is lost", "the scorer re-ranks; no fact is destroyed"],
    ["Coupling to the other", "supplies candidates", "returns one scalar per candidate"],
    ["Gradient", "none — nothing here is learned", "the only gradient in the system"],
  ], [22, 39, 39], { boldCol: 0 }),
  P("Because the coupling is one scalar wide, pressure on one side cannot deform the other. Training the scorer harder does not put a single stored fact at risk; adding ten thousand turns to memory does not change the circuit's capacity to learn. That is the whole design in one sentence: stability is architectural, plasticity is geometric, and the two never share a coordinate system."),

  H("6.1  The result, stated without varnish", HeadingLevel.HEADING_2),
  ...[
    "Stability: supported, with mechanism and evidence. Nothing in the corpus was lost, no unlabelled stale fact ever reached a prompt, and no retrieval failure occurred on the sealed split.",
    "Plasticity: mechanism sound, indicator positive, outcome untested at the horizon that matters.",
    "Quantum ranking advantage: refuted at this scale. A 1,025-parameter linear model beat the circuit significantly, at roughly 135× less compute.",
    "The component that actually earns its place is the relational block — recency, supersession and participant features — which is worth 0.324 MRR, more than everything else combined.",
  ].map(t => BULLET(t)),
  RULEP(),
  P("Companion to the main technical report. Every figure and number is regenerable from runs/*/metrics.jsonl.",
    { italics: true, color: MUTE, size: 19 }),
];

// ================================================================ assemble
const numbering = {
  config: [
    { reference: "bullets", levels: [
      { level: 0, format: LevelFormat.BULLET, text: "•", alignment: AlignmentType.LEFT,
        style: { paragraph: { indent: { left: 360, hanging: 200 } } } }] },
    { reference: "numbers", levels: [
      { level: 0, format: LevelFormat.DECIMAL, text: "%1.", alignment: AlignmentType.LEFT,
        style: { paragraph: { indent: { left: 360, hanging: 240 } } } }] },
  ],
};

const styles = {
  default: {
    document: { run: { font: "Calibri", size: 21, color: INK } },
    heading1: { run: { font: "Calibri Light", size: 34, bold: true, color: INK },
                paragraph: { spacing: { before: 400, after: 160 } } },
    heading2: { run: { font: "Calibri Light", size: 26, bold: true, color: STAB },
                paragraph: { spacing: { before: 300, after: 120 } } },
  },
};

const doc = new Document({
  creator: "Dual-Manifold ContextSync",
  title: "Ten Phases — dataset, output, stability and plasticity",
  numbering, styles,
  sections: [
    { properties: portraitProps, footers: { default: footer },
      children: [...cover, ...toc, ...part1] },
    { properties: landscapeProps, footers: { default: footer },
      children: [IMG("CIRCUIT_main.png", CONTENT_L),
                 CAP(1, "The circuit produced by Phase 4 and trained in Phase 6. The shaded wedge is the light cone from a gate on qubits (0,1): it reaches the readout qubit at layer 5, which is why depth 8 is used and why depth below 5 is refused.")] },
    { properties: portraitProps, footers: { default: footer },
      children: [IMG("CIRCUIT_gate.png", CONTENT_P),
                 CAP(2, "One gate, expanded. Fifteen parameters span the full su(4) algebra; the periodicity of those angles is the mechanism behind the plasticity claim in §5."),
                 ...part2, ...part3, ...part4, ...part6] },
  ],
});

Packer.toBuffer(doc).then(buf => {
  let out = path.join(__dirname, "..", "..", "deliverables", "Ten_Phases_Dataset_Stability_Plasticity.docx");
  try { fs.writeFileSync(out, buf); }
  catch (e) { out = out.replace(/\.docx$/, "_v2.docx"); fs.writeFileSync(out, buf); }
  console.log("wrote", out, (buf.length / 1024).toFixed(0) + " KB");
});
