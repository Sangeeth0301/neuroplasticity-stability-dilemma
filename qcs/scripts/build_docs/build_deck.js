/**
 * Dual-Manifold ContextSync — project deck.
 *   node docs/build_deck.js
 */
const fs = require("fs");
const path = require("path");
const pptxgen = require("pptxgenjs");

const ROOT = path.join(__dirname, "..", "..");
const FIGS = path.join(ROOT, "figures");
const RUNS = path.join(ROOT, "runs");

// ------------------------------------------------------------------ palette
const DARK = "0B1F27";     // deep teal-black — title / section / closing
const INK = "11161A";
const MUTE = "6B767E";
const PAPER = "FFFFFF";
const TINT = "F2F6F7";     // card tint
const TEAL = "0E5C74";     // classical / stability
const MAG = "A32467";      // quantum / plasticity
const GREEN = "2C7A58";
const AMBER = "C6852A";
const ICE = "CFE3E9";

const W = 13.333, H = 7.5;

const pres = new pptxgen();
pres.layout = "LAYOUT_WIDE";
pres.author = "Dual-Manifold ContextSync";
pres.title = "Dual-Manifold ContextSync";

let slideNo = 0;

// ------------------------------------------------------------------ helpers
function pngSize(file) {
  const b = fs.readFileSync(file);
  return { w: b.readUInt32BE(16), h: b.readUInt32BE(20) };
}

function findFigIn(dir, name) {
  for (const e of fs.readdirSync(dir, { withFileTypes: true })) {
    const full = path.join(dir, e.name);
    if (e.isDirectory()) { const hit = findFigIn(full, name); if (hit) return hit; }
    else if (e.name === name || (e.name.startsWith(name) && e.name.endsWith(".png"))) return full;
  }
  return null;
}

function findFig(name) {
  const hit = findFigIn(FIGS, name);
  if (hit) return hit;
  for (const d of fs.readdirSync(RUNS).sort().reverse()) {
    const fd = path.join(RUNS, d, "figs");
    if (!fs.existsSync(fd)) continue;
    const hit = fs.readdirSync(fd).find(f => f.startsWith(name) && f.endsWith(".png"));
    if (hit) return path.join(fd, hit);
  }
  throw new Error("figure not found: " + name);
}

/** Fit an image inside a box, preserving aspect, centred. */
function fitImage(slide, name, box) {
  const p = findFig(name);
  const d = pngSize(p);
  const ar = d.w / d.h;
  let w = box.w, h = w / ar;
  if (h > box.h) { h = box.h; w = h * ar; }
  slide.addImage({ path: p, x: box.x + (box.w - w) / 2, y: box.y + (box.h - h) / 2, w, h });
}

function newSlide(bg) {
  const s = pres.addSlide();
  s.background = { color: bg || PAPER };
  return s;
}

/** The repeated motif: two dots, teal then magenta — the two manifolds. */
function motif(slide, x, y, dark) {
  slide.addShape(pres.ShapeType.ellipse,
    { x, y, w: 0.13, h: 0.13, fill: { color: dark ? ICE : TEAL } });
  slide.addShape(pres.ShapeType.ellipse,
    { x: x + 0.2, y, w: 0.13, h: 0.13, fill: { color: MAG } });
}

function title(slide, text, sub) {
  motif(slide, 0.62, 0.46);
  slide.addText(text, {
    x: 0.55, y: 0.72, w: 12.2, h: 0.72, fontSize: 34, bold: true,
    color: INK, fontFace: "Cambria", margin: 0,
  });
  if (sub) {
    slide.addText(sub, {
      x: 0.55, y: 1.44, w: 12.2, h: 0.4, fontSize: 14, color: MUTE,
      fontFace: "Calibri", margin: 0,
    });
  }
}

function foot(slide, text) {
  slideNo += 1;
  slide.addText(text || "", {
    x: 0.55, y: 6.95, w: 10.5, h: 0.3, fontSize: 10, color: MUTE,
    fontFace: "Calibri", margin: 0,
  });
  slide.addText(String(slideNo), {
    x: 12.2, y: 6.95, w: 0.6, h: 0.3, fontSize: 10, color: MUTE,
    fontFace: "Calibri", align: "right", margin: 0,
  });
}

function card(slide, box, opts) {
  slide.addShape(pres.ShapeType.roundRect, {
    x: box.x, y: box.y, w: box.w, h: box.h, rectRadius: 0.06,
    fill: { color: opts.fill || TINT },
    line: { color: opts.line || "E1E8EA", width: 1 },
  });
}

function statBlock(slide, x, y, w, value, label, color) {
  slide.addText(value, {
    x, y, w, h: 0.85, fontSize: 42, bold: true, color, fontFace: "Cambria", margin: 0,
  });
  slide.addText(label, {
    x, y: y + 0.82, w, h: 0.6, fontSize: 12, color: MUTE, fontFace: "Calibri", margin: 0,
  });
}

function bullets(slide, items, box, size, color) {
  slide.addText(items.map((t, i) => ({
    text: t, options: { bullet: true, breakLine: i < items.length - 1 },
  })), {
    x: box.x, y: box.y, w: box.w, h: box.h, fontSize: size || 13,
    color: color || INK, fontFace: "Calibri", paraSpaceAfter: 8, margin: 0,
  });
}

function table(slide, head, rows, colW, box, opts) {
  opts = opts || {};
  const body = [
    head.map(t => ({
      text: t,
      options: { bold: true, color: MUTE, fontSize: 10.5, fill: { color: "EEF3F5" } },
    })),
    ...rows.map(r => r.map((t, i) => ({
      text: String(t),
      options: {
        fontSize: opts.size || 11,
        color: i === 0 ? INK : (opts.color || INK),
        bold: i === 0 && opts.boldFirst !== false,
      },
    }))),
  ];
  slide.addTable(body, {
    x: box.x, y: box.y, w: box.w, colW,
    border: { type: "solid", color: "E1E8EA", pt: 0.5 },
    fontFace: "Calibri", valign: "middle", rowH: opts.rowH || 0.32,
    margin: [4, 6, 4, 6],
  });
}

// ==================================================================== 1 TITLE
{
  const s = newSlide(DARK);
  s.addShape(pres.ShapeType.ellipse, { x: 0.9, y: 1.5, w: 0.2, h: 0.2, fill: { color: ICE } });
  s.addShape(pres.ShapeType.ellipse, { x: 1.22, y: 1.5, w: 0.2, h: 0.2, fill: { color: MAG } });
  s.addText("Dual-Manifold ContextSync", {
    x: 0.85, y: 2.0, w: 11.6, h: 1.2, fontSize: 46, bold: true, color: PAPER,
    fontFace: "Cambria", margin: 0,
  });
  s.addText("Solving the stability–plasticity dilemma in long, multi-user LLM chat threads with a quantum–classical hybrid", {
    x: 0.9, y: 3.2, w: 10.6, h: 0.9, fontSize: 17, color: ICE, fontFace: "Calibri", margin: 0,
  });
  s.addText([
    { text: "Complete build · Phases 0–8 delivered · 88 tests passing", options: { breakLine: true } },
    { text: "Python 3.14 · PyTorch · PennyLane · Qiskit · TensorCircuit-NG · FAISS · SQLite", options: {} },
  ], { x: 0.9, y: 5.35, w: 11, h: 0.9, fontSize: 12, color: "8FA8B2", fontFace: "Calibri", margin: 0 });
  s.addNotes("Opening. The one-line thesis: stability is architectural, plasticity is geometric, and they never share a coordinate system.");
  slideNo += 1;
}

// ==================================================================== 2 AGENDA
{
  const s = newSlide();
  title(s, "What this deck covers", "Problem → literature → architecture → every phase → results → what is next");
  const items = [
    ["01", "The problem", "Why long shared threads break language models", TEAL],
    ["02", "Literature", "What is already known, and the gap that remains", MAG],
    ["03", "Architecture", "Two manifolds joined by one scalar", TEAL],
    ["04", "The ten phases", "Inch by inch, dataset to answer", MAG],
    ["05", "Results", "Encoding · circuit · Fisher · stability · plasticity", GREEN],
    ["06", "What is next", "The experiment that would settle the open question", AMBER],
  ];
  items.forEach((it, i) => {
    const col = i % 2, row = Math.floor(i / 2);
    const x = 0.55 + col * 6.35, y = 2.05 + row * 1.55;
    s.addShape(pres.ShapeType.ellipse,
      { x, y: y + 0.06, w: 0.62, h: 0.62, fill: { color: it[3] } });
    s.addText(it[0], { x, y: y + 0.06, w: 0.62, h: 0.62, fontSize: 15, bold: true,
      color: PAPER, align: "center", valign: "middle", fontFace: "Calibri", margin: 0 });
    s.addText(it[1], { x: x + 0.85, y: y + 0.02, w: 5.2, h: 0.35, fontSize: 16, bold: true,
      color: INK, fontFace: "Calibri", margin: 0 });
    s.addText(it[2], { x: x + 0.85, y: y + 0.42, w: 5.3, h: 0.5, fontSize: 12,
      color: MUTE, fontFace: "Calibri", margin: 0 });
  });
  foot(s, "Every number in this deck was produced by the accompanying code.");
}

// ==================================================================== 3 PROBLEM
{
  const s = newSlide();
  title(s, "The problem", "A shared chat thread is a hostile environment for a language model");
  card(s, { x: 0.55, y: 2.05, w: 6.0, h: 4.5 }, {});
  s.addText("The dilemma", { x: 0.9, y: 2.3, w: 5.3, h: 0.4, fontSize: 17, bold: true,
    color: TEAL, fontFace: "Calibri", margin: 0 });
  s.addText("Stability and plasticity are usually carried by the same parameter vector θ ∈ ℝⁿ, so they compete for the same coordinates.", {
    x: 0.9, y: 2.75, w: 5.3, h: 0.9, fontSize: 13, color: INK, fontFace: "Calibri", margin: 0 });
  bullets(s, [
    "Protect the weights → the model stops adapting",
    "Let them move → earlier knowledge is overwritten",
    "This is not a tuning problem. It follows from storing both properties in one unbounded Euclidean space.",
  ], { x: 0.9, y: 3.75, w: 5.3, h: 2.4 }, 13);

  statBlock(s, 7.1, 2.15, 5.6, "600+", "turns in a single thread before any context window can hold it", TEAL);
  statBlock(s, 7.1, 3.55, 5.6, "5", "participants stating facts that contradict each other", MAG);
  statBlock(s, 7.1, 4.95, 5.6, "0.130", "of the time plain similarity search returns a fact that was already superseded", AMBER);
  foot(s, "Measured on the project corpus with a cosine retriever.");
}

// ==================================================================== 4 FAILURES
{
  const s = newSlide();
  title(s, "Three distinct failure modes", "They are usually conflated, and they need different fixes");
  const cards = [
    ["Catastrophic forgetting", TEAL,
     "A gradient step taken for turn 500 destroys the representation that encoded turn 40.",
     "The information is gone. No retrieval strategy recovers it."],
    ["Loss of plasticity", MAG,
     "Weights inflate, activations saturate, the Fisher trace decays toward zero.",
     "Parameters still exist but are functionally dead — the model has stopped being able to learn."],
    ["Context saturation", AMBER,
     "The naive fix is to retrieve more and put more in the prompt.",
     "Attention spreads thinner; quality falls even though the fact is in the window."],
  ];
  cards.forEach((c, i) => {
    const x = 0.55 + i * 4.15;
    card(s, { x, y: 2.1, w: 3.85, h: 4.15 }, {});
    s.addShape(pres.ShapeType.ellipse, { x: x + 0.3, y: 2.4, w: 0.5, h: 0.5, fill: { color: c[1] } });
    s.addText(String(i + 1), { x: x + 0.3, y: 2.4, w: 0.5, h: 0.5, fontSize: 14, bold: true,
      color: PAPER, align: "center", valign: "middle", fontFace: "Calibri", margin: 0 });
    s.addText(c[0], { x: x + 0.3, y: 3.05, w: 3.25, h: 0.6, fontSize: 16, bold: true,
      color: INK, fontFace: "Calibri", margin: 0 });
    s.addText(c[2], { x: x + 0.3, y: 3.7, w: 3.25, h: 1.0, fontSize: 12, color: INK,
      fontFace: "Calibri", margin: 0 });
    s.addText(c[3], { x: x + 0.3, y: 4.85, w: 3.25, h: 1.2, fontSize: 12, color: MUTE,
      italic: true, fontFace: "Calibri", margin: 0 });
  });
  foot(s, "The architecture is built so each of these is structurally impossible, not merely mitigated.");
}

// ==================================================================== 5 LIT REVIEW
{
  const s = newSlide();
  title(s, "Literature review", "What is established, and what each line of work leaves open");
  table(s,
    ["Work", "Contribution", "What it leaves open"],
    [
      ["McCloskey & Cohen (1989)", "Named catastrophic interference", "Descriptive; no preventive mechanism"],
      ["Kirkpatrick et al. (2017) — EWC", "Fisher-weighted penalty anchoring important weights", "Knowledge still lives in weights; hyper-parameter sensitive"],
      ["Lopez-Paz & Ranzato (2017) — GEM", "Replay with gradient constraints; the BWT / FWT metrics", "Memory cost grows; silent on plasticity"],
      ["Dohare et al. (2024), Nature", "Showed deep nets progressively lose the ability to learn", "Fixes are extrinsic patches: resets, injection, regularisers"],
      ["McClean et al. (2018); Cerezo et al. (2021)", "Barren plateaus, and their dependence on the observable", "Motivates local readouts and shallow, structured ansätze"],
      ["Mitarai et al. (2018); Schuld et al. (2019)", "Parameter-shift rule for analytic gradients", "Assumes involutory generators — untrue for a general SU(4) sum"],
      ["Lewis et al. (2020) — RAG", "Retrieval as external, non-parametric memory", "Retrieval ranking is static; no continual adaptation"],
      ["Liu et al. (2023) — Lost in the Middle", "Long contexts degrade even when the fact is present", "Confirms saturation; offers no allocation policy"],
      ["Chen & Zhang (2026), PRX Quantum", "Quantum models preserve plasticity intrinsically — compact manifold", "States explicitly that it does NOT solve stability"],
    ],
    [3.2, 4.5, 4.5], { x: 0.55, y: 1.95, w: 12.2 }, { size: 10, rowH: 0.38 });
  foot(s, "Classic references are cited from the standard literature; the 2026 PRX Quantum paper is the project's direct antecedent.");
}

// ==================================================================== 6 GAP
{
  const s = newSlide(DARK);
  motif(s, 0.62, 0.9, true);
  s.addText("The gap", { x: 0.55, y: 1.25, w: 8, h: 0.8, fontSize: 34, bold: true,
    color: PAPER, fontFace: "Cambria", margin: 0 });
  s.addText("Quantum circuits keep the ability to learn. They do not keep what was learned.", {
    x: 0.6, y: 2.25, w: 11.6, h: 0.7, fontSize: 22, color: ICE, fontFace: "Calibri", margin: 0 });
  s.addText([
    { text: "The reference paper is explicit: both quantum and classical models still forget catastrophically without an explicit stability mechanism, and it names a hybrid — a plastic quantum backbone plus classical memory — as the promising direction.", options: { breakLine: true } },
    { text: "", options: { breakLine: true } },
    { text: "So the correct division of labour is the opposite of the intuitive one.", options: { bold: true } },
  ], { x: 0.6, y: 3.3, w: 11.4, h: 1.6, fontSize: 14, color: "B9CED6", fontFace: "Calibri", margin: 0 });

  s.addShape(pres.ShapeType.roundRect, { x: 0.6, y: 5.1, w: 5.6, h: 1.5, rectRadius: 0.06,
    fill: { color: "12313C" }, line: { color: TEAL, width: 1 } });
  s.addText("Classical, append-only storage\nfor what must never be lost", {
    x: 0.85, y: 5.35, w: 5.1, h: 1.0, fontSize: 14, color: ICE, fontFace: "Calibri", margin: 0 });
  s.addShape(pres.ShapeType.roundRect, { x: 6.5, y: 5.1, w: 5.7, h: 1.5, rectRadius: 0.06,
    fill: { color: "36132A" }, line: { color: MAG, width: 1 } });
  s.addText("A compact quantum manifold\nfor what must keep learning", {
    x: 6.75, y: 5.35, w: 5.2, h: 1.0, fontSize: 14, color: "F0C9DC", fontFace: "Calibri", margin: 0 });
  slideNo += 1;
}

// ==================================================================== 7 THESIS
{
  const s = newSlide();
  title(s, "The thesis", "Put the two properties on two different mathematical objects");
  const panes = [
    ["ℳ  ·  Stability manifold", TEAL, "F1F7F9", [
      "Every semantic unit stored verbatim, append-only",
      "Immutability enforced by database triggers, not convention",
      "Nothing here is learned, so nothing here can be unlearned",
      "Forgetting degrades to a retrieval miss — bounded and fixable",
    ]],
    ["𝒰  ·  Plasticity manifold", MAG, "FBF0F5", [
      "A 10-qubit variational circuit, 1,080 angles",
      "Parameters live on a torus: they wrap, they cannot inflate",
      "Unitary evolution bounds every operator norm at 1",
      "Fisher information bounded away from zero — capacity cannot decay",
    ]],
  ];
  panes.forEach((p, i) => {
    const x = 0.55 + i * 6.3;
    card(s, { x, y: 2.05, w: 6.0, h: 3.5 }, { fill: p[2], line: p[1] });
    s.addText(p[0], { x: x + 0.35, y: 2.3, w: 5.3, h: 0.45, fontSize: 18, bold: true,
      color: p[1], fontFace: "Cambria", margin: 0 });
    bullets(s, p[3], { x: x + 0.35, y: 2.85, w: 5.3, h: 2.5 }, 12.5);
  });
  card(s, { x: 0.55, y: 5.75, w: 12.2, h: 0.95 }, { fill: "F7F9FA" });
  s.addText("The two meet at exactly one place: the circuit consumes an amplitude-encoded (query, candidate, participant) state and returns one scalar in [0,1]. That is the entire coupling — narrow enough to simulate exactly, and narrow enough to swap for a classical scorer, which is what makes every quantum claim falsifiable.", {
    x: 0.85, y: 5.95, w: 11.7, h: 0.6, fontSize: 12.5, color: INK, fontFace: "Calibri", margin: 0 });
  foot(s);
}

// ==================================================================== 8 ARCH
{
  const s = newSlide();
  title(s, "The architecture", "Four bands, one direction of flow, one gradient path");
  fitImage(s, "DIAG_A_architecture.png", { x: 0.5, y: 1.75, w: 12.3, h: 4.95 });
  foot(s, "Teal = classical and append-only · magenta = the only component that learns · slate = frozen.");
}

// ======================================================= 9 THE CIRCUIT (single)
{
  const s = newSlide();
  title(s, "The variational quantum circuit",
    "|0⟩ register · amplitude encoding · 8 brickwall layers of general SU(4) · measurement on q9");
  fitImage(s, "CIRCUIT_vqc_canonical.png", { x: 0.35, y: 1.72, w: 12.6, h: 4.42 });
  const facts = [
    ["1,080", "angles — 8 layers × 9 gates × 15", MAG],
    ["3 CNOTs", "per gate on hardware — the entangling elements", TEAL],
    ["depth ≥ 5", "light-cone floor, enforced by the config", GREEN],
    ["‖ψ‖ = 1", "at every layer — verified against PennyLane", GREEN],
  ];
  facts.forEach((f, i) => {
    const x = 0.55 + i * 3.12;
    s.addText(f[0], { x, y: 6.24, w: 1.35, h: 0.4, fontSize: 17, bold: true, color: f[2],
      fontFace: "Cambria", margin: 0 });
    s.addText(f[1], { x: x + 1.4, y: 6.28, w: 1.65, h: 0.5, fontSize: 9.5, color: MUTE,
      fontFace: "Calibri", margin: 0 });
  });
  foot(s, "Left inset: the CNOT decomposition a compiler emits for one V(θ). Right inset: entanglement measured on the trained circuit.");
}


// ==================================================================== 11 DATASET
{
  const s = newSlide();
  title(s, "The dataset", "Generated in two steps, because the separation is what creates exact ground truth");
  card(s, { x: 0.55, y: 2.05, w: 5.9, h: 1.15 }, { fill: "F1F7F9", line: TEAL });
  s.addText("1 · PLAN  —  structured, and IS the ground truth", { x: 0.8, y: 2.2, w: 5.4, h: 0.3,
    fontSize: 13, bold: true, color: TEAL, fontFace: "Calibri", margin: 0 });
  s.addText("which turn states which fact · which supersedes it · which probes it, at what distance", {
    x: 0.8, y: 2.55, w: 5.4, h: 0.55, fontSize: 11.5, color: INK, fontFace: "Calibri", margin: 0 });
  card(s, { x: 6.85, y: 2.05, w: 5.9, h: 1.15 }, { fill: "FBF0F5", line: MAG });
  s.addText("2 · SURFACE TEXT  —  templates only", { x: 7.1, y: 2.2, w: 5.4, h: 0.3,
    fontSize: 13, bold: true, color: MAG, fontFace: "Calibri", margin: 0 });
  s.addText("prose may never add or remove a fact — an LLM writing freely would destroy the ground truth", {
    x: 7.1, y: 2.55, w: 5.4, h: 0.55, fontSize: 11.5, color: INK, fontFace: "Calibri", margin: 0 });

  table(s, ["Turn kind", "What it is", "Why it exists"],
    [
      ["fact", "a participant states an (entity, attribute, value) they own", "the retrieval target"],
      ["update", "a later turn overwrites that fact", "creates the stale-versus-current trap"],
      ["probe", "a question answerable only from stated facts", "the measurement point"],
      ["distractor", "mentions a real entity with no usable value", "defeats keyword matching"],
      ["filler", "ordinary conversation", "dilution"],
    ], [2.0, 5.6, 4.6], { x: 0.55, y: 3.45, w: 12.2 }, { size: 11, rowH: 0.36 });

  const stats = [["18,000", "turns"], ["1,800", "probes"], ["1,920", "facts"],
                 ["1.000", "oracle recall"], ["0.042", "random recall"]];
  stats.forEach((st, i) => {
    const x = 0.55 + i * 2.46;
    s.addText(st[0], { x, y: 5.85, w: 2.3, h: 0.45, fontSize: 24, bold: true, color: TEAL,
      fontFace: "Cambria", margin: 0 });
    s.addText(st[1], { x, y: 6.3, w: 2.3, h: 0.3, fontSize: 11, color: MUTE,
      fontFace: "Calibri", margin: 0 });
  });
  foot(s, "Three regimes cycle across threads: single-owner, multi-owner, adversarial.");
}

// ==================================================================== 12 PHASES MAP
{
  const s = newSlide();
  title(s, "Ten phases", "Each phase's exit criterion is the next phase's precondition");
  const ph = [
    ["0", "Foundations", "reproducible substrate", TEAL],
    ["1", "Dataset", "ground truth by construction", TEAL],
    ["2", "Encoding", "text → unit-norm ℝ¹⁰²⁴", TEAL],
    ["3", "Memory ℳ", "append-only, enforced by the DB", TEAL],
    ["4", "Quantum core", "verified circuit, depth by evidence", MAG],
    ["5", "Baselines", "each arm at its own best lr", MAG],
    ["6", "Continual training", "120 tasks, full instrumentation", MAG],
    ["7", "Assembly", "budgeted prompt, frozen generator", GREEN],
    ["8", "Evaluation", "seeds, ablations, noise, sealed split", GREEN],
    ["9", "Hardening", "not started — the long-horizon run", AMBER],
  ];
  ph.forEach((p, i) => {
    const col = i % 5, row = Math.floor(i / 5);
    const x = 0.55 + col * 2.46, y = 2.15 + row * 2.15;
    card(s, { x, y, w: 2.28, h: 1.95 }, {});
    s.addShape(pres.ShapeType.ellipse, { x: x + 0.18, y: y + 0.18, w: 0.46, h: 0.46,
      fill: { color: p[3] } });
    s.addText(p[0], { x: x + 0.18, y: y + 0.18, w: 0.46, h: 0.46, fontSize: 13, bold: true,
      color: PAPER, align: "center", valign: "middle", fontFace: "Calibri", margin: 0 });
    s.addText(p[1], { x: x + 0.18, y: y + 0.72, w: 1.95, h: 0.45, fontSize: 12.5, bold: true,
      color: INK, fontFace: "Calibri", margin: 0 });
    s.addText(p[2], { x: x + 0.18, y: y + 1.15, w: 1.95, h: 0.7, fontSize: 10.5, color: MUTE,
      fontFace: "Calibri", margin: 0 });
  });
  foot(s, "Phases 0–8 are complete; phase 9 is the remaining work.");
}

// ============================================== 13-15 PHASE DETAIL TABLES
function phaseTable(head, rows, note) {
  const s = newSlide();
  title(s, head[0], head[1]);
  table(s, ["Phase", "What happens to the data", "What comes out", "Measured"],
    rows, [1.85, 4.6, 3.0, 2.75], { x: 0.55, y: 2.0, w: 12.2 },
    { size: 10.5, rowH: 0.46 });
  foot(s, note);
}

phaseTable(
  ["Phases 0–3 · building the dataset and the memory", "From nothing to an append-only store that cannot lose a fact"],
  [
    ["0 · Foundations", "No data yet. Every tunable number moves into one YAML file; seeding uses hashlib, not Python's per-process-salted hash().", "Config, seeding, run logger, typed records", "Two identical-seed runs produce identical output"],
    ["1 · Dataset", "Plan each thread first: assign a probe-distance band, then place the fact. Add supersession chains. Realise prose from the plan with templates.", "18,000 turns · 1,800 probes · 1,920 facts · sealed test split", "Oracle 1.000 · random 0.042"],
    ["2 · Encoding", "Segment losslessly; embed with a frozen encoder; whiten once on held-out threads; train a projection with InfoNCE then freeze it.", "7,200 semantic units · frozen 384-d and 256-d vectors", "Held-out recall@20: 0.879 → 0.996"],
    ["3 · Memory ℳ", "Write every unit verbatim; add summary nodes beside sources; build the supersession graph; track per-user salience; index for causal retrieval.", "SQLite store · FAISS index · 9,989 edges", "Prefilter recall 1.000 · 3 mutation tests rejected"],
  ],
  "Exit criteria are hard: nothing in phase 4 was started while any of these was red.");

phaseTable(
  ["Phases 4–6 · the quantum core and the experiment", "From a verified circuit to a 120-task continual run"],
  [
    ["4 · Quantum core", "Implement the general SU(4) gate, amplitude-encode 1024 features into 10 qubits, arrange 8 brickwall layers, read out one qubit.", "A verified circuit with 1,080 parameters", "PennyLane fidelity 1.000 · gradient agreement 8.6e-9"],
    ["5 · Baselines", "Build a parameter-matched linear twin (1,025), a wider MLP, cosine, recency, random and oracle — all behind one interface. Tune the learning rate per arm.", "Six comparison arms · configs/tuned.yaml", "Tuning moved the linear arm 0.063 → 0.965 MRR"],
    ["6 · Continual training", "Stream 120 tasks in order. Train once per task, evaluate on all past tasks, replay uniformly over history, anchor with the quantum Fisher, wrap angles.", "Trained scorers · transfer matrix · a metric row per task", "BWT positive for every arm · Tr(F) diverges as predicted"],
  ],
  "Phase 5 is where an unfair comparison was caught and corrected before it could contaminate phase 6.");

phaseTable(
  ["Phases 7–9 · delivery, evaluation and what remains", "From scores to an answer, and from an answer to a defensible result"],
  [
    ["7 · Assembly", "Count tokens exactly, reserve sub-budgets, select greedily by gain per token with a redundancy penalty, filter superseded facts, render with provenance.", "A prompt of at most B tokens · an answer", "Budget never exceeded · zero unlabelled stale facts"],
    ["8 · Evaluation", "Three seeds; seven ablations retrained from scratch; depolarising noise by trajectory sampling; open the sealed split; paired bootstrap over threads.", "Headline table · ablations · noise curves · significance", "Quantum − linear = −0.056 MRR, p = 0.000"],
    ["9 · Hardening", "Not started. Hardware parity with finite shots, a scaling study over qubit count, a real frozen LLM, and a one-command reproduction kit.", "Reproduction kit · hardware-parity report", "—"],
  ],
  "Phase 8 produced the project's central negative result, reported rather than buried.");

// ==================================================================== 17-21 RESULTS
function figSlide(t, sub, fig, note) {
  const s = newSlide();
  title(s, t, sub);
  fitImage(s, fig, { x: 0.5, y: 1.8, w: 12.3, h: 4.85 });
  foot(s, note);
}

figSlide("Result 1 · How 1024 numbers become 10 qubits",
  "Amplitude encoding is lossless precisely because the feature dimension equals 2¹⁰",
  "RES1_encoding_1024_to_10qubits.png",
  "Participation ratio 189 of 1024 basis states — up from 61 before per-block norm balancing.");

figSlide("Result 2 · Inside the variational circuit",
  "Entanglement builds with depth, the norm never drifts, and training creates the separation",
  "RES2_vqc_behaviour.png",
  "Readout separation is created by training: Cohen's d rises from ≈0 at initialisation.");

figSlide("Result 3 · Fisher information — the plasticity instrument",
  "Trace over the task sequence, eigenvalue spectrum, and effective rank",
  "RES3_fisher_information.png",
  "Capacity retained after 120 tasks: quantum 4.75× · linear 0.28× · MLP 0.06×.");

figSlide("Result 4 · Stability",
  "Memory that only grows, stale facts suppressed, and zero retrieval failures",
  "RES4_stability_evidence.png",
  "Every remaining loss is a ranking decision — ℳ and the assembler are not the bottleneck.");

figSlide("Result 5 · Plasticity",
  "Capacity, parameter norm, gradient norm, and the light cone that sets the minimum depth",
  "RES5_plasticity_evidence.png",
  "The compact manifold is the mechanism: angles wrap, Euclidean weights do not.");

// ==================================================================== 22 HEADLINE
{
  const s = newSlide();
  title(s, "Headline results", "Three seeds · 120 tasks · every arm at its own tuned learning rate");
  table(s, ["Arm", "|θ|", "rank-1", "MRR", "stale-above ↓", "wall clock"],
    [
      ["quantum (QRE)", "1,080", "0.928 ± 0.038", "0.959 ± 0.022", "0.015", "406 s"],
      ["linear (matched)", "1,025", "0.979 ± 0.002", "0.990 ± 0.001", "0.000", "3 s"],
      ["mlp (capacity ref.)", "65,665", "0.996 ± 0.002", "0.998 ± 0.001", "0.000", "20 s"],
      ["cosine (no learning)", "0", "0.894", "0.943", "0.078", "—"],
      ["recency (truncation)", "0", "0.025", "0.124", "0.521", "—"],
      ["oracle (ceiling)", "—", "1.000", "1.000", "0.000", "—"],
    ], [2.9, 1.4, 2.2, 2.2, 2.0, 1.5], { x: 0.55, y: 2.05, w: 12.2 }, { size: 11.5, rowH: 0.4 });

  card(s, { x: 0.55, y: 5.0, w: 12.2, h: 1.6 }, { fill: "FBF0F5", line: MAG });
  s.addText("The honest headline", { x: 0.85, y: 5.2, w: 5, h: 0.3, fontSize: 14, bold: true,
    color: MAG, fontFace: "Calibri", margin: 0 });
  s.addText("The quantum arm is significantly worse than its parameter-matched classical twin: Δ −0.056 MRR, 95% CI [−0.070, −0.043], p = 0.000 by paired bootstrap over threads — at roughly 135× the compute and 19× the seed-to-seed variance. The design required this comparison to be falsifiable; it falsified in the classical arm's favour at this scale.", {
    x: 0.85, y: 5.55, w: 11.6, h: 0.95, fontSize: 12.5, color: INK, fontFace: "Calibri", margin: 0 });
  foot(s, "Sealed test split agrees: quantum 0.919 MRR vs linear 0.975.");
}

// ==================================================================== 25 SCORECARD
{
  const s = newSlide();
  title(s, "Every claim, and its status", "Including the one that did not go our way");
  fitImage(s, "RES6_scorecard.png", { x: 0.5, y: 1.8, w: 12.3, h: 4.8 });
  foot(s, "A claim is only listed as supported when a specific measurement backs it.");
}

// ============================================== 22 FUTURE PLANS (objectives)
{
  const s = newSlide(DARK);
  s.addShape(pres.ShapeType.ellipse, { x: 0.62, y: 0.62, w: 0.14, h: 0.14, fill: { color: ICE } });
  s.addShape(pres.ShapeType.ellipse, { x: 0.83, y: 0.62, w: 0.14, h: 0.14, fill: { color: MAG } });
  s.addText("Future plans", { x: 0.55, y: 0.95, w: 9, h: 0.75, fontSize: 34, bold: true,
    color: PAPER, fontFace: "Cambria", margin: 0 });
  s.addText("Five objectives, in dependency order", { x: 0.58, y: 1.75, w: 9, h: 0.4,
    fontSize: 14, color: ICE, fontFace: "Calibri", margin: 0 });

  const objectives = [
    ["01", "Settle the plasticity question", "Run 1,000–3,000 tasks across three seeds on both arms, with Tr(F) and the plasticity index as the primary endpoints.", MAG],
    ["02", "Make the circuit fast enough to do it", "Batched multi-candidate state preparation, a GPU backend, or a shallower ansatz — this is what unblocks objective 1.", MAG],
    ["03", "Put a real frozen LLM in place", "Replace the extractive reader so attention dilution, and therefore the saturation turnover, becomes observable.", TEAL],
    ["04", "Establish hardware parity", "Shot-based execution on a real backend; report the shot count at which the score ranking becomes stable.", TEAL],
    ["05", "Ship a reproduction kit", "One command that regenerates every figure from the raw logs, plus the scaling study across 8, 10 and 12 qubits.", GREEN],
  ];
  objectives.forEach((o, i) => {
    const y = 2.42 + i * 0.92;
    s.addShape(pres.ShapeType.ellipse, { x: 0.6, y: y + 0.04, w: 0.5, h: 0.5,
      fill: { color: o[3] } });
    s.addText(o[0], { x: 0.6, y: y + 0.04, w: 0.5, h: 0.5, fontSize: 12, bold: true,
      color: PAPER, align: "center", valign: "middle", fontFace: "Calibri", margin: 0 });
    s.addText(o[1], { x: 1.3, y: y, w: 4.3, h: 0.4, fontSize: 14.5, bold: true,
      color: PAPER, fontFace: "Calibri", margin: 0 });
    s.addText(o[2], { x: 5.75, y: y - 0.03, w: 7.0, h: 0.75, fontSize: 11.5,
      color: "9FB8C1", fontFace: "Calibri", margin: 0 });
  });
  slideNo += 1;
}


// ---------------------------------------------------------------- write
const candidates = [
  "Dual_Manifold_ContextSync_Deck.pptx",
  "Dual_Manifold_ContextSync_Deck_v2.pptx",
  "Dual_Manifold_ContextSync_Deck_final.pptx",
];

(async () => {
  for (const name of candidates) {
    const out = path.join(ROOT, "deliverables", name);
    try {
      await pres.writeFile({ fileName: out });
      console.log("wrote", out, "·", slideNo, "slides");
      return;
    } catch (e) {
      console.log("(locked, trying next):", name);
    }
  }
  console.error("every candidate filename was locked — close the deck in PowerPoint");
})();
