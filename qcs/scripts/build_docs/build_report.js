/**
 * Builds the full project report as a .docx.
 *   node docs/build_report.js
 * Output: docs/Dual_Manifold_ContextSync_Report.docx
 */
const fs = require("fs");
const path = require("path");
const {
  Document, Packer, Paragraph, TextRun, HeadingLevel, AlignmentType, PageBreak,
  Table, TableRow, TableCell, WidthType, ShadingType, BorderStyle, ImageRun,
  TableOfContents, PageOrientation, PageNumber, Header, Footer, LevelFormat,
  convertInchesToTwip,
} = require("docx");

const ROOT = path.join(__dirname, "..", "..");
const FIGS = path.join(ROOT, "figures");
const RUNS = path.join(ROOT, "runs");

// ---------------------------------------------------------------- palette
const INK = "12171B";
const MUTE = "5B6670";
const STAB = "0F5F79";
const PLAS = "A6246A";
const OKC = "2E7D5B";
const WARN = "8A5A00";
const RULE = "C6CFD5";
const HDRBG = "EEF3F5";
const ALTBG = "F7F9FA";

const PAGE_W = 12240, PAGE_H = 15840, MARGIN = 1080;
const CONTENT_P = PAGE_W - 2 * MARGIN;              // portrait content width (DXA)
const CONTENT_L = PAGE_H - 2 * MARGIN;              // landscape content width (DXA)

// ---------------------------------------------------------------- helpers
const P = (text, opts = {}) => new Paragraph({
  spacing: { after: opts.after ?? 120, line: opts.line ?? 276 },
  alignment: opts.align,
  indent: opts.indent,
  children: [new TextRun({
    text, size: opts.size ?? 21, color: opts.color ?? INK,
    bold: opts.bold, italics: opts.italics, font: opts.font ?? "Calibri",
  })],
});

const RICH = (runs, opts = {}) => new Paragraph({
  spacing: { after: opts.after ?? 120, line: 276 },
  alignment: opts.align,
  children: runs.map(r => new TextRun({
    text: r.t, bold: r.b, italics: r.i, color: r.c ?? INK,
    size: r.s ?? 21, font: r.f ?? "Calibri",
  })),
});

const H = (text, level) => new Paragraph({
  text, heading: level,
  spacing: { before: level === HeadingLevel.HEADING_1 ? 360 : 260, after: 140 },
});

const BULLET = (text, opts = {}) => new Paragraph({
  numbering: { reference: "bullets", level: opts.level ?? 0 },
  spacing: { after: 70, line: 264 },
  children: [new TextRun({ text, size: 21, color: INK, font: "Calibri" })],
});

const NUM = (text) => new Paragraph({
  numbering: { reference: "numbers", level: 0 },
  spacing: { after: 70, line: 264 },
  children: [new TextRun({ text, size: 21, color: INK, font: "Calibri" })],
});

const CODE = (lines) => new Paragraph({
  spacing: { before: 100, after: 140, line: 240 },
  shading: { type: ShadingType.CLEAR, fill: "F2F5F6" },
  border: { left: { style: BorderStyle.SINGLE, size: 12, color: STAB, space: 6 } },
  children: lines.flatMap((l, i) => [
    ...(i ? [new TextRun({ break: 1 })] : []),
    new TextRun({ text: l, font: "Consolas", size: 17, color: INK }),
  ]),
});

const CALLOUT = (title, body, color = WARN) => new Table({
  width: { size: CONTENT_P, type: WidthType.DXA },
  columnWidths: [CONTENT_P],
  borders: {
    top: { style: BorderStyle.SINGLE, size: 4, color: RULE },
    bottom: { style: BorderStyle.SINGLE, size: 4, color: RULE },
    left: { style: BorderStyle.SINGLE, size: 18, color },
    right: { style: BorderStyle.SINGLE, size: 4, color: RULE },
    insideHorizontal: { style: BorderStyle.NONE }, insideVertical: { style: BorderStyle.NONE },
  },
  rows: [new TableRow({
    children: [new TableCell({
      width: { size: CONTENT_P, type: WidthType.DXA },
      shading: { type: ShadingType.CLEAR, fill: ALTBG },
      margins: { top: 130, bottom: 130, left: 160, right: 160 },
      children: [
        new Paragraph({
          spacing: { after: 60 },
          children: [new TextRun({ text: title, bold: true, size: 20, color, font: "Calibri" })],
        }),
        new Paragraph({
          spacing: { after: 0, line: 264 },
          children: [new TextRun({ text: body, size: 20, color: INK, font: "Calibri" })],
        }),
      ],
    })],
  })],
});

function TABLE(headers, rows, widths, opts = {}) {
  const total = widths.reduce((a, b) => a + b, 0);
  const scale = (opts.width ?? CONTENT_P) / total;
  const cols = widths.map(w => Math.round(w * scale));
  const cell = (text, i, isHeader, bold, color) => new TableCell({
    width: { size: cols[i], type: WidthType.DXA },
    shading: { type: ShadingType.CLEAR, fill: isHeader ? HDRBG : (opts.zebra && opts.rowIndex % 2 ? ALTBG : "FFFFFF") },
    margins: { top: 70, bottom: 70, left: 100, right: 100 },
    // one paragraph per line; "~~" is the explicit in-cell line break
    children: String(text).split("~~").map((line) => new Paragraph({
      spacing: { after: 0, line: 240 },
      children: [new TextRun({
        text: line.trim(), bold: isHeader || bold, size: isHeader ? 17 : 18,
        color: isHeader ? MUTE : (color ?? INK),
        font: opts.mono ? "Consolas" : "Calibri",
      })],
    })),
  });
  return new Table({
    width: { size: opts.width ?? CONTENT_P, type: WidthType.DXA },
    columnWidths: cols,
    borders: {
      top: { style: BorderStyle.SINGLE, size: 4, color: RULE },
      bottom: { style: BorderStyle.SINGLE, size: 4, color: RULE },
      left: { style: BorderStyle.NONE }, right: { style: BorderStyle.NONE },
      insideHorizontal: { style: BorderStyle.SINGLE, size: 2, color: RULE },
      insideVertical: { style: BorderStyle.NONE },
    },
    rows: [
      new TableRow({
        tableHeader: true,
        children: headers.map((h, i) => cell(h, i, true)),
      }),
      ...rows.map((r, ri) => new TableRow({
        children: r.map((c, i) => {
          opts.rowIndex = ri;
          const bold = opts.boldCol === i;
          const color = opts.colorRows && opts.colorRows[ri];
          return cell(c, i, false, bold, color);
        }),
      })),
    ],
  });
}

function findFigIn(dir, name) {
  for (const e of fs.readdirSync(dir, { withFileTypes: true })) {
    const full = path.join(dir, e.name);
    if (e.isDirectory()) { const hit = findFigIn(full, name); if (hit) return hit; }
    else if (e.name === name || (e.name.startsWith(name) && e.name.endsWith(".png"))) return full;
  }
  return null;
}

function IMG(file, widthDxa, opts = {}) {
  const src = findFigIn(FIGS, file) || path.join(RUNS, file);
  const dims = pngSize(src);
  const wPx = Math.round((widthDxa / 1440) * 96);
  const hPx = Math.round(wPx * (dims.h / dims.w));
  return new Paragraph({
    alignment: AlignmentType.CENTER,
    spacing: { before: 140, after: 80 },
    children: [new ImageRun({
      type: "png", data: fs.readFileSync(src),
      transformation: { width: wPx, height: hPx },
    })],
  });
}

function pngSize(file) {
  const b = fs.readFileSync(file);
  return { w: b.readUInt32BE(16), h: b.readUInt32BE(20) };
}

const CAP = (n, text) => new Paragraph({
  alignment: AlignmentType.CENTER,
  spacing: { after: 240 },
  children: [
    new TextRun({ text: `Figure ${n}. `, bold: true, size: 18, color: MUTE, font: "Calibri" }),
    new TextRun({ text, size: 18, color: MUTE, font: "Calibri", italics: true }),
  ],
});

const RULEP = () => new Paragraph({
  spacing: { before: 60, after: 160 },
  border: { bottom: { style: BorderStyle.SINGLE, size: 6, color: RULE, space: 1 } },
  children: [new TextRun({ text: "" })],
});

const BREAK = () => new Paragraph({ children: [new PageBreak()] });

// find a figure from any run directory
function runFig(pattern) {
  const local = findFigIn(FIGS, pattern);
  if (local) return local;
  const dirs = fs.readdirSync(RUNS).filter(d => fs.statSync(path.join(RUNS, d)).isDirectory());
  for (const d of dirs.sort().reverse()) {
    const fdir = path.join(RUNS, d, "figs");
    if (!fs.existsSync(fdir)) continue;
    const hit = fs.readdirSync(fdir).find(f => f.startsWith(pattern) && f.endsWith(".png"));
    if (hit) return path.join(fdir, hit);
  }
  return null;
}

function RUNIMG(pattern, widthDxa) {
  const src = runFig(pattern);
  if (!src) return P(`[figure ${pattern} not found]`, { color: WARN, italics: true });
  const dims = pngSize(src);
  const wPx = Math.round((widthDxa / 1440) * 96);
  const hPx = Math.round(wPx * (dims.h / dims.w));
  return new Paragraph({
    alignment: AlignmentType.CENTER,
    spacing: { before: 140, after: 80 },
    children: [new ImageRun({
      type: "png", data: fs.readFileSync(src),
      transformation: { width: wPx, height: hPx },
    })],
  });
}

// ---------------------------------------------------------------- sections
const portraitProps = {
  page: {
    size: { width: PAGE_W, height: PAGE_H },
    margin: { top: MARGIN, bottom: MARGIN, left: MARGIN, right: MARGIN },
  },
};
const landscapeProps = {
  page: {
    size: { width: PAGE_W, height: PAGE_H, orientation: PageOrientation.LANDSCAPE },
    margin: { top: 720, bottom: 720, left: 720, right: 720 },
  },
};

const footer = new Footer({
  children: [new Paragraph({
    alignment: AlignmentType.CENTER,
    children: [new TextRun({
      children: ["Dual-Manifold ContextSync   ·   page ", PageNumber.CURRENT],
      size: 16, color: MUTE, font: "Calibri",
    })],
  })],
});

// ================================================================= CONTENT
const cover = [
  new Paragraph({ spacing: { before: 1900, after: 0 },
    children: [new TextRun({ text: "TECHNICAL REPORT", size: 20, color: PLAS, bold: true,
      font: "Calibri", characterSpacing: 60 })] }),
  new Paragraph({ spacing: { before: 200, after: 0 },
    children: [new TextRun({ text: "Dual-Manifold ContextSync", size: 60, bold: true,
      color: INK, font: "Calibri Light" })] }),
  new Paragraph({ spacing: { before: 120, after: 260 },
    children: [new TextRun({
      text: "A quantum–classical hybrid architecture for the stability–plasticity dilemma in long, multi-user LLM chat threads",
      size: 26, color: MUTE, font: "Calibri Light" })] }),
  RULEP(),
  RICH([
    { t: "Complete design, implementation and evaluation. ", b: true },
    { t: "Every number in this report was produced by the accompanying code on the machine described in §9; nothing is estimated, projected or borrowed from the literature except where explicitly attributed." },
  ], { after: 200 }),
  new Paragraph({ spacing: { before: 400, after: 0 },
    children: [new TextRun({ text: "Reference paper", bold: true, size: 20, color: STAB, font: "Calibri" })] }),
  P("Yu-Qin Chen and Shi-Xin Zhang, “Intrinsic Preservation of Plasticity in Continual Quantum Learning”, PRX Quantum 7, 033003 (2026).", { size: 20 }),
  new Paragraph({ spacing: { before: 200, after: 0 },
    children: [new TextRun({ text: "Software", bold: true, size: 20, color: STAB, font: "Calibri" })] }),
  P("Python 3.14 · PyTorch 2.10 · PennyLane 0.44 · Qiskit 2.3 · TensorCircuit-NG 1.9 · FAISS 1.15 · sentence-transformers 5.2 · SQLite. 78 source files, 88 passing tests.", { size: 20 }),
  BREAK(),
];

const toc = [
  H("Contents", HeadingLevel.HEADING_1),
  P("If the list below appears empty, click it once and press F9 (or right-click → Update Field) — Word builds the table of contents on first open.",
    { italics: true, color: MUTE, size: 19 }),
  new TableOfContents("Contents", { hyperlink: true, headingStyleRange: "1-3" }),
  BREAK(),
];

// ---- Part 1 -------------------------------------------------------------
const part1 = [
  H("1  The problem, stated precisely", HeadingLevel.HEADING_1),
  P("A shared chat thread is a hostile environment for a language model. Facts arrive continuously, from several people, and contradict each other. The thread outgrows any context window. The model must keep learning what matters to this particular conversation while never losing what was said four hundred turns ago. Those two requirements pull in opposite directions, and the tension has a name."),

  H("1.1  The stability–plasticity dilemma", HeadingLevel.HEADING_2),
  P("Stability is the ability to retain what has already been learned. Plasticity is the ability to keep absorbing new information. In a conventional neural system both properties are carried by the same object — one parameter vector θ ∈ ℝⁿ — and therefore compete for the same coordinates. Protect the weights and the model stops adapting; let them move and earlier knowledge is overwritten. This is not a tuning problem. It is a consequence of storing both properties in one unbounded Euclidean space."),
  P("Two distinct failure modes follow:"),
  BULLET("Catastrophic forgetting — a gradient step taken for turn 500 destroys the representation that encoded turn 40. The information is gone; no retrieval strategy recovers it."),
  BULLET("Loss of plasticity — the subtler failure. As weights inflate, activations saturate, the effective rank of the representation collapses, and the Fisher information trace decays toward zero. The parameters still exist but have become functionally dead: the network has not forgotten, it has stopped being able to learn."),
  P("A third failure is specific to LLM deployments and is usually mistaken for a context-length problem:"),
  BULLET("Context saturation — the naive fix for forgetting is to retrieve more and put more in the prompt. Attention is then spread thinner across more tokens, and answer quality falls even though the required fact is present in the window. More context is not better context."),

  H("1.2  Six ways an architecture in this space typically dies", HeadingLevel.HEADING_2),
  P("Before proposing anything, it is worth naming the failure modes the design must make structurally impossible rather than merely mitigate."),
  TABLE(
    ["#", "Failure mode", "Why it kills the system"],
    [
      ["F1", "Learning memory into weights", "If a fact from turn 40 lives only in a trained parameter, some later gradient step will overwrite it. Forgetting becomes a property of the design and every mitigation is a patch on a leak."],
      ["F2", "Confusing more context with better context", "Retrieval that maximises recall fills the window and dilutes attention. Saturation is an allocation problem under a hard budget, not a retrieval problem."],
      ["F3", "Putting the quantum circuit on the generation path", "A circuit that must emit tokens, or run per token, is both intractable and unfalsifiable. The quantum component must sit where its output is low-dimensional and its benefit measurable."],
      ["F4", "Assuming quantum solves stability", "It does not. The reference paper is explicit that variational circuits preserve plasticity and still forget catastrophically without an explicit stability mechanism."],
      ["F5", "Ignoring multi-user structure", "In a shared thread, salience is per-participant and often contradictory. A single global relevance score averages users into an incoherent persona."],
      ["F6", "Unfalsifiable evaluation", "Without ground-truth recall probes injected at generation time, “it remembers better” is unmeasurable. Evaluation must be designed before the dataset."],
    ], [6, 34, 60]),

  H("1.3  What the reference paper does and does not establish", HeadingLevel.HEADING_2),
  P("Chen and Zhang compare deep quantum neural networks against classical multilayer perceptrons across thousands of sequential tasks. Their result is specific and it is about plasticity, not memory."),
  BULLET("Classical networks lose plasticity: weight and gradient norms grow without bound, and Tr(F) — the trace of the Fisher information matrix, a direct measure of effective learning capacity — collapses. Performance on new tasks decays continuously."),
  BULLET("Quantum models do not: unitarity confines optimisation to a compact manifold, so norms stay bounded and Haar-averaged learning capacity is provably bounded away from zero, and also bounded above."),
  BULLET("The authors state plainly that this does not solve stability. Both model families still forget previously learned tasks without an explicit stability mechanism, and they name a hybrid — a plastic quantum backbone augmented with classical memory — as the promising direction."),
  CALLOUT("The design consequence",
    "Any architecture that claims quantum computation solves the memory problem is unsound. The correct division of labour is the opposite of the intuitive one: use classical, non-parametric storage for the part that must never be lost, and the quantum manifold for the part that must keep learning forever."),
  BREAK(),
];

// ---- Part 2 -------------------------------------------------------------
const part2 = [
  H("2  The thesis: two manifolds, one narrow interface", HeadingLevel.HEADING_1),
  P("The dilemma is normally posed as one dial on one parameter vector. This architecture refuses that framing and places stability and plasticity on two different mathematical objects with no shared coordinates."),

  H("2.1  ℳ — the stability manifold", HeadingLevel.HEADING_2),
  P("Every semantic unit ever produced in the thread is stored verbatim, together with a frozen embedding and full provenance, in an append-only classical store. Nothing about it is learned, so nothing about it can be unlearned. Forgetting degrades from an irreversible gradient phenomenon into a retrieval miss:"),
  CODE(["P(forget fact f)  =  P(f not selected into context)",
        "                 =  a ranking error — bounded, measurable, fixable"]),
  P("Immutability is enforced by database triggers, not by convention, because a store whose immutability depends on every caller behaving is not immutable."),

  H("2.2  𝒰 — the plasticity manifold", HeadingLevel.HEADING_2),
  P("The only component that learns online is a ten-qubit variational quantum circuit that scores relevance. Its parameters are rotation angles on a compact periodic manifold — a torus, not ℝⁿ:"),
  CODE(["U(θ) ∈ SU(2^N),   ‖U(θ)‖₂ = 1  for every θ",
        "θ wraps into (−π, π] — weights cannot inflate",
        "0 < c₁ ≤ E[Tr F(θ)] ≤ c₂ < ∞  — capacity cannot decay to zero"]),

  H("2.3  The frozen language model", HeadingLevel.HEADING_2),
  P("The generator is a pure conditional model p(y | C) over an assembled context. No gradient path to its weights exists anywhere in the codebase. Catastrophic forgetting of the model's own world knowledge is therefore not mitigated — it is absent."),

  H("2.4  The interface, and why it is deliberately narrow", HeadingLevel.HEADING_2),
  P("The two manifolds meet at exactly one place: the circuit consumes an amplitude-encoded joint state of (query, candidate memory, participant) and emits one scalar relevance in [0,1]. That is the entire coupling. It is small enough to simulate exactly, cheap enough to run per candidate, and — critically — it is drop-in replaceable by a classical scorer with identical inputs, outputs and parameter count, which is what makes every quantum claim in this report falsifiable."),

  H("2.5  The three claims under test", HeadingLevel.HEADING_2),
  TABLE(["Claim", "Mechanism", "Verdict (§7)"], [
    ["Catastrophic forgetting of world knowledge is zero", "Frozen LLM; thread knowledge in append-only ℳ", "Supported by construction"],
    ["Plasticity does not decay over long threads", "Compact unitary manifold; bounded Tr(F)", "Early indicator only — not yet tested at the required horizon"],
    ["Context saturation is bounded", "Budgeted knapsack with a hard token ceiling", "Budget respected; turnover not observable with the current reader"],
    ["The quantum layer is falsifiable", "Parameter-matched classical twin, identical interface", "Tested — and it falsified in the classical arm's favour"],
  ], [30, 38, 32]),
  BREAK(),
];

// ---- diagram A section content (landscape)
const diagA = [
  IMG("DIAG_A_architecture.png", CONTENT_L),
  CAP(1, "The complete architecture. Classical stability on the left, the quantum plasticity manifold on the right, and the single scalar-wide interface between them. Every box corresponds to a module in src/qcs/."),
];

module.exports = { P, H, BULLET, NUM, CODE, TABLE, IMG, CAP, RULEP, BREAK, CALLOUT, RICH, RUNIMG,
  portraitProps, landscapeProps, footer, cover, toc, part1, part2, diagA,
  CONTENT_P, CONTENT_L, INK, MUTE, STAB, PLAS, OKC, WARN,
  Document, Packer, Paragraph, TextRun, HeadingLevel, AlignmentType, LevelFormat,
  convertInchesToTwip, fs, path };
