// Team_14 project report as a Word file (docx-js).
//   node scripts/report/build_team_report.js        (needs `npm install` in qcs/ once)
const fs = require("fs");
const path = require("path");
const {
  Document, Packer, Paragraph, TextRun, ImageRun, Table, TableRow, TableCell, WidthType,
  AlignmentType, BorderStyle, ShadingType, LevelFormat, Footer, PageNumber, TabStopType,
  VerticalAlign, HeadingLevel, PageBreak,
} = require("docx");

const ROOT = path.resolve(__dirname, "..", "..");
const FIG = path.join(ROOT, "figures", "final");
const LOGO = process.env.LOGO_PNG || path.join(ROOT, "figures", "branding", "amrita_logo.png");
const OUT = path.join(ROOT, "deliverables", "Team_14_Dual_Manifold_ContextSync_Report.docx");

const TEAL = "0B5C73", MAG = "A3226A", GRAY = "5B6670", INK = "14191D", LINE = "C9D1D6", HEADFILL = "E8EFF2";
const BODY = "Calibri", HEAD = "Cambria";
const W = 9638; // content width in DXA (A4, 2 cm margins)

// ---------------------------------------------------------------- helpers
function runs(text, base = {}) {
  // **bold** and _italic_ inline markers
  const out = [];
  const re = /(\*\*[^*]+\*\*|_[^_]+_)/g;
  let last = 0, m;
  while ((m = re.exec(text)) !== null) {
    if (m.index > last) out.push(new TextRun({ text: text.slice(last, m.index), ...base }));
    const t = m[0];
    if (t.startsWith("**")) out.push(new TextRun({ text: t.slice(2, -2), bold: true, ...base }));
    else out.push(new TextRun({ text: t.slice(1, -1), italics: true, ...base }));
    last = m.index + t.length;
  }
  if (last < text.length) out.push(new TextRun({ text: text.slice(last), ...base }));
  return out;
}
const P = (text, o = {}) => new Paragraph({
  children: runs(text, { font: BODY, size: o.size || 19, color: o.color || INK }),
  spacing: { after: o.after ?? 70, line: o.line || 240 }, alignment: o.align || AlignmentType.JUSTIFIED,
  keepNext: o.keepNext, indent: o.indent,
});
const H1 = (text) => new Paragraph({
  heading: HeadingLevel.HEADING_1, keepNext: true,
  children: [new TextRun({ text, font: HEAD, size: 28, bold: true, color: TEAL })],
  spacing: { before: 190, after: 70 },
  border: { bottom: { style: BorderStyle.SINGLE, size: 6, color: MAG, space: 2 } },
});
const H2 = (text) => new Paragraph({
  heading: HeadingLevel.HEADING_2, keepNext: true,
  children: [new TextRun({ text, font: HEAD, size: 22, bold: true, color: INK })],
  spacing: { before: 100, after: 40 },
});
const bullets = (items, ref = "bul") => items.map((t) => new Paragraph({
  numbering: { reference: ref, level: 0 }, children: runs(t, { font: BODY, size: 19, color: INK }),
  spacing: { after: 30, line: 240 }, alignment: AlignmentType.LEFT,
}));
const numbered = (items, ref) => items.map((t) => new Paragraph({
  numbering: { reference: ref, level: 0 }, children: runs(t, { font: BODY, size: 19, color: INK }),
  spacing: { after: 30, line: 240 }, alignment: AlignmentType.LEFT,
}));
const eq = (text) => new Paragraph({
  alignment: AlignmentType.CENTER, spacing: { before: 40, after: 80 },
  children: [new TextRun({ text, font: "Cambria Math", size: 21, italics: true, color: INK })],
});
const thin = { style: BorderStyle.SINGLE, size: 4, color: LINE };
function table(rows, widths, o = {}) {
  const fs = o.size || 17;
  const total = widths.reduce((a, b) => a + b, 0);
  return new Table({
    width: { size: total, type: WidthType.DXA }, columnWidths: widths,
    borders: { top: thin, bottom: thin, left: thin, right: thin, insideHorizontal: thin, insideVertical: thin },
    rows: rows.map((row, i) => new TableRow({
      tableHeader: i === 0, cantSplit: true,
      children: row.map((cell, j) => new TableCell({
        width: { size: widths[j], type: WidthType.DXA },
        margins: { top: 35, bottom: 35, left: 80, right: 80 },
        verticalAlign: VerticalAlign.CENTER,
        shading: i === 0 ? { fill: HEADFILL, type: ShadingType.CLEAR, color: "auto" } : undefined,
        children: [new Paragraph({
          alignment: AlignmentType.LEFT, spacing: { after: 0, line: 235 },
          children: runs(String(cell), { font: BODY, size: fs, bold: i === 0 || (j === 0 && o.boldFirst !== false), color: i === 0 ? GRAY : INK }),
        })],
      })),
    })),
  });
}
const gap = (n = 80) => new Paragraph({ children: [], spacing: { after: n } });
function pngSize(file) {
  const b = fs.readFileSync(file);
  return { w: b.readUInt32BE(16), h: b.readUInt32BE(20) };
}
function figure(name, widthCm, caption) {
  const file = path.join(FIG, name);
  const { w, h } = pngSize(file);
  const wpx = Math.round(widthCm * 37.795), hpx = Math.round(wpx * h / w);
  return [
    new Paragraph({
      alignment: AlignmentType.CENTER, keepNext: true, spacing: { before: 60, after: 30 },
      children: [new ImageRun({ type: "png", data: fs.readFileSync(file), transformation: { width: wpx, height: hpx },
        altText: { title: caption, description: caption, name } })],
    }),
    new Paragraph({
      alignment: AlignmentType.CENTER, spacing: { after: 110 },
      children: runs(caption, { font: BODY, size: 17, italics: true, color: GRAY }),
    }),
  ];
}
const none = { style: BorderStyle.NONE, size: 0, color: "FFFFFF" };
const nob = { top: none, bottom: none, left: none, right: none };

// ---------------------------------------------------------------- cover
const logo = fs.readFileSync(LOGO);
const cover = [
  new Paragraph({ children: [], spacing: { before: 500, after: 0 } }),
  new Paragraph({
    alignment: AlignmentType.CENTER, spacing: { after: 700 },
    children: [new ImageRun({ type: "png", data: logo, transformation: { width: 420, height: 143 },
      altText: { title: "Amrita Vishwa Vidyapeetham logo", description: "Amrita Vishwa Vidyapeetham logo", name: "logo" } })],
  }),
  new Paragraph({
    alignment: AlignmentType.CENTER, spacing: { after: 160 },
    children: [new TextRun({ text: "PROJECT REPORT", font: BODY, size: 26, bold: true, color: MAG, characterSpacing: 80 })],
  }),
  new Paragraph({
    alignment: AlignmentType.CENTER, spacing: { after: 160 },
    children: [new TextRun({ text: "Dual-Manifold ContextSync", font: HEAD, size: 68, bold: true, color: TEAL })],
  }),
  new Paragraph({
    alignment: AlignmentType.CENTER, spacing: { after: 360 }, indent: { left: 700, right: 700 },
    border: { bottom: { style: BorderStyle.SINGLE, size: 10, color: MAG, space: 14 } },
    children: [new TextRun({
      text: "Keeping stability and plasticity together with a classical memory and a 10-qubit variational circuit",
      font: HEAD, size: 28, color: GRAY, italics: true })],
  }),
  new Paragraph({ children: [], spacing: { after: 240 } }),
  new Table({
    width: { size: 7600, type: WidthType.DXA }, columnWidths: [2300, 5300], alignment: AlignmentType.CENTER,
    borders: { top: thin, bottom: thin, left: thin, right: thin, insideHorizontal: thin, insideVertical: thin },
    rows: [["Subject", "Quantum Computing and Algorithm"], ["Team", "Team_14"], ["Project", "A_14 · Neuro-Plasticity"], ["Date", "October 2026"]]
      .map(([a, b]) => new TableRow({
        children: [
          new TableCell({ width: { size: 2300, type: WidthType.DXA }, margins: { top: 110, bottom: 110, left: 160, right: 120 },
            shading: { fill: HEADFILL, type: ShadingType.CLEAR, color: "auto" }, verticalAlign: VerticalAlign.CENTER,
            children: [new Paragraph({ children: [new TextRun({ text: a, font: BODY, size: 24, bold: true, color: TEAL })] })] }),
          new TableCell({ width: { size: 5300, type: WidthType.DXA }, margins: { top: 110, bottom: 110, left: 160, right: 120 },
            verticalAlign: VerticalAlign.CENTER,
            children: [new Paragraph({ children: [new TextRun({ text: b, font: BODY, size: 24, color: INK })] })] }),
        ] })),
  }),
];

// ---------------------------------------------------------------- body
const body = [];
const add = (...x) => x.flat().forEach((e) => body.push(e));

add(H1("Abstract"));
add(P("A language model that follows a long group chat must remember facts stated hundreds of messages ago, and it must accept those facts when someone changes them. These two needs pull against each other; the tension is the stability-plasticity dilemma. This project builds a scorer that ranks earlier chat messages by how useful they are for answering a new question, and it keeps the two needs apart. Old facts are held in an append-only memory that training cannot edit. A 10-qubit variational circuit, simulated on an ordinary computer, learns what makes a message relevant."));
add(P("We tested it on a generated dataset of 18,000 chat turns, over 120 sequential tasks and three seeds, against a linear model of the same size and a larger network. The original circuit measured one qubit and ranked below the linear model (test MRR 0.861 against 0.974). Measuring all ten qubits with learned weights closed the gap to 0.974, a tie. Every model remembered old facts, including facts more than 400 turns old. The circuit also kept more of its ability to learn a reversed rule after 120 tasks (about 0.84 against about 0.2 for the classical models at their normal learning rates). A fairness check showed that classical models recover part of that loss when tuned, so the supported conclusion is faster and steadier adaptation. The results cover 120 tasks, a simulated circuit and a small test set."));
add(P("**Keywords:** continual learning, stability-plasticity dilemma, variational quantum circuit, amplitude encoding, append-only memory, retrieval ranking", { align: AlignmentType.LEFT }));

add(H1("1. Introduction"));
add(P("Imagine a group chat with five people and about 600 messages. Early on someone writes \"my flight is on the 5th.\" Much later they write \"it moved to the 9th.\" Hundreds of messages after that, someone asks \"when is my flight?\" A helper that reads this chat must answer the 9th. It must still know the old facts that never changed, and it must prefer the newest version of the ones that did."));
add(P("The helper faces the stability-plasticity dilemma. **Stability** means keeping what was learned earlier. **Plasticity** means staying able to learn new or changed information. In an ordinary network both live in the same weights: hold them still and the model cannot adapt; let them move and older knowledge is overwritten. Three failures follow."));
add(table([
  ["Failure", "What happens"],
  ["Catastrophic forgetting", "Training on a late message overwrites what the model stored about an early one. Once lost, no retrieval method can bring it back."],
  ["Loss of plasticity", "Weights grow and units saturate over time. The parameters remain, but the model can no longer learn much from new data."],
  ["Context overload", "Putting more text into the prompt does not help. Attention spreads thin and answers get worse even when the right fact is in the window."],
], [2400, 7238]));
add(gap(60));
add(P("**Objectives.** (1) Build a system that ranks earlier messages by relevance and keeps old knowledge safe. (2) Use a 10-qubit variational circuit as the part that learns, and test whether it keeps its ability to learn better than classical models. (3) Measure stability, plasticity and the two together with tests a reader can check. (4) Report results as they came out, including those that do not favour the circuit."));
add(P("**Contributions.** A complete pipeline from raw chat to a relevance score, with a classical memory and a simulated quantum circuit; a generated dataset in which the correct answer to every question is known exactly; a readout change (all ten qubits instead of one) that raised test MRR from 0.861 to 0.974; and a comparison with a size-matched linear model and a larger network, with a fairness check on learning rates."));

add(H1("2. Background"));
add(H2("2.1 Continual learning"));
add(P("In continual learning a model meets tasks one after another and cannot go back over all old data. Common remedies are replay (mix stored old examples into new training), regularisation (penalise changes to parameters that mattered; Elastic Weight Consolidation measures this with the Fisher information) and memory-based methods (keep knowledge outside the weights). Forgetting is usually reported as backward transfer, the change in accuracy on old tasks after new ones are learned."));
add(H2("2.2 The quantum computing used here"));
add(...bullets([
  "**Qubit.** A bit is 0 or 1. A qubit is described by two amplitudes a and b with a² + b² = 1. Measuring gives 0 with probability a² and 1 with probability b².",
  "**Many qubits.** n qubits need 2ⁿ amplitudes, so ten qubits need 1,024. A list of 1,024 numbers with total squared length 1 is exactly a 10-qubit state.",
  "**Gates.** A gate changes the amplitudes and is unitary: it rotates the state vector without changing its length, so the state cannot blow up or vanish. A two-qubit gate also creates entanglement, after which the qubits cannot be described separately.",
  "**Measurement as a number.** For one qubit, the Z-expectation is P(reads 0) minus P(reads 1), between −1 and +1, computed directly from the amplitudes.",
  "**Variational circuit.** A circuit with adjustable angles, trained by gradient descent on a classical computer like a network's weights.",
  "**Simulation.** No quantum hardware is used. The state is an array of 1,024 numbers and each gate is a matrix multiplied into it. The result is exact; the cost doubles with every added qubit.",
]));
add(eq("|ψ⟩ = Σᵢ xᵢ |i⟩ ,   Σᵢ xᵢ² = 1   (i = 0 … 1023)"));
add(H2("2.3 Related work"));
add(table([
  ["Work", "Contribution", "What it leaves open"],
  ["McCloskey and Cohen (1989)", "Named catastrophic interference", "Describes the problem, no way to prevent it"],
  ["Kirkpatrick et al. (2017), EWC", "Penalty that anchors important weights", "Knowledge still sits in the weights; sensitive to settings"],
  ["Lopez-Paz and Ranzato (2017), GEM", "Replay with gradient constraints; BWT and FWT", "Memory grows; says little about plasticity"],
  ["Dohare et al. (2024)", "Deep networks gradually lose the ability to learn", "Fixes are add-ons such as resets or noise"],
  ["McClean (2018); Cerezo (2021)", "Barren plateaus and their link to the observable", "Points to local readouts and shallow circuits"],
  ["Mitarai (2018); Schuld (2019)", "Parameter-shift rule for exact gradients", "Needs involutory generators, which a general SU(4) gate lacks"],
  ["Lewis et al. (2020), RAG", "Retrieval as external memory", "Ranking is fixed and does not adapt"],
  ["Liu et al. (2023)", "Long contexts degrade even if the fact is present", "Confirms overload, gives no inclusion policy"],
  ["Chen and Zhang (2026)", "Quantum models keep plasticity on a compact manifold", "States that this does not solve forgetting"],
], [2700, 3600, 3338], { size: 16 }));
add(gap(50));
add(P("The last row is the direct basis of this project. Chen and Zhang argue that a circuit with unitary gates and angle parameters keeps its learning capacity, but also state that both quantum and classical models forget without a separate memory, and name a hybrid of a plastic quantum part and a classical memory as the promising direction. This project builds that hybrid."));

add(H1("3. Proposed approach"));
add(P("The design gives stability and plasticity to two different objects so that they stop competing for the same parameters. **Stability: the memory M.** Every message is split into units of about 40 tokens and stored word for word in a database that only adds rows; triggers reject any edit or delete, and summaries are added beside the originals, never in place of them. Nothing in it is trained, so a gradient step cannot touch it, and a failure to use an old fact is a ranking mistake, not a loss. **Plasticity: the circuit U.** A 10-qubit variational circuit with 1,080 angles. Angles repeat every full turn, so after each task they are folded back into the range −π to π and cannot grow without limit, and every gate is unitary. Chen and Zhang argue this keeps learning capacity from fading; we treat that as a claim to test."));
add(P("The two sides meet at one point: the circuit receives a feature vector for a (question, candidate) pair and returns one relevance score between 0 and 1. A circuit this small can be simulated exactly, and any classical scorer with the same interface can replace it, so every comparison changes one component and nothing else. We test three hypotheses:"));
add(...numbered([
  "**H1.** Old facts stay retrievable however long training goes on, because the memory is outside the learning path.",
  "**H2.** The circuit ranks as well as a classical scorer with the same number of parameters.",
  "**H3.** After many tasks the circuit can still learn a rule it has never seen, and does so better than the classical scorers.",
], "n_hyp"));

add(H1("4. System architecture"));
add(...figure("architecture.png", 14.4, "Figure 1. System architecture. Teal: classical and add-only. Pink: the quantum part, the only component that is trained. Green and grey: delivery, with the language model frozen."));
add(P("Data flows downward through four bands. The stability band ingests and segments text, encodes it with a frozen encoder, stores it in memory M and retrieves the top candidates. The interface band builds a 1,024-number feature vector and loads it into a 10-qubit state. The plasticity band applies the circuit and reads out a score. The delivery band picks facts within a token budget for a frozen language model and compares answers with the known truth. The only upward flow is the training signal, which changes the circuit angles and readout weights and nothing else."));

add(H1("5. Dataset"));
add(P("We generate the chats so that the correct answer to every question is known exactly. Each thread is planned first: who states which fact, which later turn changes it, and which turn asks about it and how many turns back. Templates then turn the plan into messages, so the wording can never add or remove a fact."));
add(table([
  ["Turn type", "What it is", "Why it is there"],
  ["fact", "A participant states an (entity, attribute, value)", "The target to retrieve"],
  ["update", "A later turn changes that fact", "Creates old-versus-new traps"],
  ["probe", "A question answerable from stated facts", "The measurement point"],
  ["distractor", "Mentions a real entity with no usable value", "Defeats keyword matching"],
  ["filler", "Ordinary conversation", "Dilutes the thread"],
], [1500, 4600, 3538]));
add(gap(60));
add(table([
  ["Property", "Value", "Property", "Value"],
  ["Threads", "30 (24 train, 6 sealed test)", "Probes", "1,800 (60 per thread)"],
  ["Turns per thread / total", "600 / 18,000", "Probe-to-fact distance", "10, 50, 200 or 500 turns"],
  ["Participants / facts per thread", "5 / 60", "Filler / distractor turns", "55% / 25%"],
  ["Facts later replaced", "about 35% (update threads)", "Oracle / random recall", "1.000 / 0.042"],
], [2500, 2319, 2400, 2419], { size: 16 }));
add(gap(60));
add(P("Three kinds of thread rotate through the data: all facts owned by one participant; facts spread over several owners, with someone else asking; and adversarial threads, the only ones with updates, in which about 35% of facts are later replaced by a correction. The invented example below (not from the dataset) shows how the pieces fit."));
add(table([
  ["Turn", "Speaker", "Message", "Type"],
  ["12", "Asha", "My flight is on the 5th.", "fact"],
  ["85", "Ben", "Has anyone tried the new cafe?", "filler"],
  ["140", "Asha", "My flight moved to the 9th.", "update"],
  ["540", "Asha", "When is my flight?", "probe"],
], [800, 1200, 5638, 2000], { size: 16 }));
add(gap(60));
add(P("The correct answer is turn 140; turn 12 is a stale fact that is just as similar to the question, so a similarity-only scorer may rank it first. **Training view.** The stream uses the first 12 training threads (720 probes). For each probe the memory returns the 64 best earlier candidates (earlier turns only). A candidate is positive if it states the needed fact (about 1.6%); about 1.4% are stale and act as hard negatives. Six probes form one task, so the model sees 120 tasks in order and trains on each once. The 6 sealed test threads give 360 probes and were not used to choose any setting."));

add(H1("6. The pipeline, step by step"));
add(P("A question and one candidate pass through thirteen steps. Only steps 10 and 11 are trained.", { keepNext: true }));
add(table([
  ["Step", "What happens", "Output", "Trained?"],
  ["1. Segment", "Cut each turn into units of about 40 tokens, no text lost", "Text units", "No"],
  ["2. Encode", "Frozen sentence encoder (MiniLM)", "384 numbers", "No"],
  ["3. Whiten", "Spread the numbers evenly; fitted once (256 components, shrinkage 0.1)", "256 numbers", "Fitted once"],
  ["4. Project", "Small trained projection, then frozen; recall@20 0.879 to 0.996", "256 numbers", "Once"],
  ["5. Store", "Write units verbatim; summaries every 50 turns; supersession links", "Database rows", "No"],
  ["6. Retrieve", "FAISS search over earlier units only; top candidates (64 in training)", "64 candidates", "No"],
  ["7. Features", "Join four blocks of 256 numbers (see below)", "1,024 numbers", "No"],
  ["8. Normalise", "Balance the blocks, scale the vector to length 1", "Unit vector", "No"],
  ["9. Load qubits", "Amplitude encoding of the 1,024 numbers", "10-qubit state", "No"],
  ["10. Circuit", "Eight layers of two-qubit gates", "Evolved state", "Yes"],
  ["11. Read out", "Measure every qubit, weighted sum, sigmoid", "Score r in [0,1]", "Yes"],
  ["12. Assemble", "Pick facts by value per token; drop or label stale ones", "Prompt, 3,072 tokens", "No"],
  ["13. Answer", "A frozen language model reads the prompt", "Text", "No"],
], [1350, 5088, 1900, 1300], { size: 16 }));
add(gap(60));
add(P("**Feature blocks.** Question (256): the projected question embedding. Candidate (256): the projected candidate embedding. Interaction (256): the two multiplied entry by entry, whose sum is their similarity. Relational (256): how old the candidate is in turns and seconds, whether the same user wrote it, who owns the fact, what the entity graph says (for example that it was superseded) and the asking user's salience. Four blocks of 256 make 1,024 = 2¹⁰, which is why the circuit has 10 qubits."));
add(P("**Memory.** An episodic tier holds every unit verbatim and rejects UPDATE and DELETE. A semantic tier adds summaries every 50 turns beside the sources. An entity graph records supersedes, contradicts and shares-entity links; a contradiction is stored, not resolved. Salience is a running average over each participant's own turns, one 64-number vector per thread and user. Retrieval looks at earlier turns only, so the model cannot see the future, and its recall is 1.000 on the project corpus. **Prompt.** The budget is 3,072 tokens: 256 for the system text, 256 pinned and 768 for the most recent turns. Candidates are chosen greedily by value per token (fact 1.0, summary 0.7, filler 0.35) with a redundancy penalty of 0.35; superseded facts are dropped or labelled. In our experiments an extractive reader stands in for the language model, so quality is measured by how highly the scorer ranks the correct fact."));

add(H1("7. The quantum circuit"));
add(...figure("circuit.png", 15.8, "Figure 2. The complete circuit: amplitude encoding, eight layers of two-qubit gates (each a row of five gates and a staggered row of four), a measurement of every qubit with learned weights, and the score. Bottom left: one gate opened up. Bottom right: what is trained."));
add(P("**Why ten qubits.** The feature vector has 1,024 numbers and 2¹⁰ = 1,024, so ten qubits hold it exactly. The qubit count is set by the feature size. **Loading the data.** The unit-length vector x becomes the amplitudes of the state, |ψ₀⟩ = Σ xᵢ|i⟩, where |i⟩ is the basis state whose ten-bit label is i. This amplitude encoding loses nothing. On real hardware such a state is costly to prepare; in simulation it is set directly."));
add(P("**The gate.** Each gate acts on two neighbouring qubits and is a general 4×4 unitary with 15 free angles, one for each non-identity product Gₖ of two Pauli matrices. On hardware a compiler would write it as three CNOTs plus single-qubit rotations (Figure 2, bottom left)."));
add(eq("V(θ) = exp( −(i/2) Σₖ θₖ Gₖ ) ,   k = 1 … 15"));
add(P("**Layers.** One layer has nine gates: a row of five on the pairs (0,1), (2,3), (4,5), (6,7), (8,9), then a staggered row of four on (1,2), (3,4), (5,6), (7,8). Staggering lets information spread along the register. Eight layers give 72 gates and 8 × 9 × 15 = 1,080 trainable angles, which start near zero so the circuit begins close to the identity and the first gradients are informative. A gate on qubits 0 and 1 needs several staggered layers before it can influence a distant qubit; below that depth some angles would have exactly zero gradient, so the configuration enforces a minimum depth of five for ten qubits, and we use eight."));
add(P("**Readout.** Each qubit is measured with its Z-expectation and the ten readings are combined with learned weights. We read qubits separately and add the results, not as one measurement across all ten. Local measurements are the standard choice for avoiding barren plateaus (vanishing gradients) in larger circuits. At 10 qubits our own check found no plateau for the global measurement either (an earlier measurement that suggested one had computed a product of marginals by mistake, and was corrected in Phase 4), so the choice rests on theory and on the light-cone argument, not on a measured failure."));
add(eq("z = Σᵢ wᵢ ⟨Zᵢ⟩ ,   r = σ( β z + b )   (i = 0 … 9)"));
add(P("**Why we changed the readout.** The original design read only qubit 9, whose output is a fixed quadratic function of the 1,024 inputs. The similarity signal in the data is a straight (linear) sum of one input block, which one quadratic function cannot reproduce well, and the original circuit ranked clearly below the linear model. Learning a weight for each of ten readings gives a richer mix and raised test MRR from 0.861 to 0.974. This is our explanation; the experiments support it but we have not proved it. **What is trained:** 1,080 circuit angles, 10 readout weights, a scale β and a bias b (1,092 in all). Gradients come from automatic differentiation through the simulation, because the parameter-shift rule needs gates of a special form that a general two-qubit gate lacks."));

add(H1("8. Training and the stability mechanisms"));
add(P("Each probe comes with labelled candidates: 1 for a candidate stating the needed fact, 0 for the rest, with up to eight negatives per probe and stale facts picked first. The loss is binary cross-entropy on r, optimised with Adam (learning rate 0.02 for the circuit), three epochs per task, batch size 32. Five protections guard the learning part:"));
add(table([
  ["Mechanism", "What it does", "Setting"],
  ["Replay", "Keeps a reservoir of old examples and mixes an equal number into every batch", "Buffer 8,000; ratio 0.5"],
  ["Fisher anchor", "Penalises moving the angles that mattered for earlier tasks; importance is the quantum Fisher information, estimated by finite differences of the state", "Weight 0.01"],
  ["Trust region", "If any angle moves more than the limit during a task, the whole movement is scaled back", "0.35 per task"],
  ["Angle wrapping", "Folds each angle back into −π to π after every task", "Every task"],
  ["Drift detector", "Compares the score distribution with a reference; when it shifts, the anchor is re-based so it does not pin the model to its first tasks", "Threshold 0.25"],
], [1700, 6238, 1700], { size: 16 }));
add(gap(40));
add(eq("Ω(θ, θ*) = ½ Σₖ Fₖ (θₖ − θ*ₖ)²"));
add(P("Fₖ is the Fisher importance of angle k and θ* the angles saved at the last checkpoint, taken every 10 tasks and whenever the drift detector fires."));
add(P("**How stability is solved.** At two levels. First, the knowledge itself: the memory stores every unit verbatim, cannot be edited, is extended with summaries and a supersession graph, and is untouched by training, so an old fact stays available however long training goes on. Second, the scorer: replay, the Fisher anchor, the trust region and the drift detector protect the circuit against forgetting what \"relevant\" means. **How plasticity is addressed.** In an ordinary network weights can grow without limit, units saturate and learning slows. The circuit's angles wrap around a circle and its gates are unitary, so neither happens. The base paper argues this keeps capacity from fading, and says it does not prevent forgetting, which is why the memory exists. We test it directly: after a chosen number of tasks the model is copied and trained for a few steps on a reversed rule, in which the replaced fact is now correct; how well the copy learns is the plasticity score."));
add(P("The classical models are trained identically (same replay, an ordinary Fisher anchor, the same trust region), each at its own best learning rate: 0.02 for the circuit, 0.1 for the linear model, 0.02 for the network."));

add(H1("9. Experimental setup"));
add(table([
  ["Model", "Trainable parameters", "Description"],
  ["Quantum, 10-qubit readout (final)", "1,080 angles + 12", "The circuit of Section 7, read out on all ten qubits"],
  ["Quantum, 1-qubit readout (original)", "1,080 angles + 2", "The same circuit read out on qubit 9 only"],
  ["Linear", "1,025", "One linear layer on the 1,024 features; closest classical size match"],
  ["MLP", "65,665", "One hidden layer of 64 units; a larger capacity reference"],
  ["Cosine / Recency / Oracle", "0", "No learning (similarity; most recent first); the oracle knows the labels"],
], [3000, 1900, 4738], { size: 16 }));
add(gap(60));
add(table([
  ["Metric", "Meaning"],
  ["MRR", "Mean reciprocal rank: 1 divided by the rank of the first correct candidate, averaged over probes"],
  ["Rank-1; stale-above-current", "Share of probes with the correct candidate first; share where a replaced fact outranks the current one (lower is better)"],
  ["Stability score", "MRR on the first 30 tasks, measured after all 120 are learned"],
  ["Plasticity score", "MRR on held-out probes after 40 gradient steps on the reversed rule"],
  ["Combined score", "Harmonic mean of stability and plasticity, so a model must do well on both"],
], [2700, 6938], { size: 16 }));
add(gap(60));
add(...bullets([
  "120 tasks of 6 probes, three seeds (7, 17, 27), each model at its own tuned learning rate; results are means ± standard deviations. Settings were chosen on the training stream only; the sealed test split (6 threads, 360 probes) gives the final ranking numbers.",
  "**Forgetting matrix:** every 10th task, re-test on every past task, with no value carried forward. **Reversed-rule test:** after 1, 30, 60, 90 and 120 tasks, a copy is trained for 40 Adam steps on probes with a stale fact (stale fact now labelled correct) and scored on held-out probes of that kind.",
  "**Distance test:** test probes grouped by turns between probe and needed fact, against a baseline that sees only the last 100 turns. **Significance:** paired bootstrap over test threads (5,000 resamples); with 6 threads the intervals are wide. **Fairness check:** the reversed-rule test at six learning rates (0.001 to 0.3) and up to 80 steps, keeping each model's best.",
  "Python 3.14 and PyTorch on a CPU. The circuit is the project's own batched state-vector simulator, checked against PennyLane at fidelity 1.000; the repository's automated tests pass.",
]));

add(H1("10. Results"));
add(P("All numbers are means over three seeds after 120 tasks unless stated. **10.1 Ranking on the sealed test split.** The final circuit ties the linear model and clearly beats the original circuit; the network is the most accurate.", { keepNext: true }));
add(table([
  ["Model", "MRR", "Rank-1", "Stale above current ↓"],
  ["Quantum, 10-qubit readout", "0.974 ± 0.005", "0.951 ± 0.008", "0.043 ± 0.019"],
  ["Quantum, 1-qubit readout", "0.861 ± 0.028", "0.778 ± 0.039", "0.052 ± 0.019"],
  ["Linear", "0.974 ± 0.001", "0.951 ± 0.001", "0.000"],
  ["MLP", "0.995 ± 0.001", "0.990 ± 0.003", "0.000"],
  ["Cosine (no learning)", "0.910", "0.831", "0.130"],
], [3400, 2000, 2000, 2238], { size: 17 }));
add(gap(60));
add(table([
  ["Comparison (MRR)", "Difference", "95% interval", "Significant"],
  ["10-qubit minus 1-qubit", "+0.114", "+0.098 to +0.131", "Yes"],
  ["10-qubit minus linear", "+0.001", "−0.007 to +0.009", "No"],
  ["10-qubit minus MLP", "−0.020", "−0.030 to −0.014", "Yes"],
  ["Linear minus MLP", "−0.021", "−0.025 to −0.017", "Yes"],
], [3400, 2000, 2400, 1838], { size: 17 }));
add(gap(60));
add(P("**10.2 Stability: remembering the old.** Figure 3 re-tests every past task as training continues; yellow means remembered. No model forgets catastrophically, and the 1-qubit circuit is weaker throughout. Figure 4 groups test probes by how long ago the needed fact was stated: a model limited to the last 100 turns drops to 0 beyond 200 turns, while every model that uses the memory still finds facts older than 400 turns. Probes without and with an outdated fact among their candidates are handled about equally well (0.982 and 0.958 for the 10-qubit circuit). For facts older than 400 turns the scores are 0.966 (10-qubit), 0.863 (1-qubit), 0.942 (linear) and 0.988 (MLP); the mean over all past tasks after 120 tasks is 0.981, 0.889, 0.990 and 0.998.", { keepNext: true }));
add(...figure("forgetting_matrix.png", 14.0, "Figure 3. Forgetting matrices (seed 7) and mean score over all past tasks (3 seeds)."));
add(...figure("memory_distance.png", 14.0, "Figure 4. Left: score by distance back to the needed fact. Right: probes without (light) and with (dark) an outdated fact among the candidates."));
add(P("**10.3 Learning new things.** Every model starts untrained and learns the task within about 15 tasks. After that a trained model already does well on tasks it has never trained on, so this test alone cannot separate the models. On probes that have an outdated fact among their candidates (115 of the 360 test probes, all from update threads), the 10-qubit circuit scores 0.958 MRR, against about 0.89 for plain similarity search (linear 0.970, MLP 0.992).", { keepNext: true }));
add(...figure("learning_new.png", 13.8, "Figure 5. Left: score on each task before training on it. Right: score on probes with an outdated fact among the candidates."));
add(P("**10.4 Plasticity: learning a reversed rule.** This is the sharper check. At the start all models learn the reversed rule well (0.69 to 0.90). After 120 tasks linear and MLP have lost most of that ability (0.194 and 0.225; the MLP varies strongly across seeds, sd 0.224) while both quantum models keep it (0.835 and 0.846).", { keepNext: true }));
add(...figure("plasticity.png", 13.8, "Figure 6. Left: score on the reversed rule after 40 gradient steps, as training goes on. Right: adaptation speed after 120 tasks."));
add(P("**10.5 Both together.** Figure 7 plots stability against plasticity as training goes on. The classical models drift toward high stability and low plasticity; the 10-qubit circuit stays nearer the top right, where both are high.", { keepNext: true }));
add(...figure("dilemma.png", 13.8, "Figure 7. Left: path of each model through the stability-plasticity plane (small dots after 30, 60, 90 tasks; large dot after 120). Right: combined score over training."));
add(table([
  ["Model", "Test MRR", "Stability", "Plasticity", "Combined"],
  ["Quantum, 10-qubit readout", "0.974 ± 0.005", "0.967 ± 0.006", "0.835 ± 0.030", "0.896 ± 0.016"],
  ["Quantum, 1-qubit readout", "0.861 ± 0.028", "0.873 ± 0.033", "0.846 ± 0.055", "0.859 ± 0.041"],
  ["Linear", "0.974 ± 0.001", "0.982 ± 0.001", "0.194 ± 0.008", "0.323 ± 0.011"],
  ["MLP", "0.995 ± 0.001", "0.998 ± 0.001", "0.225 ± 0.224", "0.316 ± 0.277"],
], [3000, 1650, 1650, 1669, 1669], { size: 17 }));
add(gap(80));
add(P("**10.6 Fairness check.** The plasticity numbers use each model's own training learning rate, so a critic could say the classical models just need another step size. Figure 8 repeats the test at six learning rates and keeps each model's best.", { keepNext: true }));
// figure + text side by side
{
  const file = path.join(FIG, "fairness.png");
  const { w, h } = pngSize(file);
  const wpx = Math.round(8.0 * 37.795), hpx = Math.round(wpx * h / w);
  add(new Table({
    width: { size: W, type: WidthType.DXA }, columnWidths: [4700, 4938],
    borders: { ...nob, insideHorizontal: none, insideVertical: none },
    rows: [new TableRow({ cantSplit: true, children: [
      new TableCell({ width: { size: 4700, type: WidthType.DXA }, borders: nob, verticalAlign: VerticalAlign.CENTER, children: [
        new Paragraph({ alignment: AlignmentType.CENTER, children: [new ImageRun({ type: "png", data: fs.readFileSync(file), transformation: { width: wpx, height: hpx },
          altText: { title: "Figure 8", description: "Reversed-rule score at each model's best learning rate", name: "fairness" } })] }),
        new Paragraph({ alignment: AlignmentType.CENTER, children: runs("Figure 8. Reversed-rule score at each model's best learning rate, after 40 steps (light) and 80 steps (dark).", { font: BODY, size: 17, italics: true, color: GRAY }) }),
      ] }),
      new TableCell({ width: { size: 4938, type: WidthType.DXA }, borders: nob, verticalAlign: VerticalAlign.CENTER, margins: { left: 160 }, children: [
        P("The classical models recover a good part of the loss, but not all of it. With the best rate and 80 steps, linear reaches 0.72 and the MLP 0.71 (seeds 0.95, 0.29, 0.89). The circuit reaches 0.97, and is already at 0.86 after 40 steps against 0.49 and 0.55.", { align: AlignmentType.LEFT }),
        P("The circuit is therefore faster and steadier. The classical models can still learn the new rule.", { align: AlignmentType.LEFT }),
      ] }),
    ] })],
  }));
}

add(H1("11. Discussion"));
add(P("The memory keeps old facts available, the circuit matches a classical scorer of the same size, and the circuit keeps more of its ability to learn a changed rule. The plasticity gain is real but smaller than the first measurement suggested.", { keepNext: true }));
add(table([
  ["Claim", "Status", "Evidence"],
  ["Knowledge in the memory cannot be erased by training", "Supported", "Append-only store; attempted edits rejected by the database"],
  ["Reading out all qubits closes the ranking gap", "Supported", "+0.114 MRR on the test split; interval excludes 0"],
  ["The circuit matches a linear model of the same size", "Supported", "Difference 0.001, interval −0.007 to +0.009"],
  ["The circuit ranks better than the classical models", "Not supported", "The MLP scores higher (0.995 against 0.974)"],
  ["The circuit adapts to a new rule faster and more steadily", "Supported, with a caveat", "0.97 against 0.72 and 0.71 at each model's best learning rate"],
  ["Stale facts are kept below current ones", "Mostly", "0.043 for the circuit, 0.000 classical, 0.130 plain similarity"],
  ["The Fisher trace shows capacity is preserved", "Not supported", "Noisy; the MLP's did not fall"],
  ["Plasticity holds over 1,000 or more tasks", "Not tested", "Only 120 tasks were run"],
], [4000, 1800, 3838], { size: 16 }));
add(gap(60));
add(P("**Memory.** The no-memory baseline scores 0 on facts older than 200 turns, while every model using the memory still finds facts older than 400 turns. This is the clearest effect in the project and it comes from the memory, not the circuit. **Ranking.** The original circuit trailed because its single readout could not express the similarity signal; with ten readings it matches a linear model of similar size, while the larger MLP is better, so there is no ranking advantage. **Plasticity.** At normal learning rates the classical models become very stable but much less able to learn a reversed rule, and the circuit keeps that ability; when tuned, the classical models recover, so the difference is speed and consistency. **Base paper.** The direction agrees with Chen and Zhang, but we do not claim to reproduce their result: their effect appears over horizons far longer than 120 tasks, and the Fisher trace they rely on was noisy here, so we rely on the direct reversed-rule test."));

add(H1("12. Limitations and future work"));
add(...bullets([
  "**Short horizon:** 120 tasks; whether the plasticity advantage holds over thousands of tasks is untested. **Small test set:** 6 threads, so significance tests are coarse.",
  "**Simulation only:** the circuit is simulated exactly, with no hardware or finite-shot run, and costs about 135 times the compute of the linear model. **Stand-in reader:** an extractive reader replaces a language model, so ranking quality is measured, not generated answers.",
  "**One definition of plasticity:** the reversed rule is one defined test, not a live conversation shift. **Stale facts:** the circuit ranks a replaced fact first in 4.3% of cases where classical models never do. **Synthetic data:** real conversations may behave differently. **Seed variation:** the original circuit varied between seeds (sd 0.028 in test MRR), the MLP's plasticity score even more.",
]));
add(P("**Future work.** (1) Run 1,000 to 3,000 tasks on three seeds and track the reversed-rule score at checkpoints. (2) Generate more held-out threads. (3) Simulate finite shots and noise for the 10-qubit readout (ten measurements per score). (4) Replace the extractive reader with a real frozen language model. (5) Add a one-command reproduction script and a scaling study over 8, 10 and 12 qubits."));

add(H1("13. Conclusion"));
add(P("This project built a complete pipeline for ranking earlier chat messages in long, multi-user threads, with an append-only memory that holds old facts and a simulated 10-qubit variational circuit that learns relevance. The memory did its job: facts more than 400 turns old were still found, where a model limited to recent turns found none. Reading out all ten qubits raised test MRR from 0.861 to 0.974, a tie with a linear model of the same size, although a larger network still ranks better. After 120 tasks the circuit also kept more of its ability to learn a reversed rule than the classical models did at their normal learning rates; when those models were tuned they recovered part of the difference, so the advantage is in speed and steadiness."));
add(P("The work shows that a small quantum circuit can serve as the learning part of a hybrid system without losing ranking quality and holds its adaptability over this horizon. It shows no ranking advantage and does not settle what happens over thousands of tasks, on real hardware or with a real language model."));

add(H1("References"));
{
  const refs = [
    "McCloskey, M. and Cohen, N. J. (1989). Catastrophic interference in connectionist networks: the sequential learning problem. Psychology of Learning and Motivation, 24.",
    "Kirkpatrick, J. et al. (2017). Overcoming catastrophic forgetting in neural networks. PNAS, 114(13).",
    "Lopez-Paz, D. and Ranzato, M. (2017). Gradient episodic memory for continual learning. NeurIPS 30.",
    "Dohare, S. et al. (2024). Loss of plasticity in deep continual learning. Nature, 632.",
    "McClean, J. R. et al. (2018). Barren plateaus in quantum neural network training landscapes. Nature Communications, 9.",
    "Cerezo, M. et al. (2021). Cost function dependent barren plateaus in shallow parametrized quantum circuits. Nature Communications, 12.",
    "Mitarai, K. et al. (2018). Quantum circuit learning. Physical Review A, 98.",
    "Schuld, M. et al. (2019). Evaluating analytic gradients on quantum hardware. Physical Review A, 99.",
    "Lewis, P. et al. (2020). Retrieval-augmented generation for knowledge-intensive NLP tasks. NeurIPS 33.",
    "Liu, N. F. et al. (2023). Lost in the middle: how language models use long contexts. TACL.",
    "Chen, Y.-Q. and Zhang, S.-X. (2026). Intrinsic preservation of plasticity in continual quantum learning. PRX Quantum, 7, 033003.",
  ];
  add(...refs.map((t) => new Paragraph({
    numbering: { reference: "n_refs", level: 0 }, alignment: AlignmentType.LEFT,
    spacing: { after: 20, line: 230 }, children: runs(t, { font: BODY, size: 17, color: INK }),
  })));
  add(P("Taken from the project's literature review; check page numbers and venues against the originals before submission.", { size: 17, color: GRAY, align: AlignmentType.LEFT }));
}

add(H1("Appendix A. Glossary"));
add(table([
  ["Term", "Meaning", "Term", "Meaning"],
  ["Stability", "Keeping what was learned earlier", "Unitary", "A transformation that keeps vector length unchanged"],
  ["Plasticity", "Being able to learn new or changed information", "Gate", "A unitary operation on one or two qubits"],
  ["Probe", "A question at a known point, with a known answer", "Entanglement", "A correlation between qubits that cannot be described qubit by qubit"],
  ["Stale fact", "A fact a later turn has replaced", "Variational circuit", "A circuit with adjustable angles trained by gradient descent"],
  ["Qubit / amplitude", "The quantum bit / a number whose square is an outcome probability", "Fisher information", "How sensitive a model's output is to its parameters"],
  ["MRR", "Mean reciprocal rank; 1 if the answer is ranked first", "Barren plateau", "A region where gradients are so small that training stalls"],
], [1500, 3319, 1700, 3119], { size: 16 }));

add(H1("Appendix B. Reproducing the results"));
add(P("The code is in the project's qcs folder. The scripts below were added for the final experiments and write to the runs and figures folders; the cached training features are rebuilt by the dataset builder before the first run. Measured numbers are collected in RESULTS_FINAL.md."));
["python -m pip install -r requirements.txt && python -m pip install -e . && python -m pytest",
 "python scripts/experiments/final_eval.py multi 7   # also evidence_eval.py, plasticity_fairness.py; models: quantum multi linear mlp; seeds 7 17 27",
 "python scripts/figures/make_final_figures.py && python scripts/report/finalize_results.py",
].forEach((l) => add(new Paragraph({
  spacing: { after: 10 }, shading: { fill: "F3F5F6", type: ShadingType.CLEAR, color: "auto" },
  children: [new TextRun({ text: l, font: "Consolas", size: 16, color: INK })],
})));

// ---------------------------------------------------------------- document
const numCfg = (ref) => ({
  reference: ref,
  levels: [{ level: 0, format: LevelFormat.DECIMAL, text: "%1.", alignment: AlignmentType.LEFT,
    style: { paragraph: { indent: { left: 440, hanging: 340 } } } }],
});
const doc = new Document({
  creator: "Team_14", title: "Dual-Manifold ContextSync", description: "Project report, Team_14",
  styles: {
    default: { document: { run: { font: BODY, size: 20 } } },
    paragraphStyles: [
      { id: "Heading1", name: "Heading 1", basedOn: "Normal", next: "Normal", quickFormat: true,
        run: { size: 28, bold: true, font: HEAD, color: TEAL }, paragraph: { spacing: { before: 230, after: 90 }, outlineLevel: 0 } },
      { id: "Heading2", name: "Heading 2", basedOn: "Normal", next: "Normal", quickFormat: true,
        run: { size: 22, bold: true, font: HEAD, color: INK }, paragraph: { spacing: { before: 130, after: 60 }, outlineLevel: 1 } },
    ],
  },
  numbering: { config: [
    { reference: "bul", levels: [{ level: 0, format: LevelFormat.BULLET, text: "•", alignment: AlignmentType.LEFT,
      style: { paragraph: { indent: { left: 400, hanging: 260 } } } }] },
    numCfg("n_hyp"), numCfg("n_refs"),
  ] },
  sections: [
    { properties: { page: { size: { width: 11906, height: 16838 }, margin: { top: 1134, bottom: 1134, left: 1134, right: 1134 } } },
      children: cover },
    { properties: { page: { size: { width: 11906, height: 16838 }, margin: { top: 850, bottom: 850, left: 1134, right: 1134 } } },
      footers: { default: new Footer({ children: [new Paragraph({
        tabStops: [{ type: TabStopType.RIGHT, position: W }],
        border: { top: { style: BorderStyle.SINGLE, size: 4, color: LINE, space: 4 } },
        children: [
          new TextRun({ text: "Team_14  ·  Dual-Manifold ContextSync  ·  Quantum Computing and Algorithm", font: BODY, size: 16, color: GRAY }),
          new TextRun({ text: "\tPage ", font: BODY, size: 16, color: GRAY }),
          new TextRun({ children: [PageNumber.CURRENT], font: BODY, size: 16, color: GRAY }),
        ] })] }) },
      children: body },
  ],
});
Packer.toBuffer(doc).then((buf) => { fs.writeFileSync(OUT, buf); console.log("wrote", OUT, buf.length); });
