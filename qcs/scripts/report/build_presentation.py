"""Final deck, in the design of the original (dark title, serif headings, colour dots, light cards).

    python scripts/report/build_presentation.py -> deliverables/Team_14_Dual_Manifold_ContextSync_Presentation.pptx
"""
from __future__ import annotations

import json
from pathlib import Path

from PIL import Image
from pptx import Presentation
from pptx.dml.color import RGBColor
from pptx.enum.shapes import MSO_SHAPE
from pptx.enum.text import MSO_ANCHOR, PP_ALIGN
from pptx.oxml.ns import qn
from pptx.util import Emu, Inches, Pt
from lxml import etree

ROOT = Path(__file__).resolve().parents[2]
FIG = ROOT / "figures" / "final"
S = json.load(open(ROOT / "results" / "final_summary.json"))
EV = json.load(open(ROOT / "results" / "evidence_summary.json"))


def rgb(h):
    return RGBColor.from_string(h)


TEAL, MAG, GREEN, INK, GRAY = "0B5C73", "A3226A", "2A7D57", "14191D", "5B6670"
TEAL_BG, MAG_BG, GREY_BG, LINE = "EFF5F7", "FAEEF4", "F6F8F9", "D5DDE1"
DARK = "0B1F26"
HEAD, BODY = "Cambria", "Calibri"

prs = Presentation()
prs.slide_width, prs.slide_height = Inches(13.333), Inches(7.5)
BLANK = prs.slide_layouts[6]
page = [0]


# --------------------------------------------------------------------------- helpers
def textbox(s, x, y, w, h, paras, size=16, color=INK, font=BODY, bold=False, align=PP_ALIGN.LEFT,
            anchor=MSO_ANCHOR.TOP, space=6, line=None):
    box = s.shapes.add_textbox(Inches(x), Inches(y), Inches(w), Inches(h))
    tf = box.text_frame
    tf.word_wrap = True
    tf.vertical_anchor = anchor
    tf.margin_left = tf.margin_right = Inches(0.04)
    tf.margin_top = tf.margin_bottom = Inches(0.02)
    paras = paras if isinstance(paras, list) else [paras]
    for i, p_ in enumerate(paras):
        p = tf.paragraphs[0] if i == 0 else tf.add_paragraph()
        spec = p_ if isinstance(p_, dict) else {"t": p_}
        p.alignment = align
        p.space_after = Pt(spec.get("space", space))
        if line:
            p.line_spacing = line
        r = p.add_run()
        r.text = spec["t"]
        f = r.font
        f.size = Pt(spec.get("size", size))
        f.bold = spec.get("bold", bold)
        f.italic = spec.get("italic", False)
        f.name = spec.get("font", font)
        f.color.rgb = rgb(spec.get("color", color))
        if spec.get("bullet"):
            pPr = p._p.get_or_add_pPr()
            pPr.set("marL", str(int(Inches(0.28))))
            pPr.set("indent", str(-int(Inches(0.28))))
            bu = etree.SubElement(pPr, qn("a:buChar"))
            bu.set("char", "•")
    return box


def bullets(items, **kw):
    return [dict(t=t, bullet=True, **kw) for t in items]


def card(s, x, y, w, h, fill=GREY_BG, line=LINE, lw=1.0, radius=0.035):
    sh = s.shapes.add_shape(MSO_SHAPE.ROUNDED_RECTANGLE, Inches(x), Inches(y), Inches(w), Inches(h))
    sh.adjustments[0] = radius
    sh.fill.solid()
    sh.fill.fore_color.rgb = rgb(fill)
    sh.line.color.rgb = rgb(line)
    sh.line.width = Pt(lw)
    sh.shadow.inherit = False
    return sh


def dots(s, x=0.55, y=0.42, c1=TEAL, c2=MAG):
    for i, c in enumerate((c1, c2)):
        d = s.shapes.add_shape(MSO_SHAPE.OVAL, Inches(x + i * 0.25), Inches(y), Inches(0.17), Inches(0.17))
        d.fill.solid()
        d.fill.fore_color.rgb = rgb(c)
        d.line.fill.background()
        d.shadow.inherit = False


def slide(title, sub=None, foot=None):
    s = prs.slides.add_slide(BLANK)
    page[0] += 1
    dots(s)
    textbox(s, 0.55, 0.68, 12.2, 0.8, title, 34, INK, HEAD, True)
    if sub:
        textbox(s, 0.55, 1.45, 12.2, 0.5, sub, 16, GRAY)
    textbox(s, 12.2, 6.95, 0.7, 0.3, str(page[0] + 1), 11, "8A949B", align=PP_ALIGN.RIGHT)
    if foot:
        textbox(s, 0.55, 6.95, 11.4, 0.3, foot, 11, "7B868E")
    return s


def fit(s, name, x, y, maxw, maxh, center=True):
    path = FIG / name
    iw, ih = Image.open(path).size
    sc = min(maxw / iw, maxh / ih)
    w, h = iw * sc, ih * sc
    px = x + (maxw - w) / 2 if center else x
    s.shapes.add_picture(str(path), Inches(px), Inches(y), Inches(w), Inches(h))
    return w, h


def cell_border(cell, color=LINE, w=6350):
    tcPr = cell._tc.get_or_add_tcPr()
    for tag in ("a:lnL", "a:lnR", "a:lnT", "a:lnB"):
        ln = etree.SubElement(tcPr, qn(tag), w=str(w), cap="flat", cmpd="sng", algn="ctr")
        sf = etree.SubElement(ln, qn("a:solidFill"))
        etree.SubElement(sf, qn("a:srgbClr"), val=color)


def table(s, rows, x, y, colw, size=13, rowh=0.44, bold_first_col=True, header=True):
    shp = s.shapes.add_table(len(rows), len(rows[0]), Inches(x), Inches(y), Inches(sum(colw)),
                             Inches(rowh * len(rows)))
    t = shp.table
    t.first_row = False
    t.horz_banding = False
    for j, cw in enumerate(colw):
        t.columns[j].width = Inches(cw)
    for i, row in enumerate(rows):
        t.rows[i].height = Inches(rowh)
        for j, v in enumerate(row):
            c = t.cell(i, j)
            c.text = ""
            tf = c.text_frame
            tf.word_wrap = True
            c.margin_left = c.margin_right = Inches(0.1)
            c.margin_top = c.margin_bottom = Inches(0.04)
            c.vertical_anchor = MSO_ANCHOR.MIDDLE
            r = tf.paragraphs[0].add_run()
            r.text = str(v)
            head = header and i == 0
            r.font.size = Pt(size - 1 if head else size)
            r.font.name = BODY
            r.font.bold = head or (bold_first_col and j == 0)
            r.font.color.rgb = rgb(GRAY if head else INK)
            c.fill.solid()
            c.fill.fore_color.rgb = rgb("EAF0F2" if head else "FFFFFF")
            cell_border(c)
    return t


def pm(a, k, d=3):
    return f"{S[a][k][0]:.{d}f} ± {S[a][k][1]:.{d}f}"


# --------------------------------------------------------------------------- 1 title
s = prs.slides.add_slide(BLANK)
bg = s.background.fill
bg.solid()
bg.fore_color.rgb = rgb(DARK)
dots(s, 1.15, 1.95, "CFE3E9", MAG)
textbox(s, 1.1, 2.55, 11, 1.2, "Dual-Manifold ContextSync", 54, "FFFFFF", HEAD, True)
textbox(s, 1.15, 3.95, 10.5, 1.2, "Keeping stability and plasticity together in long, multi-user chat threads "
        "with a classical memory and a 10-qubit circuit", 22, "D3E3E8")
textbox(s, 1.15, 5.95, 11, 0.9, ["Project A_14 · Neuro-Plasticity",
                                 "Python · PyTorch · PennyLane · Qiskit · FAISS · SQLite"], 14, "93A9B1")
page[0] = 0

# --------------------------------------------------------------------------- 2 contents
s = slide("What this deck covers", "From the problem to the results")
items = [("01", "The problem", "Why long shared chats are hard for a language model"),
         ("02", "Background", "What is already known and what is missing"),
         ("03", "Design", "The memory, the circuit and how they connect"),
         ("04", "Data and build", "The dataset and the ten build phases"),
         ("05", "What we changed", "Reading out every qubit instead of one"),
         ("06", "Results", "Remembering, learning, and both together"),
         ("07", "Limits and next steps", "What is still open")]
for i, (n, t, d) in enumerate(items):
    col, row = i % 2, i // 2
    x, y = 0.75 + col * 6.2, 2.25 + row * 1.12
    textbox(s, x, y, 0.8, 0.6, n, 28, MAG, HEAD, True)
    textbox(s, x + 0.85, y - 0.02, 5.1, 0.4, t, 19, INK, BODY, True)
    textbox(s, x + 0.85, y + 0.4, 5.1, 0.5, d, 14, GRAY)

# --------------------------------------------------------------------------- 3 problem
s = slide("The problem", "Why a long shared chat is hard for a language model")
card(s, 0.55, 2.1, 6.9, 4.5, TEAL_BG, TEAL)
textbox(s, 0.85, 2.3, 6.3, 0.5, "Two things a model has to do at once", 20, TEAL, HEAD, True)
textbox(s, 0.85, 2.95, 6.3, 3.6, [
    "Stability means keeping what was learned earlier.",
    "Plasticity means being able to learn new or changed information.",
    "In an ordinary network both live in the same weights, so they compete. If the weights are held "
    "still, the model cannot adapt. If they move freely, older knowledge is overwritten.",
    "This is the stability-plasticity dilemma."], 16.5, INK, space=10)
stats = [("600+", "turns in a single thread, more than a context window can hold"),
         ("5", "participants, who sometimes state facts that contradict each other"),
         ("13%", "of the time plain similarity search returns a fact that was already replaced")]
for i, (n, d) in enumerate(stats):
    y = 2.1 + i * 1.55
    card(s, 7.75, y, 5.0, 1.38, "FFFFFF", LINE)
    textbox(s, 7.95, y + 0.2, 1.6, 0.9, n, 36, MAG, HEAD, True, anchor=MSO_ANCHOR.MIDDLE)
    textbox(s, 9.55, y + 0.2, 3.1, 1.0, d, 13.5, GRAY, anchor=MSO_ANCHOR.MIDDLE)
textbox(s, 7.75, 6.7, 5.0, 0.3, "Measured on the project corpus with a cosine retriever.", 10.5, "7B868E")

# --------------------------------------------------------------------------- 4 failure modes
s = slide("Three ways it goes wrong", "They are often lumped together, but each needs its own fix")
modes = [("1", "Catastrophic forgetting", "Training on turn 500 overwrites what the model stored about turn 40. "
          "Once it is gone, no retrieval method can bring it back.", TEAL),
         ("2", "Loss of plasticity", "Weights grow and units saturate over time. The parameters are still there, "
          "but the model can no longer learn much from new data.", MAG),
         ("3", "Context overload", "The simple fix is to put more into the prompt. Attention spreads thin and "
          "answers get worse even when the right fact is in the window.", GREEN)]
for i, (n, t, d, c) in enumerate(modes):
    x = 0.55 + i * 4.2
    card(s, x, 2.1, 3.95, 3.5, "FFFFFF", c, 1.4)
    textbox(s, x + 0.3, 2.3, 0.8, 0.7, n, 36, c, HEAD, True)
    textbox(s, x + 0.3, 3.1, 3.4, 0.5, t, 19, INK, BODY, True)
    textbox(s, x + 0.3, 3.7, 3.4, 1.9, d, 14, GRAY)
card(s, 0.55, 5.85, 12.2, 0.85, GREY_BG, LINE)
textbox(s, 0.8, 5.9, 11.7, 0.75, "Our design addresses each one: the memory cannot be overwritten, the circuit is "
        "built to keep its ability to adapt, and the prompt is filled under a fixed token budget.",
        14.5, INK, anchor=MSO_ANCHOR.MIDDLE)

# --------------------------------------------------------------------------- 5 related work
s = slide("Related work", "What is established and what each line of work leaves open",
          "Classic references are cited from the standard literature. The 2026 PRX Quantum paper is the direct basis of this project.")
rows = [["Work", "Contribution", "What it leaves open"],
        ["McCloskey and Cohen (1989)", "Named catastrophic interference", "Describes the problem, no way to prevent it"],
        ["Kirkpatrick et al. (2017), EWC", "Penalty that anchors important weights", "Knowledge still sits in the weights; sensitive to settings"],
        ["Lopez-Paz and Ranzato (2017), GEM", "Replay with gradient constraints; BWT and FWT metrics", "Memory grows; says little about plasticity"],
        ["Dohare et al. (2024), Nature", "Showed deep networks gradually lose the ability to learn", "Fixes are add-ons such as resets or noise injection"],
        ["McClean (2018); Cerezo (2021)", "Barren plateaus and their link to the observable", "Points to local readouts and shallow circuits"],
        ["Mitarai (2018); Schuld (2019)", "Parameter-shift rule for exact gradients", "Needs involutory generators, which a general SU(4) gate lacks"],
        ["Lewis et al. (2020), RAG", "Retrieval as external memory", "The ranking is fixed and does not adapt over time"],
        ["Liu et al. (2023)", "Long contexts degrade even when the fact is present", "Confirms overload, gives no allocation policy"],
        ["Chen and Zhang (2026), PRX Quantum", "Quantum models keep plasticity on a compact manifold", "Says outright that this does not solve forgetting"]]
table(s, rows, 0.55, 2.0, [3.6, 4.6, 4.0], 12.5, 0.47)

# --------------------------------------------------------------------------- 6 gap
s = slide("The gap", "Quantum circuits keep the ability to learn, but not what they learned")
card(s, 0.55, 2.1, 12.2, 1.9, GREY_BG, LINE)
textbox(s, 0.85, 2.25, 11.6, 1.7, [
    "The reference paper reports that both quantum and classical models still forget catastrophically "
    "unless something else holds the old knowledge. It suggests a hybrid: a plastic quantum part together "
    "with a classical memory.",
    "That suggests splitting the work between the two."], 16.5, INK, space=10, anchor=MSO_ANCHOR.MIDDLE)
card(s, 0.55, 4.3, 5.95, 2.3, TEAL_BG, TEAL)
textbox(s, 0.85, 4.5, 5.4, 0.5, "Classical, append-only storage", 21, TEAL, HEAD, True)
textbox(s, 0.85, 5.2, 5.4, 1.3, "Holds what must never be lost. Nothing in it is trained, so training cannot erase it.", 16, INK)
card(s, 6.8, 4.3, 5.95, 2.3, MAG_BG, MAG)
textbox(s, 7.1, 4.5, 5.4, 0.5, "A compact quantum circuit", 21, MAG, HEAD, True)
textbox(s, 7.1, 5.2, 5.4, 1.3, "Holds what must keep learning. Its angles wrap around, so they cannot grow without limit.", 16, INK)

# --------------------------------------------------------------------------- 7 approach
s = slide("Our approach", "Two properties, carried by two different objects")
card(s, 0.55, 2.05, 5.95, 3.5, TEAL_BG, TEAL)
textbox(s, 0.85, 2.2, 5.4, 0.5, "ℳ  Stability: the memory", 21, TEAL, HEAD, True)
textbox(s, 0.85, 2.85, 5.4, 2.7, bullets([
    "Every message unit is stored word for word, add-only",
    "The database rejects edits and deletes through triggers",
    "Nothing here is trained, so a gradient step cannot erase it",
    "A lost fact can only appear as a ranking mistake, which can be corrected"]), 15.5, INK, space=8)
card(s, 6.8, 2.05, 5.95, 3.5, MAG_BG, MAG)
textbox(s, 7.1, 2.2, 5.4, 0.5, "U  Plasticity: the circuit", 21, MAG, HEAD, True)
textbox(s, 7.1, 2.85, 5.4, 2.7, bullets([
    "A 10-qubit variational circuit with 1,080 angles",
    "Angles wrap around a circle instead of growing",
    "Every gate is unitary, so the state keeps unit length",
    "The paper argues this keeps learning capacity from fading. We test it over 120 tasks"]), 15.5, INK, space=8)
card(s, 0.55, 5.75, 12.2, 1.0, GREY_BG, LINE)
textbox(s, 0.8, 5.78, 11.7, 0.95, "The two sides meet at one point. The circuit reads a feature vector for a (query, candidate) pair "
        "and returns one relevance score between 0 and 1. Because the interface is this narrow, a classical "
        "scorer with the same interface can replace the circuit, which makes the comparison fair.",
        13.5, INK, anchor=MSO_ANCHOR.MIDDLE)

# --------------------------------------------------------------------------- 8 architecture
s = slide("System architecture", "Four bands, one direction of data flow, one place where learning happens")
fit(s, "architecture.png", 0.55, 1.9, 12.2, 4.7)
textbox(s, 0.55, 6.6, 12.2, 0.35, "Teal: classical and add-only. Pink: the quantum part, the only component that is trained. "
        "Green and grey: delivery, with the language model frozen.", 12, GRAY, align=PP_ALIGN.CENTER)

# --------------------------------------------------------------------------- 9 circuit
s = slide("The quantum circuit")
fit(s, "circuit.png", 0.4, 1.4, 12.5, 5.5)
textbox(s, 0.55, 6.93, 11.5, 0.3, "The simulator multiplies each gate's 4×4 unitary directly. The CNOT form is what a compiler would give for hardware.", 10.5, "7B868E")

# --------------------------------------------------------------------------- 10 dataset
s = slide("The dataset", "Generated in two steps, so the correct answer is known exactly")
card(s, 0.55, 2.05, 6.0, 4.65, GREY_BG, LINE)
textbox(s, 0.8, 2.2, 5.5, 4.4, [
    {"t": "1. Plan", "bold": True, "color": TEAL, "size": 18},
    "Each thread is planned first: who states which fact, which later turn changes it, and which turn "
    "asks about it and how far back.",
    {"t": "2. Write the text", "bold": True, "color": TEAL, "size": 18},
    "Templates turn the plan into chat messages. The wording can never add or remove a fact, "
    "so the ground truth stays exact.",
    {"t": "Three kinds of thread rotate: one owner, several owners, and adversarial. Only adversarial threads contain updates (about 35% of their facts are replaced later).", "color": GRAY}], 16.5, INK, space=12)
rows = [["Turn type", "What it is", "Why it is there"],
        ["fact", "a participant states an (entity, attribute, value)", "the target to retrieve"],
        ["update", "a later turn changes that fact", "creates old-versus-new traps"],
        ["probe", "a question answerable from stated facts", "the measurement point"],
        ["distractor", "mentions a real entity with no usable value", "defeats keyword matching"],
        ["filler", "ordinary conversation", "dilutes the thread"]]
table(s, rows, 6.8, 2.05, [1.25, 2.6, 2.1], 11.5, 0.5)
facts = [("18,000", "turns"), ("1,800", "probes"), ("1,920", "facts"), ("1.000", "oracle recall"), ("0.042", "random recall")]
for i, (n, d) in enumerate(facts):
    x = 6.8 + i * 1.19
    textbox(s, x, 5.35, 1.2, 0.55, n, 19, MAG, HEAD, True, align=PP_ALIGN.CENTER)
    textbox(s, x, 5.9, 1.2, 0.5, d, 11, GRAY, align=PP_ALIGN.CENTER)

# --------------------------------------------------------------------------- 11 phases
s = slide("Ten build phases", "Each phase has an exit test that has to pass before the next one starts",
          "Phases 0 to 8 are finished. Phase 9 is the remaining work.")
ph = [("0", "Foundations", "one config file, fixed seeds"), ("1", "Dataset", "ground truth by construction"),
      ("2", "Encoding", "text to a unit vector in 1024 dimensions"), ("3", "Memory", "append-only, enforced by the database"),
      ("4", "Quantum core", "verified circuit, depth chosen from evidence"), ("5", "Baselines", "each model at its own best learning rate"),
      ("6", "Continual training", "120 tasks, full logging"), ("7", "Assembly", "budgeted prompt, frozen generator"),
      ("8", "Evaluation", "seeds, ablations, noise, sealed test split"), ("9", "Hardening", "long run, hardware, real LLM (not started)")]
for i, (n, t, d) in enumerate(ph):
    col, row = i % 5, i // 5
    x, y = 0.55 + col * 2.46, 2.2 + row * 2.2
    c = GRAY if n == "9" else (MAG if n in ("4", "6") else TEAL)
    card(s, x, y, 2.3, 1.95, "FFFFFF", c, 1.2)
    textbox(s, x + 0.15, y + 0.1, 0.8, 0.6, n, 28, c, HEAD, True)
    textbox(s, x + 0.15, y + 0.75, 2.05, 0.4, t, 14.5, INK, BODY, True)
    textbox(s, x + 0.15, y + 1.15, 2.05, 0.8, d, 11.5, GRAY)

# --------------------------------------------------------------------------- 12 phase results
s = slide("What each phase measured", "The main check recorded for every finished phase")
rows = [["Phase", "What happens", "Result"],
        ["0 Foundations", "All settings in one file, seeds fixed with a stable hash", "two runs with the same seed give the same output"],
        ["1 Dataset", "Plan each thread, then fill in text from templates", "oracle recall 1.000, random 0.042"],
        ["2 Encoding", "Frozen encoder, whitening, projection trained once", "held-out recall@20 rose from 0.879 to 0.996"],
        ["3 Memory", "Verbatim store, summaries added beside sources, supersession graph", "prefilter recall 1.000, 3 mutation tests rejected"],
        ["4 Quantum core", "General SU(4) gate, amplitude encoding, brickwall layers", "fidelity 1.000 against PennyLane, gradients agree to 8.6e-9"],
        ["5 Baselines", "Linear twin, MLP, cosine, recency, random, oracle", "tuning the learning rate moved linear from 0.063 to 0.965 MRR"],
        ["6 Continual training", "120 tasks in order, replay, quantum Fisher anchor", "backward transfer positive for every model"],
        ["7 Assembly", "Exact token counts, greedy selection, stale-fact filter", "budget never exceeded, no unlabelled stale fact served"],
        ["8 Evaluation", "3 seeds, ablations, noise, sealed split, bootstrap", "ranking unchanged under depolarising noise up to p = 1e-3"]]
table(s, rows, 0.55, 2.0, [2.3, 5.2, 4.7], 12, 0.47)

# --------------------------------------------------------------------------- 13 what we changed
s = slide("What we changed", "Read out every qubit instead of one",
          "Seed 7 first run, then 3 seeds. The test split is the sealed one.")
card(s, 0.55, 2.05, 5.7, 4.65, GREY_BG, LINE)
textbox(s, 0.8, 2.2, 5.2, 4.4, [
    {"t": "The problem we found", "bold": True, "color": TEAL, "size": 17},
    "The original circuit measured a single qubit. That gives a quadratic function of the features, which "
    "cannot reproduce the cosine-similarity signal in the data. It trailed the linear model on ranking.",
    {"t": "The change", "bold": True, "color": TEAL, "size": 17},
    "Measure all 10 qubits and learn one weight for each: z = Σ w_q⟨Z_q⟩, then r = σ(βz + b). "
    "The weights start at the old setting, so step 0 is identical to the original model.",
    {"t": "Tried and dropped: a fixed similarity prior (worse), deeper circuits, larger β, "
          "more epochs (capacity became erratic).", "color": GRAY, "size": 14}], 15.5, INK, space=10)
rows = [["Sealed test split", "1-qubit", "10-qubit", "Linear"],
        ["MRR", pm("quantum", "test_mrr"), pm("multi", "test_mrr"), pm("linear", "test_mrr")],
        ["Rank-1", pm("quantum", "test_rank1"), pm("multi", "test_rank1"), pm("linear", "test_rank1")],
        ["Stale fact ranked first", pm("quantum", "test_stale"), pm("multi", "test_stale"), pm("linear", "test_stale")]]
table(s, rows, 6.5, 2.05, [2.0, 1.4, 1.4, 1.45], 12, 0.62)
card(s, 6.5, 4.75, 6.25, 1.95, MAG_BG, MAG)
textbox(s, 6.75, 4.85, 5.8, 1.8, [
    "The change adds 0.114 MRR on the test split (95% CI 0.098 to 0.131).",
    "Against the linear model the difference is 0.001 (CI -0.007 to 0.009). That is a tie."], 14.5, INK,
    space=8, anchor=MSO_ANCHOR.MIDDLE)

# --------------------------------------------------------------------------- 14 how we test
s = slide("How we test stability and plasticity", "Three questions, each with its own measurement")
q = [("Does it remember the old?", "After every 10th task, re-test the model on every task it has seen. "
      "Also ask for facts stated up to 500 turns earlier.", "Stability", TEAL),
     ("Can it learn something new?", "Stop after a given number of tasks, copy the model, and train it for "
      "40 steps on a reversed rule: the replaced fact is now the correct answer.", "Plasticity", MAG),
     ("Both together", "Take the harmonic mean of the two scores, so a model has to do well on both. "
      "Track it as training goes on.", "Combined", GREEN)]
for i, (t, d, tag, c) in enumerate(q):
    x = 0.55 + i * 4.2
    card(s, x, 2.1, 3.95, 3.7, "FFFFFF", c, 1.4)
    textbox(s, x + 0.3, 2.3, 3.4, 0.4, tag.upper(), 12, c, BODY, True)
    textbox(s, x + 0.3, 2.75, 3.4, 0.9, t, 20, INK, HEAD, True)
    textbox(s, x + 0.3, 3.8, 3.4, 2.0, d, 15.5, GRAY)
card(s, 0.55, 6.05, 12.2, 0.6, GREY_BG, LINE)
textbox(s, 0.8, 6.07, 11.7, 0.55, "MRR (mean reciprocal rank) is 1 when the correct fact is ranked first and falls as it moves down the list.",
        12.5, INK, anchor=MSO_ANCHOR.MIDDLE)

# --------------------------------------------------------------------------- 15 remembers
s = slide("Result 1: it remembers the old", "Every past task re-tested as training continued (seed 7 shown, mean of 3 seeds below)")
fit(s, "forgetting_matrix.png", 0.55, 1.95, 12.2, 4.6)
textbox(s, 0.55, 6.5, 12.2, 0.45, "Yellow means remembered. No model forgets catastrophically. After 120 tasks the 10-qubit "
        "circuit averages 0.98 over all past tasks, against 0.99 for linear and 1.00 for the MLP.", 12.5, GRAY, align=PP_ALIGN.CENTER)

# --------------------------------------------------------------------------- 16 far back
s = slide("Result 2: it recalls facts from far back", "Sealed test split, grouped by how long ago the needed fact was stated")
fit(s, "memory_distance.png", 0.55, 1.95, 12.2, 4.55)
textbox(s, 0.55, 6.45, 12.2, 0.5, "A model that sees only the last 100 turns scores 0 beyond 200 turns. With the memory, facts older than "
        "400 turns are still found: 0.97 for the 10-qubit circuit, 0.94 for linear, 0.99 for the MLP.", 12.5, GRAY, align=PP_ALIGN.CENTER)

# --------------------------------------------------------------------------- 17 learns new
s = slide("Result 3: it learns new things", "Score on tasks it has not trained on, and on probes that have an outdated fact among the candidates")
fit(s, "learning_new.png", 0.55, 1.95, 12.2, 4.4)
textbox(s, 0.55, 6.35, 12.2, 0.6, "Every model starts untrained and picks up the task within about 15 tasks. On probes with an outdated fact among the candidates (the situation an update creates), the "
        "10-qubit circuit scores 0.96, compared with 0.89 for plain similarity search. After the first tasks, a trained model already "
        "does well on unseen tasks, so the reversed-rule test on the next slide is the sharper check of plasticity.",
        12, GRAY, align=PP_ALIGN.CENTER)

# --------------------------------------------------------------------------- 18 reversed rule
s = slide("Result 4: learning a reversed rule", "After many tasks, can the model still learn something that contradicts what it knows?")
fit(s, "plasticity.png", 0.55, 1.95, 12.2, 4.4)
textbox(s, 0.55, 6.4, 12.2, 0.55, "At each model's training learning rate, linear and MLP lose most of this ability over 120 tasks "
        "(about 0.8 down to about 0.2). Both quantum models keep it at about 0.84. The next slides check how much of that is down to "
        "the learning rate.", 12.5, GRAY, align=PP_ALIGN.CENTER)

# --------------------------------------------------------------------------- 19 dilemma
s = slide("Both together", "Each model's path through the stability and plasticity plane")
fit(s, "dilemma.png", 0.55, 1.95, 12.2, 4.5)
textbox(s, 0.55, 6.45, 12.2, 0.5, f"After 120 tasks the combined score is {S['multi']['combined_end'][0]:.2f} for the 10-qubit circuit, "
        f"{S['linear']['combined_end'][0]:.2f} for linear and {S['mlp']['combined_end'][0]:.2f} for the MLP. The classical models end up very "
        "stable but much less able to adapt. The circuit stays closer to the top right.", 12.5, GRAY, align=PP_ALIGN.CENTER)

# --------------------------------------------------------------------------- 20 fairness
s = slide("Is it just the learning rate?", "Every model re-learns the reversed rule at the best of six learning rates")
fit(s, "fairness.png", 0.55, 1.95, 6.6, 4.4)
card(s, 7.4, 2.1, 5.35, 4.3, GREY_BG, LINE)
textbox(s, 7.65, 2.25, 4.9, 4.1, [
    {"t": "Partly.", "bold": True, "size": 20, "color": MAG, "font": HEAD},
    "With the best step size and 80 steps, the classical models recover a good part of the loss: "
    "linear reaches 0.72 and the MLP 0.71, although the MLP is erratic across seeds (0.95, 0.29, 0.89).",
    "The circuit is still faster and steadier: 0.86 after 40 steps against 0.49 and 0.55, and 0.97 after 80 steps.",
    "The classical models can still learn the new rule. The circuit is faster and more consistent at it."],
    15.5, INK, space=11)

# --------------------------------------------------------------------------- 21 scorecard
s = slide("Scorecard", "Mean ± standard deviation over 3 seeds, after 120 tasks")
rows = [["Model", "Test MRR", "Stability", "Plasticity", "Combined", "Parameters"]]
for a, n, p in [("multi", "Quantum, 10-qubit readout", "1,080"), ("quantum", "Quantum, 1-qubit readout", "1,080"),
                ("linear", "Linear", "1,025"), ("mlp", "MLP", "65,665")]:
    rows.append([n, pm(a, "test_mrr"), pm(a, "retention_end"), pm(a, "reversal_end"), pm(a, "combined_end"), p])
table(s, rows, 0.55, 2.0, [3.3, 1.9, 1.9, 1.9, 1.9, 1.3], 13.5, 0.62)
card(s, 0.55, 5.25, 12.2, 1.45, GREY_BG, LINE)
textbox(s, 0.8, 5.3, 11.7, 1.35, [
    "Stability is the score on the first 30 tasks after all 120. Plasticity is the score on the reversed rule after 40 steps, "
    "at each model's own learning rate. With the best learning rate and 80 steps it is 0.97 for the circuit, 0.72 for linear and 0.71 for the MLP.",
    "The MLP has the best raw accuracy and stability, with about 60 times as many parameters."], 13, INK, space=6,
    anchor=MSO_ANCHOR.MIDDLE)

# --------------------------------------------------------------------------- 22 claims
s = slide("What the evidence supports", "Each claim and the number behind it")
rows = [["Claim", "Status", "Evidence"],
        ["Knowledge in the memory cannot be erased by training", "Supported", "append-only store, 3 mutation tests rejected"],
        ["Reading out all qubits closes the ranking gap", "Supported", "+0.114 MRR on the test split, CI excludes 0"],
        ["The circuit matches a linear model of the same size", "Supported", "difference 0.001, CI -0.007 to 0.009"],
        ["The circuit ranks better than classical models", "Not supported", "the MLP scores higher (0.995 vs 0.974)"],
        ["The circuit adapts to new rules faster and more steadily", "Supported, with a caveat", "0.97 vs 0.72 and 0.71 at the best learning rate"],
        ["Stale facts are kept below current ones", "Mostly", "0.043 error for the circuit, 0.000 for classical, 0.130 for cosine"],
        ["Fisher trace shows capacity is preserved", "Not supported", "noisy, and the MLP's did not fall"],
        ["Plasticity holds over 1,000+ tasks", "Not tested", "only 120 tasks were run"]]
table(s, rows, 0.55, 2.0, [5.1, 2.4, 4.7], 12.5, 0.52)

# --------------------------------------------------------------------------- 23 limits
s = slide("Limits and next steps")
card(s, 0.55, 1.6, 5.95, 4.9, GREY_BG, LINE)
textbox(s, 0.85, 1.8, 5.4, 0.5, "Limits", 22, TEAL, HEAD, True)
textbox(s, 0.85, 2.45, 5.4, 4.2, bullets([
    "120 tasks is short. The long-run plasticity claim is untested.",
    "The test split has 6 threads, so the significance tests are coarse.",
    "The circuit is simulated exactly. There is no hardware or finite-shot run yet.",
    "The reversed rule is one defined test of plasticity, not a real shift in a conversation.",
    "The circuit still ranks a stale fact first more often than the classical models."]), 15.5, INK, space=11)
card(s, 6.8, 1.6, 5.95, 4.9, MAG_BG, MAG)
textbox(s, 7.1, 1.8, 5.4, 0.5, "Next steps", 22, MAG, HEAD, True)
textbox(s, 7.1, 2.45, 5.4, 4.2, [
    "1.  Run 1,000 to 3,000 tasks on 3 seeds and track the reversed-rule score at checkpoints.",
    "2.  Generate more held-out threads for tighter statistics.",
    "3.  Simulate finite shots and noise for the 10-qubit readout.",
    "4.  Replace the extractive reader with a real frozen language model.",
    "5.  Add a one-command reproduction script and a scaling study over 8, 10 and 12 qubits."], 15.5, INK, space=11)

# --------------------------------------------------------------------------- 24 base paper
s = prs.slides.add_slide(BLANK)
page[0] += 1
s.background.fill.solid()
s.background.fill.fore_color.rgb = rgb(DARK)
dots(s, 1.15, 1.2, "CFE3E9", MAG)
textbox(s, 1.1, 1.75, 11, 0.9, "Base paper", 40, "FFFFFF", HEAD, True)
textbox(s, 1.1, 2.9, 11, 0.9, "Intrinsic Preservation of Plasticity in Continual Quantum Learning", 24, "D3E3E8", HEAD, True)
textbox(s, 1.15, 3.6, 11, 0.5, "Yu-Qin Chen and Shi-Xin Zhang · PRX Quantum 7, 033003 (2026)", 16, "93A9B1")
textbox(s, 1.15, 4.4, 10.5, 2.2, [
    "It shows that quantum models keep plasticity because unitary evolution confines the parameters to a compact manifold.",
    "It also states that this does not solve forgetting. Both quantum and classical models still need a separate memory.",
    "This project builds the hybrid it points to: a plastic quantum circuit with a classical memory."], 16, "D3E3E8", space=10)

out = ROOT / "deliverables" / "Team_14_Dual_Manifold_ContextSync_Presentation.pptx"
prs.save(out)
print(out, len(prs.slides._sldIdLst), "slides")
