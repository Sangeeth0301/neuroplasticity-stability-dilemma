"""Publication-quality architecture diagrams for the project report.

Four figures, all drawn with matplotlib primitives so they stay editable and
version-controlled rather than being opaque bitmaps:

    figs/DIAG_A_architecture.png   the master dual-manifold architecture
    figs/DIAG_B_pipeline.png       tensor shapes at every stage, S0 → S15
    figs/DIAG_C_circuit.png        the 10-qubit variational circuit in detail
    figs/DIAG_D_stack.png          the technology stack, layer by layer

Run:  python scripts/figures/make_diagrams.py
"""

from __future__ import annotations

from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
from matplotlib.patches import FancyArrowPatch, FancyBboxPatch, Rectangle  # noqa: E402

OUT = Path("figures") / "architecture"

# ---- palette: roles, not decoration ----------------------------------------
INK = "#12171B"
MUTE = "#5B6670"
LINE = "#C6CFD5"
PAPER = "#FFFFFF"
STAB = "#0F5F79"        # classical / stability
STAB_BG = "#E8F1F4"
PLAS = "#A6246A"        # quantum / plasticity
PLAS_BG = "#F8E9F1"
NEUT = "#4A5A63"        # interface
NEUT_BG = "#EEF1F3"
WARN = "#B0761A"
OK = "#2E7D5B"
FROZEN = "#3A4A54"
FROZEN_BG = "#E4E8EA"

FONT = "DejaVu Sans"
plt.rcParams.update({"font.family": FONT, "figure.facecolor": PAPER,
                     "savefig.facecolor": PAPER})


def box(ax, x, y, w, h, title, lines=None, fc=NEUT_BG, ec=NEUT, lw=1.2,
        title_size=9.5, body_size=7.6, title_color=None, radius=0.6, bold=True,
        align="left"):
    ax.add_patch(FancyBboxPatch((x, y), w, h, boxstyle=f"round,pad=0,rounding_size={radius}",
                                fc=fc, ec=ec, lw=lw, zorder=2))
    tx = x + 0.7 if align == "left" else x + w / 2
    ha = "left" if align == "left" else "center"
    ax.text(tx, y + h - 1.0, title, fontsize=title_size, color=title_color or ec,
            fontweight="bold" if bold else "normal", ha=ha, va="top", zorder=3)
    if lines:
        for i, ln in enumerate(lines):
            ax.text(tx, y + h - 2.35 - i * 1.30, ln, fontsize=body_size, color=INK,
                    ha=ha, va="top", zorder=3)


def arrow(ax, p0, p1, color=NEUT, lw=1.6, style="-|>", rad=0.0, ls="-", z=4):
    ax.add_patch(FancyArrowPatch(p0, p1, arrowstyle=style, mutation_scale=13,
                                 color=color, lw=lw, linestyle=ls, zorder=z,
                                 connectionstyle=f"arc3,rad={rad}",
                                 shrinkA=2, shrinkB=2))


def elbow(ax, pts, color=NEUT, lw=1.6, z=4):
    """Right-angled polyline with one arrowhead — keeps long routes from crossing boxes."""
    for a, b in zip(pts[:-2], pts[1:-1]):
        ax.plot([a[0], b[0]], [a[1], b[1]], color=color, lw=lw, zorder=z,
                solid_capstyle="round")
    arrow(ax, pts[-2], pts[-1], color, lw=lw, z=z)


def label(ax, x, y, text, size=7.2, color=MUTE, ha="center", style="normal", weight="normal"):
    ax.text(x, y, text, fontsize=size, color=color, ha=ha, va="center",
            style=style, fontweight=weight, zorder=5)


def band(ax, x, y, w, h, text, fc, ec, title_side="left"):
    ax.add_patch(Rectangle((x, y), w, h, fc=fc, ec=ec, lw=1.4, zorder=1, alpha=1.0))
    tx = x + 0.8 if title_side == "left" else x + w - 0.8
    ax.text(tx, y + h - 0.9, text, fontsize=10.5, color=ec, fontweight="bold",
            ha=title_side, va="top", zorder=3)


def new_ax(w, h, xlim, ylim):
    fig, ax = plt.subplots(figsize=(w, h))
    ax.set_xlim(*xlim)
    ax.set_ylim(*ylim)
    ax.axis("off")
    return fig, ax


# ============================================================================
# DIAGRAM A — master architecture
# ============================================================================
def diagram_a() -> Path:
    fig, ax = new_ax(17.5, 10.6, (0, 175), (0, 106))

    ax.text(3, 102, "Dual-Manifold ContextSync", fontsize=20, fontweight="bold", color=INK)
    ax.text(3, 97.6, "Stability and plasticity on two different mathematical objects, "
                     "joined by one scalar-wide interface",
            fontsize=10.5, color=MUTE)
    ax.plot([3, 172], [95.2, 95.2], color=LINE, lw=1.2)

    # ---------------- stability manifold -----------------------------------
    band(ax, 3, 44, 62, 49, "ℳ  ·  STABILITY MANIFOLD   (classical, append-only)",
         STAB_BG, STAB)

    box(ax, 5.5, 79, 27, 10.5, "L0 · INGEST & SEGMENT",
        ["thread event → semantic units (SU)", "20–60 tokens · speaker · turn · ts",
         "lossless: concat(SU) == turn"], STAB_BG, STAB)
    box(ax, 35.5, 79, 27, 10.5, "L1 · FROZEN ENCODER",
        ["MiniLM-L6-v2 · requires_grad = False", "cached by sha256(text)",
         "384-d embedding per SU"], STAB_BG, STAB)

    box(ax, 5.5, 46.0, 57, 30, "L2 · MEMORY MANIFOLD  (SQLite + FAISS)", None, "#FFFFFF", STAB)
    sub = [
        (7.5, 60.6, "EPISODIC TIER", ["every SU, verbatim, write-once",
                                      "UPDATE / DELETE rejected by DB trigger"]),
        (35.0, 60.6, "SEMANTIC TIER", ["summary nodes every K=50 turns",
                                       "ADDED beside sources, never replacing"]),
        (7.5, 47.4, "ENTITY / SUPERSESSION DAG", ["supersedes · contradicts · shares_entity",
                                                  "contradiction is stored, not resolved"]),
        (35.0, 47.4, "PER-USER SALIENCE", ["EMA over each participant's own turns",
                                           "64-d vector per (thread, user)"]),
    ]
    for x, y, t, ls in sub:
        box(ax, x, y, 25.5, 11.4, t, ls, STAB_BG, STAB, title_size=8.2, body_size=7.0,
            radius=0.4)

    label(ax, 5.5, 45.0, "growth is monotone:  d|ℳ|/dt ≥ 0    ⇒    forgetting reduces to a "
                         "retrieval miss, never a gradient overwrite",
          size=8.0, color=STAB, style="italic", ha="left")

    # ---------------- interface --------------------------------------------
    band(ax, 68, 44, 34, 49, "INTERFACE  ·  the only coupling", NEUT_BG, NEUT)
    box(ax, 70, 79, 30, 10.5, "L2b · CANDIDATE PREFILTER",
        ["FAISS cosine, causality-masked", "top-M (M = 64 train / 256 serve)",
         "measured recall of gold fact: 1.000"], "#FFFFFF", NEUT)
    box(ax, 70, 61.5, 30, 15.5, "S5 · FEATURE ASSEMBLY  →  ℝ¹⁰²⁴",
        ["[ p_q ‖ p_c ‖ p_q ⊙ p_c ‖ ρ ]", "256 + 256 + 256 + 256 = 1024 = 2¹⁰",
         "per-block norm targets 0.35/0.35/0.20/0.10",
         "ρ: recency · graph · speaker · salience",
         "every ρ feature bounded to [−1,1]"], "#FFFFFF", NEUT, body_size=7.2)
    box(ax, 70, 46.5, 30, 13, "S6–S7 · TO HILBERT SPACE",
        ["x ← x / ‖x‖₂        (unit sphere S¹⁰²³)", "|ψ₀⟩ = Σⱼ xⱼ |j⟩   amplitude encoding",
         "1024 reals → 1024 amplitudes, lossless", "participation ratio 189 / 1024"],
        "#FFFFFF", NEUT, body_size=7.2)

    # ---------------- plasticity manifold -----------------------------------
    band(ax, 105, 44, 67, 49, "U  ·  PLASTICITY MANIFOLD   (quantum, compact, unitary)",
         PLAS_BG, PLAS)

    box(ax, 107, 70, 30.5, 19.5, "L3 · QUANTUM RELEVANCE ENGINE",
        ["N = 10 qubits · brickwall SU(4)", "depth L = 8  ⇒  |θ| = 1,080",
         "V(θ)=exp(−i/2 Σ θₖ σ⊗σ), 15 params/gate", "|ψ_L⟩ = U(θ)|ψ₀⟩   (unitary: ‖ψ‖ ≡ 1)",
         "readout ⟨Z₉⟩ → r = σ(β⟨Z⟩+b)", "θ lives on a torus — angles wrap, never inflate"],
        "#FFFFFF", PLAS, body_size=7.2)

    box(ax, 140, 70, 30.5, 19.5, "L4 · STABILITY CONTROLLER",
        ["reservoir replay B (uniform over history)", "QFIM anchor  Ω = ½(θ−θ*)ᵀF_Q(θ−θ*)",
         "trust region  ‖Δθ‖∞ ≤ δ", "PSI drift detector → re-checkpoint θ*",
         "angle wrapping into (−π, π]", "the quantum core buys plasticity ONLY"],
        "#FFFFFF", PLAS, body_size=7.2)

    box(ax, 107, 46.5, 63.5, 21, "WHY THIS MANIFOLD", None, "#FFFFFF", PLAS)
    for i, ln in enumerate([
        "unitarity  ⇒  ‖U(θ)‖₂ = 1 for every θ, so no layer can amplify a signal without bound",
        "compact periodic parameters  ⇒  weights cannot inflate; the optimisation landscape stays navigable",
        "Haar-averaged Fisher information is bounded above and below  ⇒  learning capacity cannot decay to zero",
        "measured here: Tr(F) rises to 2.5× over 120 tasks while the classical arms fall to 0.30× and 0.06×",
    ]):
        ax.text(108.6, 63.5 - i * 2.6, "•  " + ln, fontsize=7.6, color=INK, va="top")

    # ---------------- bottom band ------------------------------------------
    band(ax, 3, 8, 169, 31, "SELECTION, GENERATION AND FEEDBACK", "#F6F8F9", NEUT,
         title_side="right")

    box(ax, 6, 20, 36, 15, "L5 · CONTEXT ASSEMBLER",
        ["greedy submodular knapsack, (1−1/e)-optimal", "max Σ r_c·v_c − λ·redundancy(S)",
         "s.t. Σ tokens(c) ≤ B   (hard ceiling)", "reserved sub-budgets: system / pinned /",
         "recent / selected", "supersession filter: stale ⇒ dropped or",
         "labelled OUTDATED beside CURRENT"], "#FFFFFF", NEUT, body_size=7.2)

    box(ax, 48, 20, 33, 15, "FROZEN LANGUAGE MODEL",
        ["p(y | C) — inference only", "NO gradient path exists to these weights",
         "catastrophic forgetting of world", "knowledge is ABSENT, not mitigated",
         "(stand-in: ExtractiveReader, which", " answers only from the assembled context)"],
        FROZEN_BG, FROZEN, body_size=7.2)

    box(ax, 87, 20, 37, 15, "L6 · FEEDBACK & EVALUATION",
        ["probe ground truth → labels ℓ_c ∈ {0,1}", "ℒ = BCE(r, ℓ) + λ_Ω·Ω(θ, θ*)",
         "metrics: rank-1 · MRR · AUC · BWT / FWT", "plasticity index · Tr(F) · stale-above",
         "one JSONL row per task, schema-guarded"], "#FFFFFF", OK, body_size=7.2)

    box(ax, 130, 20, 39, 15, "GRADIENT SCOPE  (what actually learns)",
        ["θ (1,080 circuit angles)          ✔ learns", "β, b (readout scale, bias)        ✔ learns",
         "encoder · whitening · projection  ✘ frozen", "memory ℳ · entity DAG             ✘ append-only",
         "language model                    ✘ frozen"], "#FFFFFF", WARN, body_size=7.2)

    label(ax, 87, 11.5, "Stability is architectural (append-only storage + a frozen generator).   "
                       "Plasticity is geometric (a compact unitary manifold).   "
                       "The two never share a coordinate system.",
          size=9.4, color=INK, weight="bold")

    # ---------------- flow arrows ------------------------------------------
    arrow(ax, (32.5, 84.2), (35.5, 84.2), STAB)
    arrow(ax, (49, 79), (49, 75.5), STAB)
    arrow(ax, (62.5, 84.2), (70, 84.2), NEUT)
    arrow(ax, (85, 79), (85, 77), NEUT)
    arrow(ax, (85, 61.5), (85, 59.5), NEUT)
    # interface → circuit: an elbow up the gutter, rather than a diagonal across two panels
    elbow(ax, [(100, 53), (103.5, 53), (103.5, 79.7), (107, 79.7)], PLAS)
    label(ax, 103.6, 65.0, "|ψ₀⟩", size=8.0, color=PLAS)
    label(ax, 103.6, 62.4, "1024 amplitudes", size=6.5, color=PLAS)
    arrow(ax, (137.5, 79.7), (140, 79.7), PLAS)

    # scores → assembler: routed through the gap between bands, never over a box
    elbow(ax, [(138, 44), (138, 42.6), (30, 42.6), (30, 35)], PLAS)
    label(ax, 92, 41.4, "r ∈ [0,1]^M   —   one scalar per candidate", size=7.4, color=PLAS)

    # candidate set → assembler, parallel to the score route
    elbow(ax, [(9, 46), (9, 41.0), (14, 41.0), (14, 35)], STAB)
    label(ax, 15.5, 41.0, "candidate set", size=7.0, color=STAB, ha="left")

    arrow(ax, (42, 27.5), (48, 27.5), NEUT)
    arrow(ax, (81, 27.5), (87, 27.5), FROZEN)
    elbow(ax, [(105, 19.6), (127, 19.6), (127, 22)], OK)
    label(ax, 116, 17.6, "θ ← θ − η∇ℒ", size=7.4, color=OK)

    fig.tight_layout()
    OUT.mkdir(exist_ok=True)
    p = OUT / "DIAG_A_architecture.png"
    fig.savefig(p, dpi=300, bbox_inches="tight", facecolor=PAPER)
    fig.savefig(p.with_suffix(".svg"), bbox_inches="tight", facecolor=PAPER)
    plt.close(fig)
    return p


# ============================================================================
# DIAGRAM B — the data pipeline, shape by shape
# ============================================================================
def diagram_b() -> Path:
    fig, ax = new_ax(15.5, 13.6, (0, 155), (0, 148))
    ax.text(3, 144, "How the dataset is manipulated, stage by stage",
            fontsize=17, fontweight="bold", color=INK)
    ax.text(3, 140, "Every transformation from a raw chat turn to a generated answer, "
                    "with the exact object at each step",
            fontsize=10, color=MUTE)
    ax.plot([3, 152], [137.5, 137.5], color=LINE, lw=1.2)

    stages = [
        ("S0", "Raw thread event", "{thread_id, user_id, ts, text, kind}", "JSONL row", STAB),
        ("S1", "Segmentation", "turn → 1–3 sentences per unit", "list[SemanticUnit]", STAB),
        ("S2", "Frozen encoding", "e = MiniLM(text), no_grad", "ℝ³⁸⁴", STAB),
        ("S3", "PCA whitening", "ẽ = W(e − μ), shrinkage ε = 0.1", "ℝ²⁵⁶", STAB),
        ("S4", "Learned projection", "p = P ẽ, InfoNCE-trained then FROZEN", "ℝ²⁵⁶", STAB),
        ("S5", "Feature assembly", "[p_q ‖ p_c ‖ p_q⊙p_c ‖ ρ], block-balanced", "ℝ¹⁰²⁴", NEUT),
        ("S6", "L2 normalisation", "x = x/‖x‖₂  → the unit sphere S¹⁰²³", "‖x‖₂ = 1", NEUT),
        ("S7", "Amplitude encoding", "|ψ₀⟩ = Σⱼ xⱼ|j⟩ on 10 qubits", "ℂ¹⁰²⁴ statevector", PLAS),
        ("S8", "Variational circuit", "8 brickwall layers × 9 SU(4) gates", "|ψ_L⟩, ‖ψ‖ ≡ 1", PLAS),
        ("S9", "Measurement", "⟨Z₉⟩ ∈ [−1,1] → r = σ(β⟨Z⟩+b)", "one scalar", PLAS),
        ("S10", "Candidate scores", "r for every candidate in the set", "ℝ^M ∈ [0,1]", PLAS),
        ("S11", "Budgeted selection", "greedy knapsack, gain per token", "S ⊂ candidates", NEUT),
        ("S12", "Prompt assembly", "provenance tags, chronological blocks", "≤ B tokens", NEUT),
        ("S13", "Generation", "frozen p(y | C)", "answer text", FROZEN),
        ("S14", "Feedback", "probe ground truth → ℓ ∈ {0,1}", "labels", OK),
        ("S15", "Update", "ℒ = BCE + λ_Ω Ω; θ, β, b only", "|θ| = 1,080", OK),
    ]

    y = 132.0
    for tag, name, detail, shape, color in stages:
        ax.add_patch(FancyBboxPatch((4, y - 4.4), 132, 4.9,
                                    boxstyle="round,pad=0,rounding_size=0.5",
                                    fc="#FFFFFF", ec=LINE, lw=0.9, zorder=2))
        ax.add_patch(Rectangle((4, y - 4.4), 1.5, 4.9, fc=color, ec="none", zorder=3))
        ax.text(7.5, y - 1.4, tag, fontsize=8.6, fontweight="bold", color=color, va="center")
        ax.text(15, y - 1.4, name, fontsize=9.0, fontweight="bold", color=INK, va="center")
        ax.text(50, y - 1.4, detail, fontsize=7.9, color=INK, va="center")
        ax.text(134, y - 1.4, shape, fontsize=7.9, color=color, va="center", ha="right",
                fontweight="bold")
        if y > 30:
            arrow(ax, (10, y - 4.6), (10, y - 5.9), LINE, lw=1.1)
        y -= 6.1

    notes = [
        ("The invariant that makes S7 lossless", "feature_dim must equal 2^N exactly — "
         "1024 = 2¹⁰. Config.validate() refuses any other combination."),
        ("The trap at S6", "normalisation destroys magnitude, so every ρ feature is encoded "
         "as a bounded ratio or an angle BEFORE this step."),
        ("The bug S5 was built to prevent", "unbalanced blocks let one feature family absorb "
         "the amplitude mass — participation ratio fell to 61/1024 before block balancing, 189 after."),
    ]
    ax.plot([3, 152], [30.0, 30.0], color=LINE, lw=1.0)
    yy = 25
    for title, body in notes:
        ax.text(4, yy, "▸ " + title, fontsize=8.4, fontweight="bold", color=WARN)
        ax.text(6.5, yy - 2.6, body, fontsize=7.8, color=INK, va="top", wrap=True)
        yy -= 8.2

    fig.tight_layout()
    p = OUT / "DIAG_B_pipeline.png"
    fig.savefig(p, dpi=300, bbox_inches="tight", facecolor=PAPER)
    plt.close(fig)
    return p


# ============================================================================
# DIAGRAM C — the circuit
# ============================================================================
def diagram_c() -> Path:
    fig, ax = new_ax(16.0, 9.6, (0, 160), (0, 96))
    ax.text(3, 92, "The 10-qubit Quantum Relevance Engine", fontsize=17,
            fontweight="bold", color=INK)
    ax.text(3, 88.2, "Amplitude encoding → brickwall SU(4) ansatz → local readout → "
                     "one relevance score", fontsize=10, color=MUTE)
    ax.plot([3, 157], [85.8, 85.8], color=LINE, lw=1.2)

    n, ytop, dy = 10, 78.0, 4.6
    xs, xe = 34, 116

    def qy(q):
        return ytop - q * dy

    for q in range(n):
        ax.plot([xs, xe], [qy(q), qy(q)], color=MUTE, lw=0.9, zorder=1)
        ax.text(12.5, qy(q), "|0⟩", fontsize=8, color=MUTE, ha="right", va="center")
        ax.text(15.5, qy(q), f"q{q}", fontsize=8, color=INK, ha="right", va="center")

    ax.add_patch(FancyBboxPatch((17, qy(9) - 3), 15, (n - 1) * dy + 6,
                                boxstyle="round,pad=0,rounding_size=0.6",
                                fc=NEUT_BG, ec=NEUT, lw=1.3, zorder=2))
    ax.text(24.5, qy(4) + 2.0, "STATE\nPREPARATION", fontsize=9, fontweight="bold",
            color=NEUT, ha="center", va="center", zorder=3)
    ax.text(24.5, qy(6) + 0.6, "|ψ₀⟩ = Σⱼ xⱼ|j⟩\n‖x‖₂ = 1\nMöttönen, O(2ᴺ) depth",
            fontsize=7.2, color=INK, ha="center", va="center", zorder=3)

    even = [(0, 1), (2, 3), (4, 5), (6, 7), (8, 9)]
    odd = [(1, 2), (3, 4), (5, 6), (7, 8)]
    layer_x = [39, 49, 59, 69, 79, 89, 99, 109]
    for li, lx in enumerate(layer_x):
        for (a, b) in (even if li % 2 == 0 else odd):
            ya, yb = qy(a), qy(b)
            ax.add_patch(FancyBboxPatch((lx - 2.6, min(ya, yb) - 1.3), 5.2,
                                        abs(ya - yb) + 2.6,
                                        boxstyle="round,pad=0,rounding_size=0.35",
                                        fc=PLAS_BG, ec=PLAS, lw=0.9, zorder=3))
            ax.text(lx, (ya + yb) / 2, "V(θ)", fontsize=6.6, color=PLAS, ha="center",
                    va="center", fontweight="bold", zorder=4)

    ax.annotate("", xy=(112, 82.5), xytext=(37, 82.5),
                arrowprops=dict(arrowstyle="<->", color=PLAS, lw=1.0))
    ax.text(39, 84.0, "layer 1", fontsize=7.4, color=PLAS, ha="center")
    ax.text(74, 84.0, "…", fontsize=7.4, color=PLAS, ha="center")
    ax.text(109, 84.0, "layer 8", fontsize=7.4, color=PLAS, ha="center")
    ax.text(74, 80.2, "depth L = 8   ·   9 gates per layer   ·   15 parameters per gate   "
                      "⇒   |θ| = 8 × 9 × 15 = 1,080",
            fontsize=8.4, color=PLAS, ha="center", fontweight="bold")

    ax.add_patch(FancyBboxPatch((122, qy(9) - 3.5), 34, 11,
                                boxstyle="round,pad=0,rounding_size=0.6",
                                fc="#FFFFFF", ec=OK, lw=1.3, zorder=3))
    ax.text(139, qy(9) + 5.4, "MEASUREMENT  ·  qubit 9", fontsize=8.6,
            fontweight="bold", color=OK, ha="center")
    ax.text(139, qy(9) + 1.2, "⟨Z₉⟩ = ⟨ψ_L| Z₉ |ψ_L⟩ ∈ [−1, 1]\n"
                              "r = σ(β⟨Z₉⟩ + b) ∈ [0, 1]",
            fontsize=7.6, color=INK, ha="center", va="center")
    arrow(ax, (116, qy(9)), (122, qy(9)), OK)

    ax.plot([3, 157], [30.5, 30.5], color=LINE, lw=1.0)
    ax.text(3, 27.5, "FOUR DESIGN DECISIONS, AND THE MEASUREMENT BEHIND EACH",
            fontsize=9, fontweight="bold", color=INK)

    facts = [
        (3, 22, "Why a general SU(4) gate",
         ["15 real parameters span the full su(4) algebra, so a single",
          "gate reaches any two-qubit unitary. Restricted Ry/Rz/CNOT",
          "blocks cannot, and would need far greater depth."]),
        (82, 22, "Why a LOCAL readout on qubit 9",
         ["Switching to a global ⟨Z⊗…⊗Z⟩ costs 0.092 MRR in ablation,",
          "and deep circuits with global observables are the classic",
          "barren-plateau regime."]),
        (3, 11, "Why depth ≥ 5 is mandatory",
         ["Light cone: a gate on (q0,q1) needs ⌈N/2⌉ brickwall layers to",
          "reach q9. Below that its gradient is not small but exactly",
          "zero — measured at 3.6e-16 across 25 random initialisations."]),
        (82, 11, "Why the norm never drifts",
         ["Every V(θ) is unitary, so ‖ψ‖ = 1 holds exactly at every layer.",
          "Verified numerically to 1e-4, and the whole circuit is",
          "cross-checked against PennyLane (fidelity 1.000)."]),
    ]
    for x, y, title, body in facts:
        ax.text(x, y, "▸ " + title, fontsize=8.4, fontweight="bold", color=PLAS)
        for i, ln in enumerate(body):
            ax.text(x + 2.2, y - 2.6 - i * 2.3, ln, fontsize=7.5, color=INK, va="top")

    fig.tight_layout()
    p = OUT / "DIAG_C_circuit.png"
    fig.savefig(p, dpi=300, bbox_inches="tight", facecolor=PAPER)
    plt.close(fig)
    return p


# ============================================================================
# DIAGRAM D — technology stack
# ============================================================================
def diagram_d() -> Path:
    fig, ax = new_ax(14.0, 9.0, (0, 140), (0, 90))
    ax.text(3, 86, "Technology stack, by layer", fontsize=17, fontweight="bold", color=INK)
    ax.text(3, 82.2, "What each layer runs on, and why that choice rather than the obvious one",
            fontsize=10, color=MUTE)
    ax.plot([3, 137], [79.8, 79.8], color=LINE, lw=1.2)

    rows = [
        ("Data generation", "numpy · deterministic templates",
         "an LLM writing prose freely would destroy exact ground truth", STAB),
        ("Segmentation & IO", "Python 3.11+ · pandas · pyarrow · JSONL",
         "round-trip tested: concat(SU) reproduces the turn", STAB),
        ("Text encoder", "sentence-transformers · MiniLM-L6-v2 · torch",
         "frozen and revision-pinned; a silent upgrade invalidates every cache", STAB),
        ("Geometry", "numpy / scipy SVD whitening + torch InfoNCE",
         "shrinkage ε chosen from a measured sweep, not defaulted", STAB),
        ("Memory store", "SQLite (WAL) with append-only TRIGGERS",
         "immutability enforced by the database, not by caller discipline", STAB),
        ("Retrieval", "FAISS IndexFlatIP + causality mask",
         "exact search: an approximate index would confound two failure modes", STAB),
        ("Graph", "networkx-style DAG in SQLite edges",
         "contradictions stored as edges instead of resolved by overwrite", STAB),
        ("Quantum simulation", "custom batched statevector in torch",
         "differentiable end-to-end; all M candidates in one tensor contraction", PLAS),
        ("Quantum verification", "PennyLane (independent cross-check)",
         "fidelity 1.000 and ⟨Z⟩ agreement < 1e-5 against a reference simulator", PLAS),
        ("Gradients", "autodiff (training) + parameter shift (hardware parity)",
         "agreement 8.6e-9 in the product gate form", PLAS),
        ("Optimisation", "torch Adam · per-arm tuned learning rates",
         "each arm at its own best lr — otherwise the comparison is a handicap", PLAS),
        ("Noise modelling", "trajectory-sampled depolarising channels",
         "4^N density matrices are infeasible at N=10; trajectories converge as 1/√R", PLAS),
        ("Assembly", "greedy submodular knapsack · HF tokenizer",
         "exact token counts; estimates are how budgets silently overrun", NEUT),
        ("Evaluation", "paired bootstrap over threads · JSONL metric log",
         "threads are the independent unit; resampling probes manufactures significance", OK),
        ("Figures", "matplotlib with one shared style module",
         "45 figure ids with fixed pass/fail reads, regenerable from logs", OK),
        ("Testing", "pytest — 88 tests",
         "physics, invariants and ground truth are all asserted, not assumed", OK),
    ]

    y = 75
    ax.text(6, y + 2.6, "LAYER", fontsize=7.6, fontweight="bold", color=MUTE)
    ax.text(38, y + 2.6, "TECHNOLOGY", fontsize=7.6, fontweight="bold", color=MUTE)
    ax.text(76, y + 2.6, "WHY THIS, RATHER THAN THE DEFAULT", fontsize=7.6,
            fontweight="bold", color=MUTE)
    for name, tech, why, color in rows:
        ax.add_patch(Rectangle((4, y - 2.0), 133, 4.2, fc="#FFFFFF" if y % 8 else "#F7F9FA",
                               ec="none", zorder=1))
        ax.add_patch(Rectangle((4, y - 2.0), 1.2, 4.2, fc=color, ec="none", zorder=2))
        ax.text(6.5, y, name, fontsize=8.0, color=INK, va="center", fontweight="bold")
        ax.text(38, y, tech, fontsize=7.8, color=color, va="center")
        ax.text(76, y, why, fontsize=7.6, color=MUTE, va="center")
        ax.plot([4, 137], [y - 2.0, y - 2.0], color=LINE, lw=0.5, zorder=3)
        y -= 4.4

    fig.tight_layout()
    p = OUT / "DIAG_D_stack.png"
    fig.savefig(p, dpi=300, bbox_inches="tight", facecolor=PAPER)
    plt.close(fig)
    return p


if __name__ == "__main__":
    for fn in (diagram_a, diagram_b, diagram_c, diagram_d):
        print("wrote", fn())
