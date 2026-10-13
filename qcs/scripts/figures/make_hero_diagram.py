"""The master architecture diagram — clean version.

Design rules, deliberately strict:
  · one flow direction, top to bottom, four bands
  · every node carries a title and AT MOST one line of detail
  · no bullet lists, no paragraphs, no annotations inside the drawing
  · colour means one thing only: teal = classical, magenta = quantum,
    slate = frozen, green = evaluation
  · anything that needs a sentence belongs in the report, not in the figure

Run:  python scripts/figures/make_hero_diagram.py
"""

from __future__ import annotations

from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
from matplotlib.patches import FancyArrowPatch, FancyBboxPatch  # noqa: E402

OUT = Path("figures") / "architecture"

INK = "#11161A"
MUTE = "#6B767E"
HAIR = "#D8DFE3"
PAPER = "#FFFFFF"

STAB = "#0E5C74"
STAB_BG = "#F1F7F9"
PLAS = "#A32467"
PLAS_BG = "#FBF0F5"
NEUT = "#41535C"
NEUT_BG = "#F4F6F7"
OK = "#2C7A58"
OK_BG = "#F0F7F3"
FROZ = "#5A6B75"

plt.rcParams.update({"font.family": "DejaVu Sans", "figure.facecolor": PAPER,
                     "savefig.facecolor": PAPER})


def node(ax, x, y, w, h, title, detail, color, fill="#FFFFFF", tsize=10.5, dsize=8.2):
    ax.add_patch(FancyBboxPatch((x, y), w, h,
                                boxstyle="round,pad=0,rounding_size=0.9",
                                fc=fill, ec=color, lw=1.5, zorder=3))
    cx = x + w / 2
    if detail:
        ax.text(cx, y + h * 0.60, title, fontsize=tsize, color=INK, fontweight="bold",
                ha="center", va="center", zorder=4)
        ax.text(cx, y + h * 0.28, detail, fontsize=dsize, color=MUTE,
                ha="center", va="center", zorder=4)
    else:
        ax.text(cx, y + h / 2, title, fontsize=tsize, color=INK, fontweight="bold",
                ha="center", va="center", zorder=4)


def band(ax, x, y, w, h, tag, color, fill):
    ax.add_patch(FancyBboxPatch((x, y), w, h,
                                boxstyle="round,pad=0,rounding_size=1.2",
                                fc=fill, ec=color, lw=1.1, zorder=1, alpha=1.0))
    ax.text(x + 2.6, y + h - 2.4, tag, fontsize=8.6, color=color, fontweight="bold",
            ha="left", va="center", zorder=4)


def flow(ax, p0, p1, color=MUTE, lw=1.8, dashed=False, z=5):
    ax.add_patch(FancyArrowPatch(p0, p1, arrowstyle="-|>", mutation_scale=15,
                                 color=color, lw=lw, zorder=z, shrinkA=1, shrinkB=1,
                                 linestyle=(0, (5, 3)) if dashed else "-"))


def chip(ax, x, y, text, color):
    ax.text(x, y, text, fontsize=8.4, color=color, ha="center", va="center",
            zorder=6, bbox=dict(boxstyle="round,pad=0.35", fc=PAPER, ec=color, lw=0.9))


def main() -> Path:
    fig, ax = plt.subplots(figsize=(15.0, 9.4))
    ax.set_xlim(0, 174)
    ax.set_ylim(0, 106)
    ax.axis("off")

    # ---------------- header ------------------------------------------------
    ax.text(6, 99.5, "Dual-Manifold ContextSync", fontsize=21, fontweight="bold", color=INK)
    ax.text(6, 95.2, "Continual memory for long, multi-user chat threads",
            fontsize=11.5, color=MUTE)
    ax.plot([6, 164], [92.0, 92.0], color=HAIR, lw=1.2)

    # ---------------- band 1 · classical ------------------------------------
    band(ax, 6, 71, 158, 18, "STABILITY  ·  CLASSICAL", STAB, STAB_BG)
    node(ax, 11, 74, 33, 10, "Ingest & segment", "chat turns → semantic units", STAB)
    node(ax, 50, 74, 33, 10, "Frozen encoder", "MiniLM · never fine-tuned", STAB)
    node(ax, 89, 74, 33, 10, "Memory  ℳ", "append-only store + entity graph", STAB)
    node(ax, 128, 74, 31, 10, "Retrieve", "top-M causal candidates", STAB)
    for x0, x1 in ((44, 50), (83, 89), (122, 128)):
        flow(ax, (x0, 79), (x1, 79), STAB)

    flow(ax, (85, 71), (85, 66), MUTE)
    chip(ax, 85, 68.5, "candidates", MUTE)

    # ---------------- band 2 · interface ------------------------------------
    band(ax, 6, 46, 158, 18, "INTERFACE", NEUT, NEUT_BG)
    node(ax, 30, 49, 45, 10, "Feature assembly", "query ‖ candidate ‖ interaction ‖ context", NEUT)
    node(ax, 95, 49, 45, 10, "Quantum encoding", "unit vector → 10-qubit amplitudes", NEUT)
    flow(ax, (75, 54), (95, 54), NEUT)
    chip(ax, 85, 56.6, "ℝ¹⁰²⁴", NEUT)

    flow(ax, (85, 46), (85, 41), MUTE)
    chip(ax, 85, 43.5, "|ψ₀⟩", MUTE)

    # ---------------- band 3 · quantum --------------------------------------
    band(ax, 6, 21, 158, 18, "PLASTICITY  ·  QUANTUM", PLAS, PLAS_BG)
    node(ax, 30, 24, 45, 10, "Variational circuit", "10 qubits · depth 8 · 1,080 angles", PLAS)
    node(ax, 95, 24, 45, 10, "Readout", "single-qubit measurement → score", PLAS)
    flow(ax, (75, 29), (95, 29), PLAS)
    chip(ax, 85, 31.6, "U(θ)|ψ₀⟩", PLAS)

    flow(ax, (85, 21), (85, 16), MUTE)
    chip(ax, 85, 18.5, "relevance r", MUTE)

    # ---------------- band 4 · delivery -------------------------------------
    band(ax, 6, -1, 158, 17, "DELIVERY", OK, OK_BG)
    node(ax, 14, 2, 40, 9.5, "Context assembler", "budgeted selection ≤ B tokens", OK)
    node(ax, 65, 2, 40, 9.5, "Frozen LLM", "inference only · no gradient", FROZ, fill="#F2F4F5")
    node(ax, 116, 2, 40, 9.5, "Evaluation", "probe ground truth → labels", OK)
    flow(ax, (54, 6.75), (65, 6.75), OK)
    flow(ax, (105, 6.75), (116, 6.75), FROZ)

    # ---------------- feedback ---------------------------------------------
    dash = dict(color=PLAS, lw=1.5, ls=(0, (5, 3)), zorder=5)
    ax.plot([156, 168], [6.75, 6.75], **dash)
    ax.plot([168, 168], [6.75, 36.4], **dash)
    ax.plot([168, 52.5], [36.4, 36.4], **dash)
    flow(ax, (52.5, 36.4), (52.5, 34), PLAS, lw=1.5, dashed=True)
    ax.text(169.6, 21, "θ update", fontsize=8.2, color=PLAS, rotation=90,
            ha="center", va="center", fontweight="bold")

    # ---------------- legend ------------------------------------------------
    ax.plot([6, 168], [-7.5, -7.5], color=HAIR, lw=1.0)
    items = [
        (8, STAB, "classical · append-only, nothing is learned"),
        (56, PLAS, "quantum · the only component that learns online"),
        (108, FROZ, "frozen · no gradient path exists"),
    ]
    for x, c, t in items:
        ax.add_patch(FancyBboxPatch((x, -11.6), 3.2, 2.2, boxstyle="round,pad=0,rounding_size=0.4",
                                    fc=c, ec=c, zorder=4))
        ax.text(x + 4.6, -10.5, t, fontsize=8.4, color=MUTE, ha="left", va="center")
    ax.text(168, -10.5, "solid = data   ·   dashed = gradient", fontsize=8.4, color=MUTE,
            ha="right", va="center")

    ax.set_ylim(-14.5, 106)
    fig.tight_layout()
    OUT.mkdir(exist_ok=True)
    p = OUT / "DIAG_A_architecture.png"
    fig.savefig(p, dpi=300, bbox_inches="tight", facecolor=PAPER)
    fig.savefig(OUT / "DIAG_A_architecture.svg", bbox_inches="tight", facecolor=PAPER)
    plt.close(fig)
    return p


if __name__ == "__main__":
    print("wrote", main())
