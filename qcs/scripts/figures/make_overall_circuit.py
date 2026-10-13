"""The whole project as a single hybrid circuit.

Classical pre-processing rails feed one 1024-dimensional unit vector into a 10-qubit
register; the register is evolved by 8 brickwall layers of general SU(4) gates; one qubit
is measured; the resulting scalar runs out through classical post-processing to an answer;
and the loss returns along a dashed rail to the only parameters that learn.

    figs/CIRCUIT_overall.png

Run:  python scripts/figures/make_overall_circuit.py
"""

from __future__ import annotations

from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
from matplotlib.patches import FancyArrowPatch, FancyBboxPatch  # noqa: E402

OUT = Path("figures") / "architecture"

INK = "#11161A"
MUTE = "#6B767E"
HAIR = "#D9E0E4"
PAPER = "#FFFFFF"
TEAL = "#0E5C74"
TEAL_BG = "#F1F7F9"
MAG = "#A32467"
MAG_BG = "#FBEFF4"
GREEN = "#2C7A58"
GREEN_BG = "#F0F7F3"
FROZ = "#5A6B75"
FROZ_BG = "#EFF2F3"
CONE = "#C6852A"

plt.rcParams.update({"font.family": "DejaVu Sans", "figure.facecolor": PAPER,
                     "savefig.facecolor": PAPER})

N = 10
DEPTH = 8
EVEN = [(0, 1), (2, 3), (4, 5), (6, 7), (8, 9)]
ODD = [(1, 2), (3, 4), (5, 6), (7, 8)]


def rbox(ax, x, y, w, h, fc, ec, lw=1.3, z=3, r=0.8):
    ax.add_patch(FancyBboxPatch((x, y), w, h,
                                boxstyle=f"round,pad=0,rounding_size={r}",
                                fc=fc, ec=ec, lw=lw, zorder=z))


def stage(ax, x, y, w, h, title, detail, color, fill):
    rbox(ax, x, y, w, h, fill, color)
    ax.text(x + w / 2, y + h * 0.62, title, fontsize=8.8, color=INK, fontweight="bold",
            ha="center", va="center", zorder=4)
    if detail:
        ax.text(x + w / 2, y + h * 0.26, detail, fontsize=7.2, color=MUTE,
                ha="center", va="center", zorder=4)


def arrow(ax, p0, p1, color, lw=1.5, dashed=False, z=6):
    ax.add_patch(FancyArrowPatch(p0, p1, arrowstyle="-|>", mutation_scale=12,
                                 color=color, lw=lw, zorder=z, shrinkA=1, shrinkB=1,
                                 linestyle=(0, (5, 3)) if dashed else "-"))


def chip(ax, x, y, text, color, size=7.4):
    ax.text(x, y, text, fontsize=size, color=color, ha="center", va="center", zorder=8,
            bbox=dict(boxstyle="round,pad=0.3", fc=PAPER, ec=color, lw=0.9))


def main() -> Path:
    fig, ax = plt.subplots(figsize=(20.0, 9.6))
    ax.set_xlim(0, 252)
    ax.set_ylim(0, 122)
    ax.axis("off")

    # ---------------------------------------------------------------- header
    ax.text(5, 116, "The complete hybrid circuit", fontsize=21, fontweight="bold", color=INK)
    ax.text(5, 111.5, "Everything the project does to one (query, candidate) pair — "
                      "classical rails in, quantum register, measurement, classical rails out, "
                      "and the gradient returning",
            fontsize=10.5, color=MUTE)
    ax.plot([5, 247], [108, 108], color=HAIR, lw=1.1)

    ytop, dy = 88.0, 6.0

    def qy(q):
        return ytop - q * dy

    # ============================================ classical pre-processing
    rbox(ax, 5, 26, 46, 79, TEAL_BG, TEAL, lw=1.2, z=1, r=1.2)
    ax.text(7.5, 102.5, "CLASSICAL  ·  PRE-PROCESSING", fontsize=9, color=TEAL,
            fontweight="bold", va="center", zorder=4)
    ax.text(7.5, 99.0, "frozen after Phase 2 — nothing here learns", fontsize=7.4,
            color=MUTE, va="center", style="italic", zorder=4)

    pre = [
        (86.5, "Ingest & segment", "turn → semantic units", "text"),
        (72.0, "Frozen encoder", "MiniLM · no_grad", "ℝ³⁸⁴"),
        (57.5, "Whiten + project", "fit once, then frozen", "ℝ²⁵⁶"),
        (43.0, "Feature assembly", "[ p_q ‖ p_c ‖ p_q⊙p_c ‖ ρ ]", "ℝ¹⁰²⁴"),
        (30.5, "L2 normalise", "‖x‖₂ = 1", "unit sphere"),
    ]
    for i, (y, t, d, shape) in enumerate(pre):
        stage(ax, 8, y, 40, 10.5, t, d, TEAL, PAPER)
        chip(ax, 44.5, y + 12.6, shape, TEAL, 7.0) if i else None
        if i < len(pre) - 1:
            arrow(ax, (28, y), (28, y - 4.0), TEAL, 1.3)
    chip(ax, 28, 27.6, "x ∈ ℝ¹⁰²⁴,  ‖x‖₂ = 1", TEAL, 7.6)

    # bus into the register
    ax.plot([51, 57], [35.8, 35.8], color=TEAL, lw=1.8, zorder=6)
    ax.plot([57, 57], [35.8, qy(4.5)], color=TEAL, lw=1.8, zorder=6)
    arrow(ax, (57, qy(4.5)), (61.5, qy(4.5)), TEAL, 1.8)

    # ============================================ quantum register
    rbox(ax, 60, 20, 128, 85, MAG_BG, MAG, lw=1.2, z=1, r=1.2)
    ax.text(62.5, 102.5, "QUANTUM  ·  THE ONLY COMPONENT THAT LEARNS", fontsize=9,
            color=MAG, fontweight="bold", va="center", zorder=4)
    ax.text(62.5, 99.0, "1,080 angles on a compact manifold — they wrap, they cannot inflate",
            fontsize=7.4, color=MUTE, va="center", style="italic", zorder=4)

    xs, xe = 84.0, 168.0
    for q in range(N):
        ax.plot([xs, xe], [qy(q), qy(q)], color="#8A959B", lw=1.0, zorder=2)
        ax.text(xs - 1.6, qy(q), f"q{q}", fontsize=7.6, color=INK, ha="right",
                va="center", fontweight="bold", zorder=4)

    rbox(ax, 62, qy(9) - 3.4, 20, (N - 1) * dy + 6.8, FROZ_BG, FROZ, lw=1.4)
    ax.text(72, qy(3.6), "STATE\nPREPARATION", fontsize=8.4, fontweight="bold", color=FROZ,
            ha="center", va="center", zorder=4)
    ax.text(72, qy(6.0), "|ψ₀⟩ = Σⱼ xⱼ|j⟩", fontsize=8.6, color=INK, ha="center",
            va="center", zorder=4)
    ax.text(72, qy(7.1), "1024 amplitudes\nlossless", fontsize=7.0, color=MUTE,
            ha="center", va="center", zorder=4)

    cols = 2 * DEPTH
    col_x = np.linspace(88.5, 164.0, cols)
    for ci, x in enumerate(col_x):
        for (a, b) in (EVEN if ci % 2 == 0 else ODD):
            ya, yb = qy(a), qy(b)
            rbox(ax, x - 1.9, min(ya, yb) - 1.5, 3.8, abs(ya - yb) + 3.0,
                 MAG_BG, MAG, lw=0.95, z=4, r=0.4)
            ax.text(x, (ya + yb) / 2, "V", fontsize=6.6, color=MAG, ha="center",
                    va="center", fontweight="bold", zorder=5)

    for layer in range(DEPTH):
        x0, x1 = col_x[2 * layer] - 2.6, col_x[2 * layer + 1] + 2.6
        ax.plot([x0, x1], [qy(0) + 4.2, qy(0) + 4.2], color=MAG, lw=0.9, zorder=4)
        ax.text((x0 + x1) / 2, qy(0) + 6.0, f"L{layer + 1}", fontsize=7.0, color=MAG,
                ha="center", va="center", fontweight="bold", zorder=4)
    ax.text(126, qy(9) - 7.0, "8 layers × 9 general SU(4) gates × 15 parameters = 1,080 angles",
            fontsize=8.6, color=MAG, ha="center", fontweight="bold", zorder=4)

    # measurement
    rbox(ax, 172, qy(9) - 4.4, 30, 12.5, GREEN_BG, GREEN, lw=1.4)
    ax.text(187, qy(9) + 5.2, "MEASURE  q9", fontsize=8.6, fontweight="bold", color=GREEN,
            ha="center", zorder=4)
    ax.text(187, qy(9) + 1.6, "⟨Z₉⟩ ∈ [−1, 1]", fontsize=8.6, color=INK, ha="center", zorder=4)
    ax.text(187, qy(9) - 1.8, "r = σ(β⟨Z₉⟩ + b)", fontsize=7.6, color=MUTE, ha="center", zorder=4)
    arrow(ax, (168, qy(9)), (172, qy(9)), GREEN, 1.6)
    for q in range(N - 1):
        ax.plot([168, 170.5], [qy(q), qy(q)], color="#C9D1D6", lw=1.0, zorder=2)
    ax.text(170.0, qy(4), "traced out", fontsize=7.0, color=MUTE, rotation=90,
            ha="center", va="center", zorder=4)

    # ============================================ classical post-processing
    rbox(ax, 206, 26, 42, 79, GREEN_BG, GREEN, lw=1.2, z=1, r=1.2)
    ax.text(208.5, 102.5, "CLASSICAL  ·  DELIVERY", fontsize=9, color=GREEN,
            fontweight="bold", va="center", zorder=4)
    ax.text(208.5, 99.0, "budgeted, and the generator is frozen", fontsize=7.4,
            color=MUTE, va="center", style="italic", zorder=4)

    post = [
        (86.5, "Scores for all candidates", "r ∈ [0,1]^M", PAPER, GREEN),
        (72.0, "Budgeted knapsack", "gain per token, ≤ B", PAPER, GREEN),
        (57.5, "Supersession filter", "stale ⇒ dropped or labelled", PAPER, GREEN),
        (43.0, "Frozen LLM", "no gradient path exists", FROZ_BG, FROZ),
        (30.5, "Answer + probe label", "ℓ ∈ {0,1}", PAPER, GREEN),
    ]
    for i, (y, t, d, fc, ec) in enumerate(post):
        stage(ax, 209, y, 36, 10.5, t, d, ec, fc)
        if i < len(post) - 1:
            arrow(ax, (227, y), (227, y - 4.0), GREEN, 1.3)

    ax.plot([202, 204], [qy(9), qy(9)], color=GREEN, lw=1.6, zorder=6)
    ax.plot([204, 204], [qy(9), 91.75], color=GREEN, lw=1.6, zorder=6)
    arrow(ax, (204, 91.75), (209, 91.75), GREEN, 1.6)

    # ============================================ gradient return
    dash = dict(color=MAG, lw=1.5, ls=(0, (5, 3)), zorder=6)
    ax.plot([227, 227], [26.0, 14.0], **dash)
    ax.plot([227, 126], [14.0, 14.0], **dash)
    arrow(ax, (126, 14.0), (126, 20.0), MAG, 1.5, dashed=True)
    ax.text(176, 16.0, "ℒ = BCE(r, ℓ) + λ_Ω·Ω(θ, θ*)        θ ← θ − η∇ℒ", fontsize=8.6,
            color=MAG, ha="center", fontweight="bold", zorder=7)
    ax.text(176, 11.2, "the gradient reaches 1,080 circuit angles plus β and b — and nothing else",
            fontsize=7.6, color=MUTE, ha="center", zorder=7)

    # memory side-rail
    rbox(ax, 5, 6, 46, 15, PAPER, TEAL, lw=1.2, r=0.8)
    ax.text(28, 17.0, "MEMORY  ℳ", fontsize=8.6, fontweight="bold", color=TEAL,
            ha="center", zorder=4)
    ax.text(28, 12.6, "append-only store · entity graph · salience", fontsize=7.2,
            color=INK, ha="center", zorder=4)
    ax.text(28, 8.8, "supplies the candidates — no gradient ever touches it", fontsize=7.0,
            color=MUTE, ha="center", style="italic", zorder=4)
    arrow(ax, (28, 21), (28, 26.5), TEAL, 1.4)

    # ============================================ footer legend
    ax.plot([5, 247], [2.6, 2.6], color=HAIR, lw=1.0)
    legend = [
        (5, TEAL, "classical · frozen or append-only"),
        (72, MAG, "quantum · the 1,080 angles that learn"),
        (146, GREEN, "delivery & evaluation"),
        (200, FROZ, "frozen generator"),
    ]
    for x, c, t in legend:
        rbox(ax, x, -2.6, 3.0, 2.0, c, c, lw=0, r=0.3)
        ax.text(x + 4.2, -1.6, t, fontsize=7.6, color=MUTE, va="center")
    ax.set_ylim(-5.5, 122)

    fig.tight_layout()
    OUT.mkdir(exist_ok=True)
    p = OUT / "CIRCUIT_overall.png"
    fig.savefig(p, dpi=300, bbox_inches="tight", facecolor=PAPER)
    fig.savefig(OUT / "CIRCUIT_overall.svg", bbox_inches="tight", facecolor=PAPER)
    plt.close(fig)
    return p


if __name__ == "__main__":
    print("wrote", main())
