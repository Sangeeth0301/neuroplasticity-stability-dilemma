"""Architecture and circuit diagrams for the final deck.

    python scripts/figures/make_slide_diagrams.py  ->  figures/final/{architecture,circuit}.{png,svg}

Both reflect the final model: the circuit is read out on all 10 qubits with learned weights.
"""
from __future__ import annotations

from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import Circle, FancyArrowPatch, FancyBboxPatch, Rectangle

OUT = Path(__file__).resolve().parents[2] / "figures" / "final"
OUT.mkdir(parents=True, exist_ok=True)

INK, MUTE, LINE = "#14191D", "#5B6670", "#C9D1D6"
TEAL, TEAL_BG = "#0B5C73", "#EAF3F6"
MAG, MAG_BG = "#A3226A", "#FAEDF4"
GREEN, GREEN_BG = "#2A7D57", "#EDF6F1"
SLATE, SLATE_BG = "#4A5A63", "#EEF1F3"
plt.rcParams.update({"font.family": "DejaVu Sans"})
FS = [1.0]


def rbox(ax, x, y, w, h, fc, ec, lw=1.6, r=1.0, z=2):
    ax.add_patch(FancyBboxPatch((x, y), w, h, boxstyle=f"round,pad=0,rounding_size={r}",
                                fc=fc, ec=ec, lw=lw, zorder=z))


def text(ax, x, y, s, size=10, color=INK, weight="normal", ha="center", va="center",
         style="normal", z=6):
    ax.text(x, y, s, fontsize=size * FS[0], color=color, fontweight=weight, ha=ha, va=va,
            fontstyle=style, zorder=z)


def arrow(ax, p0, p1, color=SLATE, lw=1.8, ls="-", z=5, rad=0.0):
    ax.add_patch(FancyArrowPatch(p0, p1, arrowstyle="-|>", mutation_scale=13, color=color,
                                 lw=lw, ls=ls, zorder=z, connectionstyle=f"arc3,rad={rad}"))


def save(fig, name):
    fig.savefig(OUT / f"{name}.png", dpi=220, bbox_inches="tight", facecolor="white")
    fig.savefig(OUT / f"{name}.svg", bbox_inches="tight", facecolor="white")
    plt.close(fig)


# =============================================================================
# Architecture
# =============================================================================
def architecture():
    FS[0] = 1.14
    fig, ax = plt.subplots(figsize=(15.8, 7.4))
    ax.set_xlim(0, 158)
    ax.set_ylim(0, 74)
    ax.axis("off")

    def band(y, h, title, fc, ec):
        rbox(ax, 2, y, 146, h, fc, ec, 1.4, 1.2, z=1)
        text(ax, 5, y + h - 2.6, title, 10.5, ec, "bold", ha="left")

    def node(x, y, w, h, head, sub, ec, fc="white"):
        rbox(ax, x, y, w, h, fc, ec, 1.6, 1.0)
        text(ax, x + w / 2, y + h * 0.66, head, 11.5, INK, "bold")
        text(ax, x + w / 2, y + h * 0.29, sub, 8.4, MUTE)

    # stability band
    band(54, 18, "STABILITY  ·  classical, add-only", TEAL_BG, TEAL)
    xs = [6, 41, 76, 111]
    heads = [("Ingest and segment", "chat turns → short text units"),
             ("Frozen encoder", "MiniLM, never fine-tuned"),
             ("Memory ℳ", "append-only store + entity graph"),
             ("Retrieve", "top-M earlier candidates")]
    for x, (h, s) in zip(xs, heads):
        node(x, 56, 32, 10, h, s, TEAL)
    for x in xs[:-1]:
        arrow(ax, (x + 32, 61), (x + 35, 61), TEAL)

    # interface band
    band(36, 15, "INTERFACE  ·  the only link between the two sides", SLATE_BG, SLATE)
    node(26, 38, 44, 9, "Feature assembly", "query | candidate | interaction | context  →  1024 numbers", SLATE)
    node(80, 38, 44, 9, "Amplitude encoding", "unit vector → state of 10 qubits", SLATE)
    arrow(ax, (70, 42.5), (80, 42.5), SLATE)

    # plasticity band
    band(18, 15, "PLASTICITY  ·  quantum, the only part that learns", MAG_BG, MAG)
    node(26, 20, 44, 9, "Variational circuit", "10 qubits · 8 layers · 1,080 angles", MAG)
    node(80, 20, 44, 9, "Readout", "measure all 10 qubits, learned weights → score", MAG)
    arrow(ax, (70, 24.5), (80, 24.5), MAG)

    # delivery band
    band(1, 14, "DELIVERY", GREEN_BG, GREEN)
    node(10, 2.2, 36, 8.6, "Context assembler", "pick facts within a token budget", GREEN)
    node(57, 2.2, 36, 8.6, "Frozen language model", "inference only, no gradient", SLATE, "#F2F4F5")
    node(104, 2.2, 36, 8.6, "Evaluation", "known answers → training labels", GREEN)
    arrow(ax, (46, 6.5), (57, 6.5), GREEN)
    arrow(ax, (93, 6.5), (104, 6.5), SLATE)

    # vertical flow
    for y0, y1, lab in [(54, 51, "candidates"), (36, 33, "state |ψ₀⟩"), (18, 15, "score r")]:
        arrow(ax, (75, y0), (75, y1), SLATE, lw=1.6)
        text(ax, 82, (y0 + y1) / 2, lab, 8.5, MUTE, ha="left", style="italic")

    # gradient loop, routed through the right margin
    ax.plot([140, 153, 153], [6.5, 6.5, 25.5], color=MAG, lw=1.8, ls="--", zorder=3)
    arrow(ax, (153, 25.5), (148.3, 25.5), MAG, ls="--")
    ax.text(156, 16, "gradient step: changes only θ and w", fontsize=10.5, color=MAG, fontweight="bold",
            rotation=90, ha="center", va="center")
    fig.tight_layout()
    save(fig, "architecture")


# =============================================================================
# Circuit
# =============================================================================
def circuit():
    FS[0] = 1.14
    fig, ax = plt.subplots(figsize=(14.2, 7.6))
    ax.set_xlim(0, 196)
    ax.set_ylim(2, 109)
    ax.axis("off")

    n = 10
    top, gap = 98.0, 5.6
    wy = [top - i * gap for i in range(n)]          # q0 at the top
    x_enc0, x_enc1 = 12, 33
    x_l0, lw_, gw = 40, 9.2, 5.0                     # first layer x, layer pitch, gate width
    x_end = x_l0 + 8 * lw_ + 1

    # wires + labels
    for i, y in enumerate(wy):
        ax.plot([8, x_end + 3], [y, y], color="#8A969E", lw=1.1, zorder=1)
        text(ax, 5.5, y, f"q{i}", 9.5, MUTE)
        text(ax, 10.2, y, "|0⟩", 7.8, MUTE)

    # amplitude encoding block
    rbox(ax, x_enc0, wy[-1] - 3.2, x_enc1 - x_enc0, wy[0] - wy[-1] + 6.4, SLATE_BG, SLATE, 1.6, 1.2)
    ymid = (wy[0] + wy[-1]) / 2
    text(ax, (x_enc0 + x_enc1) / 2, ymid + 9, "Amplitude", 10.5, SLATE, "bold")
    text(ax, (x_enc0 + x_enc1) / 2, ymid + 5.5, "encoding", 10.5, SLATE, "bold")
    text(ax, (x_enc0 + x_enc1) / 2, ymid - 1, "|ψ₀⟩ = Σᵢ xᵢ |i⟩", 10.5)
    text(ax, (x_enc0 + x_enc1) / 2, ymid - 6, "x ∈ ℝ¹⁰²⁴", 9, MUTE)
    text(ax, (x_enc0 + x_enc1) / 2, ymid - 9.3, "unit length", 9, MUTE)

    # 8 layers, each = an even row of 5 gates followed by an odd row of 4 gates (9 gates per layer)
    pitch = lw_ / 2
    gw2 = pitch - 1.0
    for col in range(16):
        x0 = x_l0 + col * pitch + (pitch - gw2) / 2
        start_q = 0 if col % 2 == 0 else 1
        for a in range(start_q, n - 1, 2):
            ya, yb = wy[a], wy[a + 1]
            ax.add_patch(Rectangle((x0, yb - 2.1), gw2, ya - yb + 4.2, fc=MAG_BG, ec=MAG,
                                   lw=1.3, zorder=3))
            text(ax, x0 + gw2 / 2, (ya + yb) / 2, "V", 9.5, MAG, "bold")
    for L in range(8):
        xc = x_l0 + L * lw_ + lw_ / 2
        text(ax, xc, wy[0] + 5.2, f"L{L + 1}", 9.5, MAG, "bold")
        if L < 7:
            ax.plot([x_l0 + (L + 1) * lw_] * 2, [wy[0] + 3.2, wy[-1] - 3.0], color=MAG, lw=0.7,
                    ls=":", alpha=0.6, zorder=0)
    ax.plot([x_l0 + 1, x_end - 1], [wy[0] + 7.6] * 2, color=MAG, lw=1.4)
    text(ax, (x_l0 + x_end) / 2, wy[0] + 10, "8 layers, each with two staggered rows (9 gates)",
         10.5, MAG, "bold")

    # measurement of every qubit
    xm = x_end + 5
    for y in wy:
        ax.add_patch(Rectangle((xm, y - 2.0), 7.5, 4.0, fc=GREEN_BG, ec=GREEN, lw=1.3, zorder=3))
        text(ax, xm + 3.75, y, "⟨Z⟩", 8.6, GREEN, "bold")
        ax.plot([xm + 7.5, xm + 13], [y, y], color=GREEN, lw=1.1, zorder=2)
        text(ax, xm + 16.2, y, "× w" + str(wy.index(y)), 8.6, GREEN, "bold")
    sx, sy = xm + 33, ymid
    for y in wy:
        ax.plot([xm + 19.5, sx - 3], [y, sy], color=GREEN, lw=0.9, alpha=0.75, zorder=2)
    ax.add_patch(Circle((sx, sy), 3.1, fc="white", ec=GREEN, lw=1.7, zorder=4))
    text(ax, sx, sy, "Σ", 14, GREEN, "bold")
    text(ax, xm + 12, wy[0] + 9.2, "measure every qubit", 9.5, GREEN, "bold")
    text(ax, xm + 12, wy[0] + 6.2, "and weight each one", 9.5, GREEN, "bold")

    # score box
    bx = sx + 7
    rbox(ax, bx, sy - 7.5, 25.5, 15, "white", GREEN, 1.7, 1.2)
    text(ax, bx + 12.75, sy + 3.0, "z = Σ w_q ⟨Z_q⟩", 10)
    text(ax, bx + 12.75, sy - 1.8, "r = σ( β z + b )", 11, INK, "bold")
    text(ax, bx + 12.75, sy - 5.4, "relevance in [0, 1]", 8.8, MUTE)
    arrow(ax, (sx + 3.1, sy), (bx, sy), GREEN, lw=1.8)
    arrow(ax, (bx + 25.5, sy), (bx + 31, sy), GREEN, lw=1.8)
    text(ax, bx + 31.5, sy, "r", 14, GREEN, "bold", ha="left")

    # ---------------- lower left: one gate, opened up -------------------
    gy0 = 6
    rbox(ax, 4, gy0, 112, 29, "#FBFBFC", LINE, 1.2, 1.2, z=1)
    text(ax, 8, gy0 + 25.5, "One gate V(θ), opened up", 11.5, INK, "bold", ha="left")
    text(ax, 8, gy0 + 21.7, "A general two-qubit unitary has 15 free angles. A compiler writes it as 3 CNOTs "
         "plus single-qubit rotations.", 8.0, MUTE, ha="left")
    ya, yb = gy0 + 14, gy0 + 6.5
    ax.plot([9, 76], [ya, ya], color="#8A969E", lw=1.1, zorder=1)
    ax.plot([9, 76], [yb, yb], color="#8A969E", lw=1.1, zorder=1)
    text(ax, 6.6, ya, "a", 9, MUTE)
    text(ax, 6.6, yb, "b", 9, MUTE)

    def u(x, y, lab, w=7.4):
        ax.add_patch(Rectangle((x - w / 2, y - 2.3), w, 4.6, fc=MAG_BG, ec=MAG, lw=1.2, zorder=3))
        text(ax, x, y, lab, 8.2, MAG, "bold")

    def cnot(x, c, t):
        ax.plot([x, x], [yb, ya], color=MAG, lw=1.5, zorder=3)
        ax.add_patch(Circle((x, c), 0.85, fc=MAG, ec=MAG, zorder=4))
        ax.add_patch(Circle((x, t), 2.0, fc="white", ec=MAG, lw=1.5, zorder=4))
        ax.plot([x - 2.0, x + 2.0], [t, t], color=MAG, lw=1.3, zorder=5)
        ax.plot([x, x], [t - 2.0, t + 2.0], color=MAG, lw=1.3, zorder=5)

    u(15, ya, "U"), u(15, yb, "U")
    cnot(24, yb, ya)
    u(33, ya, "Rz"), u(33, yb, "Ry")
    cnot(42, ya, yb)
    u(51, yb, "Ry")
    cnot(60, yb, ya)
    u(69, ya, "U"), u(69, yb, "U")
    text(ax, 15, gy0 + 2.2, "U: 3 angles", 8.2, MUTE)
    text(ax, 69, gy0 + 2.2, "U: 3 angles", 8.2, MUTE)
    text(ax, 33, gy0 + 2.2, "1 angle each", 8.2, MUTE)
    text(ax, 51, gy0 + 2.2, "1 angle", 8.2, MUTE)
    text(ax, 96, ya, "4 × 3 + 3 = 15 angles", 10, MAG, "bold")
    text(ax, 96, yb, "3 CNOTs", 10, MAG, "bold")

    # ---------------- lower right: the numbers ---------------------------
    rbox(ax, 120, gy0, 74, 29, "#FBFBFC", LINE, 1.2, 1.2, z=1)
    text(ax, 124, gy0 + 25.5, "What is trained", 11.5, INK, "bold", ha="left")
    rows = [("8 × 9 × 15", "= 1,080 circuit angles θ"),
            ("+ 10", "readout weights w_q"),
            ("+ 2", "scale β and bias b"),
            ("‖ψ‖ = 1", "after every layer, because each V is unitary")]
    for i, (a, b) in enumerate(rows):
        y = gy0 + 19.5 - i * 5.2
        text(ax, 124, y, a, 10.5, MAG, "bold", ha="left")
        text(ax, 142, y, b, 8.8, INK, ha="left")
    fig.tight_layout()
    save(fig, "circuit")


if __name__ == "__main__":
    architecture()
    circuit()
    print("ok")
