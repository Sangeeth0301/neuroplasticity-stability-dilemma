"""The definitive quantum-circuit figures.

    figs/CIRCUIT_main.png        the full 10-qubit circuit, with the light cone drawn
    figs/CIRCUIT_gate.png        anatomy of one general SU(4) gate
    figs/CIRCUIT_pennylane.png   the SAME circuit rendered by PennyLane from the real model

The third figure exists for a reason: it is drawn by an independent library directly
from the model object, so the schematic cannot quietly drift away from the code.

Run:  python scripts/figures/make_circuit_figure.py
"""

from __future__ import annotations

from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
from matplotlib.patches import FancyArrowPatch, FancyBboxPatch, Polygon  # noqa: E402

OUT = Path("figures") / "circuit"

INK = "#11161A"
MUTE = "#6B767E"
HAIR = "#D8DFE3"
PAPER = "#FFFFFF"
PLAS = "#A32467"
PLAS_BG = "#FBEFF4"
PLAS_SOFT = "#F6DDE8"
NEUT = "#41535C"
NEUT_BG = "#F1F4F5"
OK = "#2C7A58"
CONE = "#E9A23B"

plt.rcParams.update({"font.family": "DejaVu Sans", "figure.facecolor": PAPER,
                     "savefig.facecolor": PAPER})

N = 10
DEPTH = 8
EVEN = [(0, 1), (2, 3), (4, 5), (6, 7), (8, 9)]
ODD = [(1, 2), (3, 4), (5, 6), (7, 8)]


def light_cone_frontier():
    """Which qubits a gate acting on (q0,q1) in layer 1 can influence, sub-column by
    sub-column. Returns [(column_index, max_qubit_reached), ...]."""
    influenced = {0, 1}
    out = []
    col = 0
    for layer in range(DEPTH):
        for pairs in (EVEN, ODD):
            for a, b in pairs:
                if a in influenced or b in influenced:
                    influenced |= {a, b}
            out.append((col, max(influenced)))
            col += 1
    return out


# ---------------------------------------------------------------- main figure
def circuit_main() -> Path:
    fig, ax = plt.subplots(figsize=(17.0, 8.6))
    ax.set_xlim(0, 200)
    ax.set_ylim(-14, 86)
    ax.axis("off")

    ax.text(4, 80, "The 10-qubit Quantum Relevance Engine", fontsize=20,
            fontweight="bold", color=INK)
    ax.text(4, 75.6, "8 layers of 9 general SU(4) gates  ·  1,080 trainable angles  ·  "
                     "every gate unitary, so ‖ψ‖ = 1 throughout",
            fontsize=10.5, color=MUTE)
    ax.plot([4, 196], [72.6, 72.6], color=HAIR, lw=1.1)

    ytop, dy = 64.0, 5.6
    xs, xe = 42.0, 158.0

    def qy(q):
        return ytop - q * dy

    # --- light cone (drawn first, behind everything) ------------------------
    cols = light_cone_frontier()
    col_x = np.linspace(48.5, 154.0, len(cols))
    pts = [(col_x[0] - 3.2, qy(0) + 2.6), (col_x[0] - 3.2, qy(1) - 2.6)]
    for (c, qmax), x in zip(cols, col_x):
        pts.append((x + 2.6, qy(qmax) - 2.6))
    pts.append((col_x[-1] + 2.6, qy(0) + 2.6))
    ax.add_patch(Polygon(pts, closed=True, fc=CONE, ec=CONE, alpha=0.13, lw=1.1,
                         ls=(0, (4, 3)), zorder=1))
    reach = next(i for i, (_, qm) in enumerate(cols) if qm == N - 1)
    ax.plot([col_x[reach] + 2.6, col_x[reach] + 2.6], [qy(9) - 3.4, qy(9) - 6.4],
            color=CONE, lw=1.2)
    ax.text(col_x[reach] + 2.6, qy(9) - 8.2, "light cone reaches q9\nat layer 5",
            fontsize=8.0, color="#8A5F16", ha="center", va="top", fontweight="bold")

    # --- wires ---------------------------------------------------------------
    for q in range(N):
        ax.plot([xs, xe], [qy(q), qy(q)], color="#8A959B", lw=1.0, zorder=2)
        ax.text(xs - 1.5, qy(q), f"q{q}", fontsize=9, color=INK, ha="right",
                va="center", fontweight="bold")

    # --- state preparation ---------------------------------------------------
    ax.add_patch(FancyBboxPatch((6, qy(9) - 3.6), 26, (N - 1) * dy + 7.2,
                                boxstyle="round,pad=0,rounding_size=0.9",
                                fc=NEUT_BG, ec=NEUT, lw=1.6, zorder=3))
    ax.text(19, qy(3) + 1.0, "STATE PREPARATION", fontsize=10.2, fontweight="bold",
            color=NEUT, ha="center", va="center", zorder=4)
    ax.text(19, qy(5) + 0.4, "|ψ₀⟩ = Σⱼ xⱼ |j⟩", fontsize=10.5, color=INK,
            ha="center", va="center", zorder=4)
    ax.text(19, qy(6) + 0.6, "1024 features → 1024 amplitudes", fontsize=8.0,
            color=MUTE, ha="center", va="center", zorder=4)
    ax.text(19, qy(7) + 0.8, "‖x‖₂ = 1   (lossless)", fontsize=8.0, color=MUTE,
            ha="center", va="center", zorder=4)

    # --- gates ---------------------------------------------------------------
    for ci, ((_, _), x) in enumerate(zip(cols, col_x)):
        pairs = EVEN if ci % 2 == 0 else ODD
        for (a, b) in pairs:
            ya, yb = qy(a), qy(b)
            ax.add_patch(FancyBboxPatch((x - 2.6, min(ya, yb) - 1.7), 5.2,
                                        abs(ya - yb) + 3.4,
                                        boxstyle="round,pad=0,rounding_size=0.5",
                                        fc="#FFFFFF", ec=PLAS, lw=1.15, zorder=4))
            ax.add_patch(FancyBboxPatch((x - 2.6, min(ya, yb) - 1.7), 5.2,
                                        abs(ya - yb) + 3.4,
                                        boxstyle="round,pad=0,rounding_size=0.5",
                                        fc=PLAS_BG, ec="none", alpha=0.85, zorder=4))
            ax.text(x, (ya + yb) / 2, "V", fontsize=8.2, color=PLAS, ha="center",
                    va="center", fontweight="bold", zorder=5)

    # layer brackets
    for layer in range(DEPTH):
        x0 = col_x[2 * layer] - 3.4
        x1 = col_x[min(2 * layer + 1, len(col_x) - 1)] + 3.4
        ax.plot([x0, x1], [qy(0) + 4.6, qy(0) + 4.6], color=PLAS, lw=1.0)
        ax.plot([x0, x0], [qy(0) + 4.0, qy(0) + 4.6], color=PLAS, lw=1.0)
        ax.plot([x1, x1], [qy(0) + 4.0, qy(0) + 4.6], color=PLAS, lw=1.0)
        ax.text((x0 + x1) / 2, qy(0) + 6.2, f"L{layer + 1}", fontsize=8.0, color=PLAS,
                ha="center", va="center", fontweight="bold")

    # --- measurement ---------------------------------------------------------
    mx = 164
    ax.add_patch(FancyBboxPatch((mx, qy(9) - 4.0), 32, 12.5,
                                boxstyle="round,pad=0,rounding_size=0.9",
                                fc="#FFFFFF", ec=OK, lw=1.6, zorder=4))
    ax.text(mx + 16, qy(9) + 6.0, "MEASUREMENT", fontsize=9.6, fontweight="bold",
            color=OK, ha="center", zorder=5)
    ax.text(mx + 16, qy(9) + 2.4, "⟨Z₉⟩ ∈ [−1, 1]", fontsize=10.0, color=INK,
            ha="center", zorder=5)
    ax.text(mx + 16, qy(9) - 1.4, "r = σ(β⟨Z₉⟩ + b) ∈ [0,1]", fontsize=8.6,
            color=MUTE, ha="center", zorder=5)
    ax.add_patch(FancyArrowPatch((xe, qy(9)), (mx, qy(9)), arrowstyle="-|>",
                                 mutation_scale=14, color=OK, lw=1.6, zorder=5))
    for q in range(N - 1):
        ax.plot([xe, xe + 3], [qy(q), qy(q)], color="#C8D0D5", lw=1.0, zorder=2)
    ax.text(xe + 6, qy(4), "traced out", fontsize=8.0, color=MUTE, rotation=90,
            ha="center", va="center")

    # --- footer strip --------------------------------------------------------
    ax.plot([4, 196], [-2.5, -2.5], color=HAIR, lw=1.0)
    facts = [
        (4, "Parameters", "8 × 9 × 15 = 1,080 angles, all on a torus"),
        (56, "Why depth ≥ 5", "below the light cone a gradient is exactly 0"),
        (110, "Why a local readout", "a global observable costs 0.092 MRR"),
        (158, "Verified against", "PennyLane · fidelity 1.000"),
    ]
    for x, title, body in facts:
        ax.text(x, -5.6, title, fontsize=8.6, color=PLAS, fontweight="bold")
        ax.text(x, -9.0, body, fontsize=8.2, color=MUTE)

    fig.tight_layout()
    OUT.mkdir(exist_ok=True)
    p = OUT / "CIRCUIT_main.png"
    fig.savefig(p, dpi=300, bbox_inches="tight", facecolor=PAPER)
    fig.savefig(OUT / "CIRCUIT_main.svg", bbox_inches="tight", facecolor=PAPER)
    plt.close(fig)
    return p


# ---------------------------------------------------------------- gate anatomy
def circuit_gate() -> Path:
    fig, ax = plt.subplots(figsize=(13.5, 6.4))
    ax.set_xlim(0, 150)
    ax.set_ylim(0, 72)
    ax.axis("off")

    ax.text(4, 66, "Inside one gate: the general SU(4)", fontsize=17,
            fontweight="bold", color=INK)
    ax.text(4, 61.8, "15 real parameters — the full su(4) Lie algebra, not a restricted "
                     "hardware-efficient block", fontsize=10, color=MUTE)
    ax.plot([4, 146], [59.2, 59.2], color=HAIR, lw=1.1)

    # the generator grid
    labels = ["I", "X", "Y", "Z"]
    x0, y0, cell = 8.0, 16.0, 8.4
    ax.text(x0 + 2 * cell, y0 + 4.6 * cell, "the 15 generators  σ_α ⊗ σ_β",
            fontsize=10.2, fontweight="bold", color=PLAS, ha="center")
    for i, a in enumerate(labels):
        ax.text(x0 - 3.4, y0 + (3 - i) * cell + cell / 2, a, fontsize=9.6,
                color=MUTE, ha="center", va="center", fontweight="bold")
        ax.text(x0 + i * cell + cell / 2, y0 + 4 * cell + 2.6, a, fontsize=9.6,
                color=MUTE, ha="center", va="center", fontweight="bold")
    k = 1
    for i, a in enumerate(labels):
        for j, b in enumerate(labels):
            x = x0 + j * cell
            y = y0 + (3 - i) * cell
            skip = (i == 0 and j == 0)
            ax.add_patch(FancyBboxPatch((x, y), cell - 1.1, cell - 1.1,
                                        boxstyle="round,pad=0,rounding_size=0.5",
                                        fc="#F4F6F7" if skip else PLAS_SOFT,
                                        ec=HAIR if skip else PLAS, lw=1.0))
            if skip:
                ax.text(x + cell / 2 - 0.55, y + cell / 2 - 0.55, "—", fontsize=10,
                        color=MUTE, ha="center", va="center")
            else:
                ax.text(x + cell / 2 - 0.55, y + cell / 2 - 0.05, f"{a}⊗{b}",
                        fontsize=8.8, color=INK, ha="center", va="center")
                ax.text(x + cell / 2 - 0.55, y + cell / 2 - 3.0, f"θ{k}",
                        fontsize=7.4, color=PLAS, ha="center", va="center")
                k += 1
    ax.text(x0 + 2 * cell, y0 - 6.0, "identity term dropped — it is only a global phase",
            fontsize=8.4, color=MUTE, ha="center", style="italic")

    # the maths
    tx = 58
    ax.text(tx, 52, "V(θ)  =  exp( −(i/2) · Σₖ θₖ Gₖ )", fontsize=14, color=INK,
            fontweight="bold")
    ax.text(tx, 46.5, "each Gₖ is one of the 15 Pauli products above, and Gₖ² = I",
            fontsize=9.4, color=MUTE)

    rows = [
        ("Reach", "any two-qubit unitary is reachable — so shallow depth suffices"),
        ("Unitarity", "V†V = I exactly, therefore ‖ψ‖ = 1 after every gate"),
        ("Geometry", "θ is periodic: the parameter space is a torus, not ℝ¹⁵"),
        ("Consequence", "weights cannot inflate, so learning capacity cannot decay"),
    ]
    for i, (a, b) in enumerate(rows):
        y = 39 - i * 5.4
        ax.text(tx, y, a, fontsize=9.6, color=PLAS, fontweight="bold")
        ax.text(tx + 24, y, b, fontsize=9.2, color=INK)

    ax.add_patch(FancyBboxPatch((tx, 3.4), 88, 12.0,
                                boxstyle="round,pad=0,rounding_size=0.8",
                                fc=NEUT_BG, ec=NEUT, lw=1.2))
    ax.text(tx + 3, 13.2, "Two gate forms are implemented", fontsize=9.4,
            fontweight="bold", color=NEUT)
    ax.text(tx + 3, 9.6, "exp  —  the sum form above; the reference paper's ansatz.",
            fontsize=8.4, color=INK)
    ax.text(tx + 3, 6.4, "product  —  Πₖ exp(−i θₖGₖ/2), where the parameter-shift rule is "
                         "exact (8.6e-9).", fontsize=8.4, color=INK)

    fig.tight_layout()
    p = OUT / "CIRCUIT_gate.png"
    fig.savefig(p, dpi=300, bbox_inches="tight", facecolor=PAPER)
    plt.close(fig)
    return p


# ------------------------------------------------------- authentic pennylane
def circuit_pennylane(layers_shown: int = 2) -> Path:
    """Render the real model with PennyLane, so the schematic cannot drift from the code."""
    import pennylane as qml
    import torch

    from qcs.config import Config
    from qcs.quantum.circuit import QRE

    cfg = Config.load()
    model = QRE(cfg)
    with torch.no_grad():
        U = model.gates().detach().numpy()

    rng = np.random.default_rng(0)
    x = rng.normal(size=2**N).astype(np.float64)
    x /= np.linalg.norm(x)

    dev = qml.device("default.qubit", wires=N)

    @qml.qnode(dev)
    def circuit():
        qml.StatePrep(x, wires=range(N))
        for layer in range(layers_shown):
            for gi, (q0, q1) in enumerate(model.pairs):
                qml.QubitUnitary(U[layer, gi], wires=[q0, q1])
        return qml.expval(qml.PauliZ(N - 1))

    fig, ax = qml.draw_mpl(circuit, style="pennylane", decimals=None,
                           fontsize=11)()
    fig.set_size_inches(15.5, 7.0)
    ax.set_title(f"Rendered by PennyLane directly from the model object — "
                 f"state preparation and layers 1–{layers_shown} of {DEPTH}",
                 fontsize=11.5, color=INK, pad=16)
    p = OUT / "CIRCUIT_pennylane.png"
    fig.savefig(p, dpi=220, bbox_inches="tight", facecolor=PAPER)
    plt.close(fig)
    return p


if __name__ == "__main__":
    print("wrote", circuit_main())
    print("wrote", circuit_gate())
    try:
        print("wrote", circuit_pennylane())
    except Exception as exc:  # noqa: BLE001
        print("pennylane render skipped:", exc)
