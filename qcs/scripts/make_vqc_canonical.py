"""The project's variational circuit drawn in canonical quantum-circuit notation.

Everything a physicist expects to see: |0⟩ kets, the state-preparation block, the
brickwall entangling layers with barriers, a measurement meter, the classical wire out —
plus two insets that answer "where does the entanglement actually come from" and
"how much of it is there", the second measured from the trained model.

    figs/CIRCUIT_vqc_canonical.png

Run:  python scripts/make_vqc_canonical.py
"""

from __future__ import annotations

from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
import torch  # noqa: E402
from matplotlib.patches import Arc, FancyArrowPatch, FancyBboxPatch  # noqa: E402

OUT = Path("figures") / "circuit"

INK = "#11161A"
MUTE = "#6B767E"
HAIR = "#D9E0E4"
PAPER = "#FFFFFF"
PLAS = "#A32467"
PLAS_BG = "#FBEFF4"
NEUT = "#41535C"
NEUT_BG = "#F1F4F5"
OK = "#2C7A58"
WIRE = "#7C878E"

plt.rcParams.update({"font.family": "DejaVu Sans", "figure.facecolor": PAPER,
                     "savefig.facecolor": PAPER})

N = 10
DEPTH = 8
EVEN = [(0, 1), (2, 3), (4, 5), (6, 7), (8, 9)]
ODD = [(1, 2), (3, 4), (5, 6), (7, 8)]


def rbox(ax, x, y, w, h, fc, ec, lw=1.2, z=4, r=0.5):
    ax.add_patch(FancyBboxPatch((x, y), w, h,
                                boxstyle=f"round,pad=0,rounding_size={r}",
                                fc=fc, ec=ec, lw=lw, zorder=z))


def meter(ax, cx, cy, size=4.2):
    """The standard measurement symbol: a box with a dial and a needle."""
    rbox(ax, cx - size / 2, cy - size / 2, size, size, PAPER, OK, lw=1.4, z=6, r=0.3)
    ax.add_patch(Arc((cx, cy - size * 0.18), size * 0.62, size * 0.52,
                     theta1=0, theta2=180, color=OK, lw=1.3, zorder=7))
    ax.add_patch(FancyArrowPatch((cx, cy - size * 0.20),
                                 (cx + size * 0.26, cy + size * 0.20),
                                 arrowstyle="-|>", mutation_scale=7, color=OK,
                                 lw=1.2, zorder=7))


def entanglement_curves():
    """Real entanglement entropy per bipartition, measured on the trained circuit."""
    from qcs.config import Config
    from qcs.quantum.circuit import QRE
    from qcs.quantum.encoding import amplitude_encode, entanglement_entropy, page_entropy
    from qcs.train.dataset import ScoringDataset

    cfg = Config.load("tuned")
    model = QRE(cfg)
    for ck in sorted(Path("runs").glob("*phase6_continual*/checkpoints/quantum_s7.pt")):
        model.load_state_dict(torch.load(ck, weights_only=True))
    ds = ScoringDataset.load(cfg)
    X = np.concatenate([e.X[:6] for e in ds.examples[:8]])[:24]
    with torch.no_grad():
        _, states = model.evolve(amplitude_encode(X, N), keep=True)
        out = {}
        for layer in (0, 1, 2, 4, 8):
            out[layer] = [float(entanglement_entropy(states[layer], c, N).mean())
                          for c in range(1, N)]
    page = [page_entropy(c, N) for c in range(1, N)]
    return out, page


def main() -> Path:
    fig = plt.figure(figsize=(20.5, 12.4))
    ax = fig.add_axes([0, 0, 1, 1])
    ax.set_xlim(0, 256)
    ax.set_ylim(0, 155)
    ax.axis("off")

    # -------------------------------------------------------------- header
    ax.text(6, 149, "The Variational Quantum Circuit", fontsize=23, fontweight="bold",
            color=INK)
    ax.text(6, 144, "N = 10 qubits   ·   L = 8 brickwall layers   ·   9 general SU(4) gates "
                    "per layer   ·   15 angles per gate   ·   |θ| = 1,080",
            fontsize=12, color=MUTE)
    ax.plot([6, 250], [140.5, 140.5], color=HAIR, lw=1.2)

    ytop, dy = 133.0, 7.2

    def qy(q):
        return ytop - q * dy

    # -------------------------------------------------------------- wires
    xs, xe = 20.0, 196.0
    for q in range(N):
        ax.plot([xs, xe], [qy(q), qy(q)], color=WIRE, lw=1.1, zorder=2)
        ax.text(10.5, qy(q), "|0⟩", fontsize=11, color=INK, ha="center", va="center",
                zorder=4)
        ax.text(16.5, qy(q), f"q{q}", fontsize=9, color=MUTE, ha="right", va="center",
                zorder=4)

    # -------------------------------------------------- state preparation
    rbox(ax, 22, qy(9) - 3.9, 24, (N - 1) * dy + 7.8, NEUT_BG, NEUT, lw=1.6, z=3, r=0.8)
    ax.text(34, qy(3.4), "AMPLITUDE\nENCODING", fontsize=11, fontweight="bold",
            color=NEUT, ha="center", va="center", zorder=5)
    ax.text(34, qy(5.4), "|ψ₀⟩ = Σⱼ xⱼ|j⟩", fontsize=11.5, color=INK, ha="center",
            va="center", zorder=5)
    ax.text(34, qy(6.6), "x ∈ ℝ¹⁰²⁴,  ‖x‖₂ = 1", fontsize=8.6, color=MUTE, ha="center",
            va="center", zorder=5)
    ax.text(34, qy(7.6), "1024 amplitudes — lossless", fontsize=8.2, color=MUTE,
            ha="center", va="center", zorder=5)

    # -------------------------------------------------------------- gates
    cols = 2 * DEPTH
    col_x = np.linspace(54.0, 190.0, cols)
    gate_w = 5.6
    for ci, x in enumerate(col_x):
        pairs = EVEN if ci % 2 == 0 else ODD
        for gi, (a, b) in enumerate(pairs):
            ya, yb = qy(a), qy(b)
            # the vertical connector that marks a genuine two-qubit gate
            ax.plot([x, x], [ya, yb], color=PLAS, lw=1.0, zorder=3)
            rbox(ax, x - gate_w / 2, min(ya, yb) - 2.0, gate_w, abs(ya - yb) + 4.0,
                 PLAS_BG, PLAS, lw=1.15, z=4, r=0.45)
            label = "V(θ)" if ci == 0 else "V"
            ax.text(x, (ya + yb) / 2, label, fontsize=7.6 if ci == 0 else 8.0,
                    color=PLAS, ha="center", va="center", fontweight="bold", zorder=5)

        # barrier between layers, as a circuit drawing convention
        if ci % 2 == 1 and ci < cols - 1:
            bx = (col_x[ci] + col_x[ci + 1]) / 2
            ax.plot([bx, bx], [qy(9) - 4.6, qy(0) + 4.6], color="#C3CCD1", lw=1.0,
                    ls=(0, (2, 3)), zorder=2)

    for layer in range(DEPTH):
        x0, x1 = col_x[2 * layer] - gate_w / 2, col_x[2 * layer + 1] + gate_w / 2
        ax.plot([x0, x1], [qy(0) + 5.6, qy(0) + 5.6], color=PLAS, lw=1.1, zorder=4)
        ax.plot([x0, x0], [qy(0) + 4.9, qy(0) + 5.6], color=PLAS, lw=1.1, zorder=4)
        ax.plot([x1, x1], [qy(0) + 4.9, qy(0) + 5.6], color=PLAS, lw=1.1, zorder=4)
        ax.text((x0 + x1) / 2, qy(0) + 7.6, f"layer {layer + 1}", fontsize=8.4,
                color=PLAS, ha="center", va="center", fontweight="bold", zorder=4)
    ax.text(col_x[0], qy(0) + 3.2, "even", fontsize=7.4, color=MUTE, ha="center", zorder=4)
    ax.text(col_x[1], qy(0) + 3.2, "odd", fontsize=7.4, color=MUTE, ha="center", zorder=4)

    ax.text(128, qy(9) - 6.2,
            "U(θ) = Π_{ℓ=1..8} [ Π_{odd} V(θ_{ℓ,ab}) · Π_{even} V(θ_{ℓ,ab}) ]"
            "          every V is unitary  ⇒  ‖ψ‖ = 1 at every step",
            fontsize=10.5, color=PLAS, ha="center", fontweight="bold", zorder=4)

    # -------------------------------------------------------- measurement
    meter(ax, 202, qy(9), 5.4)
    ax.plot([xe, 199.3], [qy(9), qy(9)], color=WIRE, lw=1.1, zorder=2)
    for q in range(N - 1):
        ax.plot([xe, xe + 3.0], [qy(q), qy(q)], color="#CBD3D7", lw=1.0, zorder=2)
    ax.text(xe + 5.5, qy(4.5), "traced out", fontsize=8.4, color=MUTE, rotation=90,
            ha="center", va="center", zorder=4)

    # classical double wire out of the meter
    for off in (-0.55, 0.55):
        ax.plot([204.8, 218], [qy(9) + off, qy(9) + off], color=OK, lw=1.0, zorder=3)
    ax.add_patch(FancyArrowPatch((218, qy(9)), (221, qy(9)), arrowstyle="-|>",
                                 mutation_scale=11, color=OK, lw=1.2, zorder=4))
    rbox(ax, 222, qy(9) - 5.0, 30, 10.0, PAPER, OK, lw=1.4, z=4, r=0.6)
    ax.text(237, qy(9) + 1.8, "⟨Z₉⟩ ∈ [−1, 1]", fontsize=10, color=INK, ha="center",
            zorder=5)
    ax.text(237, qy(9) - 2.2, "r = σ(β⟨Z₉⟩ + b)", fontsize=9, color=OK, ha="center",
            fontweight="bold", zorder=5)
    ax.text(237, qy(9) + 7.0, "one relevance score", fontsize=8.6, color=MUTE,
            ha="center", zorder=4)

    # ============================================================ inset A
    ax.plot([6, 250], [58, 58], color=HAIR, lw=1.1)
    ax.text(6, 53.5, "Where the entanglement comes from", fontsize=13,
            fontweight="bold", color=INK)
    ax.text(6, 49.5, "One V(θ) is a GENERAL two-qubit gate. On hardware a compiler realises it "
                     "with at most three CNOTs plus single-qubit rotations",
            fontsize=9.6, color=MUTE)

    ay0, ady = 36.0, 9.0
    axs, axe = 22.0, 116.0
    for i in range(2):
        y = ay0 - i * ady
        ax.plot([axs, axe], [y, y], color=WIRE, lw=1.1, zorder=2)
        ax.text(18.5, y, f"q{'a' if i == 0 else 'b'}", fontsize=9, color=MUTE,
                ha="right", va="center", zorder=4)

    def sq(x, row, label, w=7.0):
        y = ay0 - row * ady
        rbox(ax, x - w / 2, y - 3.0, w, 6.0, PAPER, PLAS, lw=1.1, z=4, r=0.4)
        ax.text(x, y, label, fontsize=7.6, color=PLAS, ha="center", va="center",
                fontweight="bold", zorder=5)

    def cnot(x, ctrl, targ):
        yc, yt = ay0 - ctrl * ady, ay0 - targ * ady
        ax.plot([x, x], [yc, yt], color=INK, lw=1.2, zorder=3)
        ax.add_patch(plt.Circle((x, yc), 0.9, color=INK, zorder=5))
        ax.add_patch(plt.Circle((x, yt), 2.1, fc=PAPER, ec=INK, lw=1.2, zorder=5))
        ax.plot([x - 2.1, x + 2.1], [yt, yt], color=INK, lw=1.1, zorder=6)
        ax.plot([x, x], [yt - 2.1, yt + 2.1], color=INK, lw=1.1, zorder=6)

    sq(30, 0, "B₁"); sq(30, 1, "B₂")
    cnot(42, 0, 1)
    sq(54, 0, "Rz(α)"); sq(54, 1, "Ry(β)")
    cnot(66, 1, 0)
    sq(78, 1, "Ry(γ)")
    cnot(90, 0, 1)
    sq(104, 0, "A₁"); sq(104, 1, "A₂")

    ax.text(69, 20.0, "3 CNOTs  +  15 single-qubit angles  =  one general SU(4)",
            fontsize=10, color=PLAS, ha="center", fontweight="bold")
    ax.text(69, 15.5, "The CNOTs are the entangling elements. In simulation we apply the "
                      "4×4 unitary directly;", fontsize=8.8, color=MUTE, ha="center")
    ax.text(69, 11.8, "this is the equivalent form a hardware compiler emits "
                      "(Vatan–Williams decomposition).", fontsize=8.8, color=MUTE,
            ha="center")
    ax.text(69, 6.5, "V(θ) = exp( −(i/2) Σₖ θₖ σ_α⊗σ_β )   —   the 15 non-identity Pauli products",
            fontsize=9.2, color=INK, ha="center", style="italic")

    # ============================================================ inset B
    ax.plot([128, 128], [6, 52], color=HAIR, lw=1.0)
    ax.text(136, 53.5, "How much entanglement is actually produced", fontsize=13,
            fontweight="bold", color=INK)
    ax.text(136, 49.5, "Von Neumann entropy across every bipartition, measured on the trained "
                       "circuit with real data", fontsize=9.6, color=MUTE)
    ax.text(136, 46.0, "Amplitude-encoded features arrive already close to the Page value — the "
                       "circuit sustains that entanglement while reshaping the state.",
            fontsize=9.0, color=PLAS, style="italic")

    try:
        ent, page = entanglement_curves()
        sub = fig.add_axes([0.545, 0.055, 0.42, 0.235])
        cuts = list(range(1, N))
        for i, (layer, vals) in enumerate(sorted(ent.items())):
            sub.plot(cuts, vals, marker="o", ms=4, lw=1.5,
                     color=plt.cm.plasma(i / max(len(ent) - 1, 1)),
                     label=f"after layer {layer}" if layer else "input |ψ₀⟩")
        sub.plot(cuts, page, ls="--", lw=1.1, color=MUTE, label="Page value (maximum)")
        sub.set_xlabel("bipartition cut  (qubits 0…k | k…9)", fontsize=9, color=MUTE)
        sub.set_ylabel("S(ρ_A)   [bits]", fontsize=9, color=MUTE)
        sub.tick_params(labelsize=8, colors=MUTE)
        for spine in ("top", "right"):
            sub.spines[spine].set_visible(False)
        for spine in ("left", "bottom"):
            sub.spines[spine].set_color(MUTE)
        sub.grid(alpha=.18, color=MUTE, lw=.5)
        sub.legend(fontsize=7.4, frameon=False, ncol=2, labelcolor=MUTE)
    except Exception as exc:  # noqa: BLE001
        ax.text(190, 30, f"[entanglement data unavailable: {exc}]", fontsize=9,
                color=MUTE, ha="center")

    OUT.mkdir(exist_ok=True)
    p = OUT / "CIRCUIT_vqc_canonical.png"
    fig.savefig(p, dpi=300, bbox_inches="tight", facecolor=PAPER)
    fig.savefig(OUT / "CIRCUIT_vqc_canonical.svg", bbox_inches="tight", facecolor=PAPER)
    plt.close(fig)
    return p


if __name__ == "__main__":
    print("wrote", main())
