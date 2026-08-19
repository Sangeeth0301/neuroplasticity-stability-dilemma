"""The single source of truth for how every figure in this project looks.

Colour is fixed by ROLE and never re-mapped:
    classical / stability -> teal      quantum / plasticity -> magenta
    oracle / bound        -> grey dash warning / attention  -> amber
Figures are saved with transparent backgrounds so they read on light and dark pages.
"""

from __future__ import annotations

from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402

C_STAB = "#2E93B5"
C_PLAS = "#C7548E"
C_OK = "#5FB3A1"
C_WARN = "#D9A441"
C_MUTE = "#8A949B"
C_ORACLE = "#8A949B"
CYCLE = [C_STAB, C_PLAS, C_OK, C_WARN, "#8E7CC3", "#C98B6B"]

ARM_COLOR = {
    "quantum": C_PLAS,
    "qre": C_PLAS,
    "classical": C_STAB,
    "mlp": C_STAB,
    "cosine": C_OK,
    "oracle": C_ORACLE,
    "random": C_WARN,
}


def apply_style() -> None:
    plt.rcParams.update({
        "figure.facecolor": "none", "axes.facecolor": "none", "savefig.facecolor": "none",
        "text.color": C_MUTE, "axes.labelcolor": C_MUTE, "axes.edgecolor": C_MUTE,
        "xtick.color": C_MUTE, "ytick.color": C_MUTE,
        "grid.color": C_MUTE, "grid.alpha": 0.18, "grid.linewidth": 0.5,
        "font.size": 8, "axes.titlesize": 9, "axes.titleweight": "bold",
        "legend.frameon": False, "legend.fontsize": 7,
        "axes.spines.top": False, "axes.spines.right": False,
        "axes.prop_cycle": matplotlib.cycler(color=CYCLE),
        "figure.dpi": 110,
    })


def save(fig, out_dir: str | Path, fig_id: str, name: str, provenance: str = "") -> Path:
    """Save as PNG + SVG under a stable `Vxx_name` filename, stamped with provenance."""
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    if provenance:
        fig.text(0.995, 0.002, provenance, ha="right", va="bottom",
                 fontsize=5.5, color=C_MUTE, alpha=0.8)
    stem = out_dir / f"{fig_id}_{name}"
    fig.savefig(f"{stem}.png", dpi=130, bbox_inches="tight", transparent=True)
    fig.savefig(f"{stem}.svg", bbox_inches="tight", transparent=True)
    plt.close(fig)
    return Path(f"{stem}.png")


def panel(ax, title: str, xlabel: str = "", ylabel: str = "") -> None:
    ax.set_title(title)
    if xlabel:
        ax.set_xlabel(xlabel)
    if ylabel:
        ax.set_ylabel(ylabel)
    ax.grid(True, axis="both")
    ax.set_axisbelow(True)


def note(ax, text: str, x: float = 0.02, y: float = 0.94) -> None:
    ax.text(x, y, text, transform=ax.transAxes, fontsize=6.5, color=C_MUTE,
            va="top", ha="left")
