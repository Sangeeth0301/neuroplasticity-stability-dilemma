"""Phase 7 figures: V41 budget composition, V42 selection waterfall,
V43 quality vs budget, V44 attention dilution, V45 latency.
"""

from __future__ import annotations

from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np

from . import style
from .style import ARM_COLOR, C_MUTE, C_OK, C_PLAS, C_STAB, C_WARN


def v41_budget_composition(rows: list[dict], budget_total: int, out_dir: Path,
                           prov: str = "") -> Path:
    """Where the tokens went, prompt by prompt.

    PASS: the ceiling is never crossed and the reserved blocks are always present.
    FAIL: any bar above the line ⇒ a tokenizer mismatch between counting and rendering.
    """
    style.apply_style()
    fig, ax = plt.subplots(1, 2, figsize=(10, 3.2))
    rows = rows[:60]
    sel = np.array([r["tokens_selected"] for r in rows])
    other = np.array([r["tokens_total"] - r["tokens_selected"] for r in rows])
    x = np.arange(len(rows))
    ax[0].bar(x, sel, color=C_PLAS, label="selected memory")
    ax[0].bar(x, other, bottom=sel, color=C_STAB, label="system + recent + query")
    ax[0].axhline(budget_total, ls="--", color=C_WARN, label="hard ceiling")
    style.panel(ax[0], "V41 · token spend per prompt", "probe", "tokens")
    ax[0].legend()

    ax[1].hist([r["n_selected"] for r in rows], bins=20, color=C_OK)
    style.panel(ax[1], "items selected per prompt", "items", "prompts")
    style.note(ax[1], f"max tokens seen: {max(r['tokens_total'] for r in rows)}\n"
                      f"ceiling: {budget_total}")

    fig.tight_layout()
    return style.save(fig, out_dir, "V41", "budget_composition", prov)


def v42_selection_waterfall(trace: list[dict], gold_idx: set[int], out_dir: Path,
                            prov: str = "") -> Path:
    """Why each item was chosen, in order — the best single debugging view in the system."""
    style.apply_style()
    fig, ax = plt.subplots(1, 2, figsize=(10, 3.4))
    ranks = [t["rank"] for t in trace]
    gains = [t["gain_per_token"] for t in trace]
    colors = [C_PLAS if t["idx"] in gold_idx else C_STAB for t in trace]
    ax[0].bar(ranks, gains, color=colors)
    style.panel(ax[0], "V42 · marginal gain per token, in selection order", "selection rank",
                "gain / token")
    style.note(ax[0], "magenta = a required fact\nteal = everything else")

    ax[1].scatter([t["tokens"] for t in trace], [t["score"] for t in trace],
                  c=colors, s=26)
    style.panel(ax[1], "score vs token cost", "tokens", "relevance score")
    fig.tight_layout()
    return style.save(fig, out_dir, "V42", "selection_waterfall", prov)


def v43_v44_saturation(curves: dict[str, dict[int, float]], dilution: dict[str, tuple],
                       out_dir: Path, prov: str = "") -> Path:
    """Quality vs token budget, and quality vs number of items at fixed budget.

    PASS: a learned arm reaches the baseline's best quality at a materially smaller
    budget; the dilution curve has an interior optimum — that IS context saturation,
    measured rather than asserted.
    """
    style.apply_style()
    fig, ax = plt.subplots(1, 2, figsize=(10, 3.4))

    for arm, curve in curves.items():
        bs = sorted(curve)
        ys = [curve[b] for b in bs]
        ax[0].plot(bs, ys, marker="o", ms=4,
                   color=ARM_COLOR.get(arm.split("_")[0], C_MUTE), label=arm)
        peak = bs[int(np.argmax(ys))]
        ax[0].axvline(peak, color=ARM_COLOR.get(arm.split("_")[0], C_MUTE), ls=":", lw=.6)
    ax[0].set_xscale("log", base=2)
    style.panel(ax[0], "V43 · answer accuracy vs token budget", "token budget B", "accuracy")
    ax[0].set_ylim(0, 1.02)
    ax[0].legend()

    for arm, (ns, accs) in dilution.items():
        ax[1].plot(ns, accs, marker="o", ms=3,
                   color=ARM_COLOR.get(arm.split("_")[0], C_MUTE), label=arm)
    style.panel(ax[1], "V44 · accuracy vs items in context (fixed B)", "items selected",
                "accuracy")
    ax[1].set_ylim(0, 1.02)
    ax[1].legend()

    fig.tight_layout()
    return style.save(fig, out_dir, "V43", "saturation", prov)


def v45_latency(stages: dict[str, list[float]], out_dir: Path, prov: str = "") -> Path:
    """Where the wall-clock goes per probe.

    PASS: circuit scoring under ~200 ms for the candidate set.
    FAIL: if it dominates, batch the statevector preparation before optimising anything else.
    """
    style.apply_style()
    names = list(stages)
    p50 = [float(np.percentile(stages[k], 50)) for k in names]
    p95 = [float(np.percentile(stages[k], 95)) for k in names]
    fig, ax = plt.subplots(1, 2, figsize=(10, 3.2))
    w = 0.38
    xs = np.arange(len(names))
    ax[0].bar(xs - w / 2, p50, w, color=C_STAB, label="p50")
    ax[0].bar(xs + w / 2, p95, w, color=C_PLAS, label="p95")
    ax[0].set_xticks(xs)
    ax[0].set_xticklabels(names, fontsize=6.5, rotation=20, ha="right")
    style.panel(ax[0], "V45 · latency by stage", "", "ms")
    ax[0].legend()

    total = np.sum([np.mean(stages[k]) for k in names])
    shares = [np.mean(stages[k]) / total for k in names]
    ax[1].barh(names, shares, color=C_OK)
    style.panel(ax[1], "share of per-probe time", "fraction", "")
    style.note(ax[1], f"mean total {total:.0f} ms")

    fig.tight_layout()
    return style.save(fig, out_dir, "V45", "latency_breakdown", prov)
