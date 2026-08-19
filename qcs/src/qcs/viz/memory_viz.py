"""Phase 3 figures: V09 memory growth, V10 ANN recall vs M, V11 consolidation map,
V12 entity DAG + salience separation.
"""

from __future__ import annotations

from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np

from . import style
from .style import C_MUTE, C_OK, C_PLAS, C_STAB, C_WARN


def v09_memory_growth(growth: dict[str, list[int]], turns: list[int],
                      out_dir: Path, prov: str = "") -> Path:
    """Units, summaries and edges as the thread advances.

    PASS: every curve monotone non-decreasing — that IS the append-only property, drawn.
    FAIL: any downward step means something was deleted or overwritten.
    """
    style.apply_style()
    fig, ax = plt.subplots(1, 2, figsize=(9.5, 3.2))
    colors = {"semantic_units": C_STAB, "summaries": C_PLAS, "edges": C_OK}
    for name, ys in growth.items():
        ax[0].plot(turns, ys, color=colors.get(name, C_MUTE), lw=1.4, label=name)
        drops = int(np.sum(np.diff(ys) < 0))
        assert drops == 0, f"{name} decreased {drops}× — append-only violated"
    style.panel(ax[0], "V09 · memory growth (append-only)", "turn index", "rows in ℳ")
    ax[0].legend()
    style.note(ax[0], "no curve may ever decrease")

    ax[1].plot(turns, np.array(growth["semantic_units"]) / np.maximum(np.array(turns), 1),
               color=C_STAB, lw=1.4)
    style.panel(ax[1], "units per turn", "turn index", "ratio")

    fig.tight_layout()
    return style.save(fig, out_dir, "V09", "memory_growth", prov)


def v10_ann_recall(Ms: list[int], recall: list[float], recall_by_band: dict[int, list[float]],
                   target: float, out_dir: Path, prov: str = "") -> Path:
    """Does the prefilter even show the scorer the right fact?

    PASS: recall ≥ target at the configured M — this is the ceiling on everything.
    FAIL: below it, and no scorer can recover; fix retrieval before touching the model.
    """
    style.apply_style()
    fig, ax = plt.subplots(1, 2, figsize=(9.5, 3.2))
    ax[0].plot(Ms, recall, marker="o", ms=4, color=C_PLAS, lw=1.6)
    ax[0].axhline(target, ls="--", color=C_MUTE, label=f"exit criterion {target}")
    style.panel(ax[0], "V10 · ground-truth fact recall vs candidate set size", "M", "recall")
    ax[0].set_ylim(0, 1.02)
    ax[0].legend()

    for i, (band, ys) in enumerate(sorted(recall_by_band.items())):
        ax[1].plot(Ms, ys, marker="o", ms=3, color=style.CYCLE[i % len(style.CYCLE)],
                   label=f"probe distance ≈ {band}")
    ax[1].axhline(target, ls="--", color=C_MUTE)
    style.panel(ax[1], "by fact→probe distance", "M", "recall")
    ax[1].set_ylim(0, 1.02)
    ax[1].legend()

    fig.tight_layout()
    return style.save(fig, out_dir, "V10", "ann_recall_vs_M", prov)


def v11_consolidation(summ: list[dict], intact: bool, out_dir: Path, prov: str = "") -> Path:
    """Summary nodes and their sources.

    PASS: every summary points at ≥1 live source and the sources are byte-identical
    afterwards. FAIL: orphaned summaries ⇒ consolidation destroyed what it summarised.
    """
    style.apply_style()
    fig, ax = plt.subplots(1, 3, figsize=(12, 3.2))

    ax[0].bar(range(len(summ[:40])), [s["n_sources"] for s in summ[:40]], color=C_PLAS)
    style.panel(ax[0], "V11 · sources per summary node", "summary (first 40)", "source units")
    style.note(ax[0], f"sources intact after consolidation: {intact}")

    ax[1].hist([s["n_tokens"] for s in summ], bins=30, color=C_STAB)
    style.panel(ax[1], "summary length", "tokens", "summaries")

    src = np.array([s["n_sources"] for s in summ])
    tok = np.array([s["n_tokens"] for s in summ])
    ax[2].scatter(src, tok, s=10, color=C_OK, alpha=.6)
    style.panel(ax[2], "compression: sources → tokens", "source units", "summary tokens")
    style.note(ax[2], f"{len(summ)} summary nodes\nadded, none replacing a source")

    fig.tight_layout()
    return style.save(fig, out_dir, "V11", "consolidation_map", prov)


def v12_entity_graph(gstats: dict, edge_kinds: dict[str, int], stale: int,
                     salience_sep: float, sal_matrix: np.ndarray, users: list[str],
                     out_dir: Path, prov: str = "") -> Path:
    """The DAG that represents contradiction instead of resolving it, plus salience.

    PASS: supersedes/contradicts edges exist, and different users' salience vectors are
    measurably apart. FAIL: salience separation ≈ 0 ⇒ ρ carries no participant signal and
    persona bleed cannot be reduced.
    """
    style.apply_style()
    fig, ax = plt.subplots(1, 3, figsize=(12, 3.3))

    ax[0].bar(list(edge_kinds), list(edge_kinds.values()),
              color=[C_STAB, C_PLAS, C_WARN, C_OK][: len(edge_kinds)])
    style.panel(ax[0], "V12 · edges in the entity DAG", "", "edges")
    ax[0].tick_params(axis="x", labelrotation=15)
    style.note(ax[0], f"{gstats['n_entities']} entities\n{stale} stale units flagged")

    im = ax[1].imshow(sal_matrix, cmap="PuOr", vmin=-1, vmax=1, aspect="auto")
    ax[1].set_xticks(range(len(users)))
    ax[1].set_xticklabels(users, fontsize=6, rotation=45)
    ax[1].set_yticks(range(len(users)))
    ax[1].set_yticklabels(users, fontsize=6)
    style.panel(ax[1], "salience cosine between participants", "", "")
    plt.colorbar(im, ax=ax[1], fraction=.046)

    ax[2].bar(["separation"], [salience_sep], color=C_PLAS)
    ax[2].axhline(0.05, ls="--", color=C_MUTE, label="minimum useful")
    style.panel(ax[2], "mean cosine distance between users", "", "1 − cos")
    ax[2].legend()
    style.note(ax[2], f"{salience_sep:.3f}")

    fig.tight_layout()
    return style.save(fig, out_dir, "V12", "entity_dag", prov)


def v10b_headroom(h: dict, out_dir: Path, prov: str = "") -> Path:
    """What cosine similarity cannot do — i.e. what the learned scorer must earn.

    PASS (for the project): rank-1 well below 1.0 and a high stale-outranks-current rate,
    meaning there is a real gap only the relational block can close.
    FAIL: cosine already at 1.0 everywhere ⇒ the benchmark is saturated and no scorer,
    quantum or classical, can demonstrate anything.
    """
    style.apply_style()
    fig, ax = plt.subplots(1, 3, figsize=(12, 3.2))

    regs = list(h["rank1_by_regime"])
    vals = [h["rank1_by_regime"][r] for r in regs]
    ax[0].bar(regs, vals, color=[C_STAB, C_OK, C_WARN][: len(regs)])
    ax[0].axhline(h["rank1"], ls="--", color=C_MUTE, label=f"overall {h['rank1']:.3f}")
    style.panel(ax[0], "V10b · cosine rank-1 by regime", "", "rank-1 accuracy")
    ax[0].set_ylim(0, 1.02)
    ax[0].legend()
    ax[0].tick_params(axis="x", labelrotation=12)

    bands = list(h["rank1_by_band"])
    ax[1].plot(bands, [h["rank1_by_band"][b] for b in bands], marker="o", ms=4, color=C_PLAS)
    style.panel(ax[1], "cosine rank-1 vs probe distance", "fact→probe distance band",
                "rank-1 accuracy")
    ax[1].set_ylim(0, 1.02)

    ax[2].bar(["stale outranks\ncurrent", "headroom\n(1 − rank1)"],
              [h["stale_outranks_current"], h["headroom"]], color=[C_WARN, C_PLAS])
    style.panel(ax[2], "what only ρ can fix", "", "rate")
    ax[2].set_ylim(0, 1.02)
    style.note(ax[2], f"{h['supersession_cases']} supersession probes\n"
                      f"{h['n_probes']} probes total")

    fig.tight_layout()
    return style.save(fig, out_dir, "V10b", "cosine_headroom", prov)
