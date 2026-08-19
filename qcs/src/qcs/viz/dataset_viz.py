"""Phase 1 figures: V01 thread structure, V02 probe distances, V03 ownership,
V04 corpus statistics.

Each function states the pass/fail read in its docstring — a plot you do not know how
to fail is decoration.
"""

from __future__ import annotations

from collections import Counter
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np

from . import style
from .style import C_MUTE, C_OK, C_PLAS, C_STAB, C_WARN

KIND_COLOR = {
    "fact": C_PLAS, "update": C_WARN, "probe": C_STAB,
    "distractor": C_OK, "filler": C_MUTE,
}


def v01_thread_structure(turns: list[dict], thread_id: str, out_dir: Path, prov: str = "") -> Path:
    """Turn-by-turn map of one thread.

    PASS: facts spread over the early thread, probes trailing them at varied gaps.
    FAIL: probes hugging their facts (task too easy) or facts bunched at the start.
    """
    style.apply_style()
    ts = sorted([t for t in turns if t["thread_id"] == thread_id], key=lambda x: x["turn_idx"])
    users = sorted({t["user_id"] for t in ts})
    row = {u: i for i, u in enumerate(users)}

    fig, ax = plt.subplots(2, 1, figsize=(11, 4.6), height_ratios=[2.2, 1])

    a = ax[0]
    for t in ts:
        c = KIND_COLOR[t["kind"]]
        alpha = 1.0 if t["kind"] in ("fact", "update", "probe") else 0.28
        lw = 1.6 if t["kind"] in ("fact", "update", "probe") else 0.8
        a.vlines(t["turn_idx"], row[t["user_id"]] - 0.36, row[t["user_id"]] + 0.36,
                 color=c, alpha=alpha, lw=lw)
    # arcs from each fact to the probe that needs it
    intro = {t["fact_id"]: t["turn_idx"] for t in ts if t["kind"] in ("fact", "update")}
    for t in ts:
        if t["kind"] != "probe":
            continue
        for fid in t["required_facts"]:
            if fid in intro:
                a.annotate("", xy=(t["turn_idx"], row[t["user_id"]]),
                           xytext=(intro[fid], row[t["user_id"]]),
                           arrowprops=dict(arrowstyle="->", color=C_STAB, alpha=.30, lw=.6,
                                           connectionstyle="arc3,rad=-0.25"))
    a.set_yticks(range(len(users)))
    a.set_yticklabels(users, fontsize=7)
    style.panel(a, f"V01 · thread {thread_id} — turn map by speaker", "", "participant")
    handles = [plt.Line2D([], [], color=c, lw=2, label=k) for k, c in KIND_COLOR.items()]
    a.legend(handles=handles, ncol=5, loc="upper right", fontsize=6.5)

    a = ax[1]
    kinds = [t["kind"] for t in ts]
    win = 25
    for kind, c in KIND_COLOR.items():
        dens = np.convolve([k == kind for k in kinds], np.ones(win) / win, mode="same")
        a.plot(dens, color=c, lw=1, label=kind)
    style.panel(a, "turn-kind density (25-turn window)", "turn index", "share")
    a.legend(ncol=5, fontsize=6.5)

    fig.tight_layout()
    return style.save(fig, out_dir, "V01", f"thread_structure_{thread_id}", prov)


def v02_probe_distances(turns: list[dict], out_dir: Path, bands: list[int], prov: str = "") -> Path:
    """Fact-to-probe distance distribution.

    PASS: mass at every configured band, including the longest.
    FAIL: the long tail missing — then there is no long-horizon memory test at all.
    """
    style.apply_style()
    d = [t["probe_distance"] for t in turns if t["kind"] == "probe"]
    fig, ax = plt.subplots(1, 2, figsize=(11, 3.2))

    a = ax[0]
    a.hist(d, bins=60, color=C_PLAS, alpha=.85)
    for b in bands:
        a.axvline(b, color=C_MUTE, ls=":", lw=.8)
        a.text(b, a.get_ylim()[1] * .92, f" {b}", fontsize=6, color=C_MUTE)
    style.panel(a, "V02 · fact → probe distance", "turns between fact and probe", "probes")
    style.note(a, f"n={len(d)}  mean={np.mean(d):.0f}  max={np.max(d)}")

    a = ax[1]
    xs = np.sort(d)
    a.plot(xs, np.arange(1, len(xs) + 1) / len(xs), color=C_STAB, lw=1.4)
    style.panel(a, "cumulative distribution", "distance", "fraction of probes")
    a.set_ylim(0, 1)

    fig.tight_layout()
    return style.save(fig, out_dir, "V02", "probe_distance_hist", prov)


def v03_fact_ownership(facts: list[dict], out_dir: Path, prov: str = "") -> Path:
    """Who owns what, and how supersession chains are distributed.

    PASS: contradictions present in the adversarial regime; ownership spread in
    multi-owner threads and concentrated in single-owner ones.
    FAIL: no supersessions -> the stale-fact test is vacuous.
    """
    style.apply_style()
    fig, ax = plt.subplots(1, 3, figsize=(11, 3.2))

    a = ax[0]
    per_regime: dict[str, Counter] = {}
    for f in facts:
        per_regime.setdefault(f["regime"], Counter())[f["owner_user"]] += 1
    regimes = sorted(per_regime)
    users = sorted({u for c in per_regime.values() for u in c})
    bottom = np.zeros(len(regimes))
    for i, u in enumerate(users):
        vals = np.array([per_regime[r].get(u, 0) for r in regimes], dtype=float)
        a.bar(regimes, vals, bottom=bottom, label=u,
              color=style.CYCLE[i % len(style.CYCLE)])
        bottom += vals
    style.panel(a, "V03 · fact ownership by regime", "", "facts")
    a.legend(ncol=2, fontsize=6)
    a.tick_params(axis="x", labelrotation=12)

    a = ax[1]
    c = Counter(f["attribute"] for f in facts)
    a.barh(list(c), list(c.values()), color=C_STAB)
    style.panel(a, "attribute mix", "facts", "")

    a = ax[2]
    sup = sum(1 for f in facts if f.get("superseded_by"))
    upd = sum(1 for f in facts if f.get("supersedes"))
    a.bar(["stable", "superseded", "updates"], [len(facts) - sup - upd, sup, upd],
          color=[C_OK, C_WARN, C_PLAS])
    style.panel(a, "supersession chains", "", "facts")
    style.note(a, f"{sup} facts overwritten later\n{upd} update turns")

    fig.tight_layout()
    return style.save(fig, out_dir, "V03", "fact_ownership", prov)


def v04_corpus_stats(stats: dict, turns: list[dict], out_dir: Path, prov: str = "") -> Path:
    """Shape of the corpus: turn kinds, lengths, speakers, thread sizes.

    PASS: turn lengths in a single sane band; every speaker active.
    FAIL: bimodal lengths -> the segmenter or templates are misfiring.
    """
    style.apply_style()
    fig, ax = plt.subplots(1, 4, figsize=(12.5, 3.0))

    a = ax[0]
    kinds = stats["train"]["kinds"]
    a.bar(list(kinds), list(kinds.values()),
          color=[KIND_COLOR.get(k, C_MUTE) for k in kinds])
    style.panel(a, "V04 · turn kinds (train)", "", "turns")
    a.tick_params(axis="x", labelrotation=25)

    a = ax[1]
    lengths = [len(t["text"].split()) for t in turns]
    a.hist(lengths, bins=30, color=C_PLAS)
    style.panel(a, "words per turn", "words", "turns")
    style.note(a, f"mean {np.mean(lengths):.1f}\np95 {np.percentile(lengths, 95):.0f}")

    a = ax[2]
    sp = stats["train"]["speakers"]
    a.bar(list(sp), list(sp.values()), color=C_STAB)
    style.panel(a, "turns per speaker", "", "turns")
    a.tick_params(axis="x", labelrotation=25)

    a = ax[3]
    per_thread = Counter(t["thread_id"] for t in turns)
    a.hist(list(per_thread.values()), bins=12, color=C_OK)
    style.panel(a, "turns per thread", "turns", "threads")
    style.note(a, f"{len(per_thread)} threads\n{stats['n_facts']} facts total")

    fig.tight_layout()
    return style.save(fig, out_dir, "V04", "corpus_stats", prov)
