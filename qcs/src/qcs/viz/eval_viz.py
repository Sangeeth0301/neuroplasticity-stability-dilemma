"""Phase 8 figures: V35 ablations, V36 per-regime + persona, V37 stale rate,
V38 noise robustness, V39 failure taxonomy, V40 significance, V46 headline table.
"""

from __future__ import annotations

from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np

from . import style
from .style import ARM_COLOR, C_MUTE, C_OK, C_PLAS, C_STAB, C_WARN


def v35_ablations(base: float, ablations: dict[str, float], out_dir: Path,
                  prov: str = "") -> Path:
    """Every ablation as a delta from the full system, sorted by effect size.

    PASS: components that the architecture argues for actually matter.
    FAIL: everything near zero ⇒ the components are decorative and should be removed.
    """
    style.apply_style()
    names = sorted(ablations, key=lambda k: ablations[k] - base)
    deltas = [ablations[n] - base for n in names]
    fig, ax = plt.subplots(1, 2, figsize=(11, 3.4))
    colors = [C_WARN if d < -0.005 else C_MUTE for d in deltas]
    ax[0].barh(names, deltas, color=colors)
    ax[0].axvline(0, color=C_MUTE, lw=.7)
    style.panel(ax[0], "V35 · ablation effect on MRR (Δ from full system)", "Δ MRR", "")
    for i, d in enumerate(deltas):
        ax[0].text(d, i, f" {d:+.3f}", va="center", fontsize=6.5, color=C_MUTE)

    ax[1].barh(names + ["full system"], [ablations[n] for n in names] + [base],
               color=[C_STAB] * len(names) + [C_PLAS])
    style.panel(ax[1], "absolute MRR", "MRR", "")
    ax[1].set_xlim(0, 1.02)

    fig.tight_layout()
    return style.save(fig, out_dir, "V35", "ablation_grid", prov)


def v36_v37_regimes(by_regime: dict[str, dict[str, float]], stale: dict[str, float],
                    out_dir: Path, prov: str = "") -> Path:
    """Per-regime accuracy and the stale-fact rate — the multi-user and supersession claims."""
    style.apply_style()
    arms = list(by_regime)
    regimes = sorted({r for v in by_regime.values() for r in v})
    fig, ax = plt.subplots(1, 2, figsize=(11, 3.4))

    w = 0.8 / max(len(arms), 1)
    for i, arm in enumerate(arms):
        vals = [by_regime[arm].get(r, np.nan) for r in regimes]
        ax[0].bar(np.arange(len(regimes)) + i * w, vals, w,
                  color=ARM_COLOR.get(arm.split("_")[0], C_MUTE), label=arm)
    ax[0].set_xticks(np.arange(len(regimes)) + w * (len(arms) - 1) / 2)
    ax[0].set_xticklabels(regimes, fontsize=7)
    style.panel(ax[0], "V36 · MRR by regime", "", "MRR")
    ax[0].set_ylim(0, 1.05)
    ax[0].legend(fontsize=6)

    # a split with no supersession probes yields NaN — that is "not measured", not zero
    names = [n for n in stale if not np.isnan(stale[n])]
    if names:
        ax[1].bar(names, [stale[n] for n in names],
                  color=[ARM_COLOR.get(n.split("_")[0], C_MUTE) for n in names])
        ax[1].set_ylim(0, max(max(stale[n] for n in names), 0.1) * 1.2)
        for i, n in enumerate(names):
            ax[1].text(i, stale[n], f"{stale[n]:.3f}", ha="center", va="bottom",
                       fontsize=6.5, color=C_MUTE)
    else:
        style.note(ax[1], "no supersession probes in this split")
    style.panel(ax[1], "V37 · stale fact outranks current  ↓ lower is better", "", "rate")
    ax[1].tick_params(axis="x", labelrotation=25)

    fig.tight_layout()
    return style.save(fig, out_dir, "V36", "regimes_and_stale", prov)


def v38_noise(noise_curve: dict[float, dict[str, float]], out_dir: Path,
              prov: str = "") -> Path:
    """Does the trained circuit survive NISQ-realistic depolarising noise?

    PASS: ranking metrics roughly flat through p = 5e-4 (the level the reference paper
    reports). FAIL: collapse ⇒ the hardware claim is unsupported.
    """
    style.apply_style()
    ps = sorted(noise_curve)
    fig, ax = plt.subplots(1, 2, figsize=(10, 3.4))
    for key, c in (("mrr", C_PLAS), ("rank1", C_STAB), ("auc", C_OK)):
        ax[0].plot(ps, [noise_curve[p][key] for p in ps], marker="o", ms=4, color=c, label=key)
    ax[0].axvline(5e-4, ls=":", color=C_WARN, lw=1)
    ax[0].text(5e-4, 0.05, " paper's level", fontsize=6.5, color=C_WARN)
    ax[0].set_xscale("symlog", linthresh=1e-4)
    style.panel(ax[0], "V38 · ranking quality vs depolarising noise", "p per gate", "metric")
    ax[0].set_ylim(0, 1.05)
    ax[0].legend()

    ax[1].plot(ps, [noise_curve[p]["stale_above_current"] for p in ps], marker="s", ms=4,
               color=C_WARN)
    ax[1].set_xscale("symlog", linthresh=1e-4)
    style.panel(ax[1], "stale-fact error vs noise  ↓ lower is better", "p per gate", "rate")

    fig.tight_layout()
    return style.save(fig, out_dir, "V38", "noise_robustness", prov)


def v39_failures(analyses: dict[str, dict], out_dir: Path, prov: str = "") -> Path:
    """Where the losses actually come from, per arm."""
    style.apply_style()
    arms = list(analyses)
    kinds = ["retrieval", "ranking", "stale", "eviction"]
    colors = {"retrieval": C_STAB, "ranking": C_PLAS, "stale": C_WARN, "eviction": C_OK}
    fig, ax = plt.subplots(1, 2, figsize=(11, 3.4))

    bottom = np.zeros(len(arms))
    for k in kinds:
        vals = np.array([analyses[a]["counts"].get(k, 0) for a in arms], float)
        ax[0].bar(arms, vals, bottom=bottom, color=colors[k], label=k)
        bottom += vals
    style.panel(ax[0], "V39 · failure taxonomy", "", "failed probes")
    ax[0].tick_params(axis="x", labelrotation=25)
    ax[0].legend(fontsize=6.5)

    ax[1].bar(arms, [analyses[a]["accuracy"] for a in arms],
              color=[ARM_COLOR.get(a.split("_")[0], C_MUTE) for a in arms])
    style.panel(ax[1], "rank-1 accuracy", "", "accuracy")
    ax[1].tick_params(axis="x", labelrotation=25)
    ax[1].set_ylim(0, 1.05)

    fig.tight_layout()
    return style.save(fig, out_dir, "V39", "failure_taxonomy", prov)


def v40_significance(tests: dict[str, dict], out_dir: Path, prov: str = "") -> Path:
    """Paired bootstrap over threads for each arm pair.

    PASS: a CI excluding zero. FAIL: overlapping ⇒ report it as no difference, not as a
    trend.
    """
    style.apply_style()
    names = list(tests)
    diffs = [tests[n]["difference"] for n in names]
    lo = [tests[n]["difference"] - tests[n]["ci_low"] for n in names]
    hi = [tests[n]["ci_high"] - tests[n]["difference"] for n in names]
    fig, ax = plt.subplots(figsize=(7.5, 0.55 * len(names) + 2.0))
    colors = [C_PLAS if tests[n]["significant"] else C_MUTE for n in names]
    ax.errorbar(diffs, np.arange(len(names)), xerr=[lo, hi], fmt="o", ms=6,
                ecolor=C_MUTE, elinewidth=1.2, capsize=3, linestyle="none")
    ax.scatter(diffs, np.arange(len(names)), c=colors, s=48, zorder=3)
    ax.axvline(0, color=C_WARN, ls="--", lw=.9)
    ax.set_yticks(np.arange(len(names)))
    ax.set_yticklabels(names, fontsize=7)
    style.panel(ax, "V40 · paired bootstrap over threads (95% CI)", "Δ MRR", "")
    for i, n in enumerate(names):
        ax.text(diffs[i], i + 0.22, f"p={tests[n]['p_value']:.3f}", fontsize=6, color=C_MUTE,
                ha="center")
    fig.tight_layout()
    return style.save(fig, out_dir, "V40", "significance", prov)


def v46_headline(table: dict[str, dict[str, tuple[float, float]]], out_dir: Path,
                 prov: str = "") -> Path:
    """Mean ± std over seeds for every arm and headline metric."""
    style.apply_style()
    arms = list(table)
    keys = list(next(iter(table.values())))
    fig, ax = plt.subplots(1, len(keys), figsize=(2.6 * len(keys), 3.2))
    ax = np.atleast_1d(ax)
    for a, key in zip(ax, keys):
        means = [table[arm][key][0] for arm in arms]
        stds = [table[arm][key][1] for arm in arms]
        a.bar(range(len(arms)), means, yerr=stds, capsize=3,
              color=[ARM_COLOR.get(x.split("_")[0], C_MUTE) for x in arms],
              error_kw={"ecolor": C_MUTE, "elinewidth": 1})
        a.set_xticks(range(len(arms)))
        a.set_xticklabels(arms, fontsize=6, rotation=35, ha="right")
        style.panel(a, key, "", "")
        for i, (m, s) in enumerate(zip(means, stds)):
            a.text(i, m + s, f"{m:.3f}", ha="center", va="bottom", fontsize=6, color=C_MUTE)
    fig.suptitle("V46 · headline results, mean ± std over seeds", fontsize=9, color=C_MUTE)
    fig.tight_layout()
    return style.save(fig, out_dir, "V46", "headline_table", prov)
