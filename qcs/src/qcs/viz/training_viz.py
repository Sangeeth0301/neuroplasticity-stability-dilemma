"""Phase 6 figures: V21 accuracy vs task, V23 Tr(F), V24 norms, V26 anchor pull,
V27 clip rate / drift, V31 CL matrix, V32 BWT/FWT, V33 plasticity index.

Every comparison overlays the arms on identical axes — never two plots side by side at
different scales.
"""

from __future__ import annotations

from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np

from . import style
from .style import ARM_COLOR, C_MUTE, C_OK, C_PLAS, C_STAB, C_WARN


def _series(rows: list[dict], arm: str, key: str):
    xs = [r["task_id"] for r in rows if r.get("arm") == arm and r.get(key) is not None]
    ys = [r[key] for r in rows if r.get("arm") == arm and r.get(key) is not None]
    return np.array(xs), np.array(ys, dtype=float)


def _smooth(y: np.ndarray, w: int = 15) -> np.ndarray:
    if len(y) < w or w < 2:
        return y
    return np.convolve(y, np.ones(w) / w, mode="valid")


def v21_v25_training(rows: list[dict], arms: list[str], out_dir: Path, prov: str = "",
                     window: int = 15) -> Path:
    """The four curves that decide the whole project.

    PASS: quantum accuracy and Tr(F) flat across the task sequence while the classical
    arm's decay; quantum norms bounded while the classical arm's grow.
    FAIL (for the hypothesis): the arms overlap — report it as a null result.
    """
    style.apply_style()
    fig, ax = plt.subplots(2, 2, figsize=(11, 6.2))

    a = ax[0, 0]
    for arm in arms:
        x, y = _series(rows, arm, "acc")
        if not len(x):
            continue
        a.plot(x[window - 1:] if len(y) >= window else x, _smooth(y, window),
               color=ARM_COLOR.get(arm.split("_")[0], C_MUTE), lw=1.4, label=arm)
    style.panel(a, "V21 · task accuracy (MRR) vs task index", "task", "MRR on the current task")
    a.legend()
    style.note(a, f"{window}-task moving average")

    a = ax[0, 1]
    for arm in arms:
        x, y = _series(rows, arm, "tr_fim")
        if not len(x):
            continue
        y = y / max(np.mean(y[:10]), 1e-12)      # normalise to the first 10 tasks
        a.plot(x[window - 1:] if len(y) >= window else x, _smooth(y, window),
               color=ARM_COLOR.get(arm.split("_")[0], C_MUTE), lw=1.4, label=arm)
    a.axhline(1.0, ls=":", color=C_MUTE, lw=.8)
    style.panel(a, "V23 · Tr(F) — effective learning capacity", "task", "Tr(F) / Tr(F)₀")
    a.set_yscale("log")
    a.legend()
    style.note(a, "flat = plasticity preserved\ncollapsing = parameters going functionally dead")

    a = ax[1, 0]
    for arm in arms:
        x, y = _series(rows, arm, "grad_norm")
        if not len(x) or np.all(y == 0):
            continue
        y = y / max(np.mean(y[:10]), 1e-12)
        a.plot(x[window - 1:] if len(y) >= window else x, _smooth(y, window),
               color=ARM_COLOR.get(arm.split("_")[0], C_MUTE), lw=1.4, label=arm)
    style.panel(a, "V24a · gradient norm (normalised to first 10 tasks)", "task", "‖∇θ‖ ratio")
    a.legend()

    a = ax[1, 1]
    for arm in arms:
        x, y = _series(rows, arm, "weight_norm_unwrapped")
        if not len(x):
            continue
        y = y / max(np.mean(y[:10]), 1e-12)
        a.plot(x, y, color=ARM_COLOR.get(arm.split("_")[0], C_MUTE), lw=1.4, label=arm)
    style.panel(a, "V24b · parameter norm (unwrapped)", "task", "‖θ‖ ratio")
    a.legend()
    style.note(a, "the compact manifold is the mechanism:\nangles wrap, Euclidean weights do not")

    fig.tight_layout()
    return style.save(fig, out_dir, "V21", "training_curves", prov)


def v26_v29_stability(rows: list[dict], arms: list[str], out_dir: Path, prov: str = "") -> Path:
    """Is the stability machinery doing anything?

    PASS: anchor pull is a sawtooth resetting at checkpoints; clip rate between ~1% and
    ~30%; drift events aligned with thread transitions.
    """
    style.apply_style()
    fig, ax = plt.subplots(1, 3, figsize=(12, 3.2))

    for arm in arms:
        x, y = _series(rows, arm, "anchor_pull")
        if len(x):
            ax[0].plot(x, y, color=ARM_COLOR.get(arm.split("_")[0], C_MUTE), lw=1.0, label=arm)
    style.panel(ax[0], "V26 · anchor pull ‖θ − θ*‖", "task", "distance")
    ax[0].legend()
    style.note(ax[0], "sawtooth = checkpoints firing\nmonotone growth = detector never fires")

    for arm in arms:
        x, y = _series(rows, arm, "drift_psi")
        if len(x):
            ax[1].plot(x, y, color=ARM_COLOR.get(arm.split("_")[0], C_MUTE), lw=.9, label=arm)
    style.panel(ax[1], "V29 · score-distribution drift (PSI)", "task", "PSI")
    ax[1].legend()

    for arm in arms:
        x, y = _series(rows, arm, "clip_rate")
        if len(x):
            ax[2].plot(x, y, color=ARM_COLOR.get(arm.split("_")[0], C_MUTE), lw=1.2, label=arm)
    ax[2].axhline(0.5, ls="--", color=C_WARN, lw=.8, label="strangling threshold")
    style.panel(ax[2], "V27 · trust-region clip rate", "task", "cumulative clip rate")
    ax[2].set_ylim(0, 1)
    ax[2].legend()

    fig.tight_layout()
    return style.save(fig, out_dir, "V26", "stability_machinery", prov)


def v31_v33_continual(results: dict, out_dir: Path, prov: str = "") -> Path:
    """The continual-learning summary: transfer matrix, BWT/FWT, plasticity index."""
    style.apply_style()
    arms = list(results)
    fig, ax = plt.subplots(1, 3, figsize=(12.5, 3.6))

    A = results[arms[0]]["A"]
    im = ax[0].imshow(A, cmap="viridis", aspect="auto", vmin=0, vmax=1)
    style.panel(ax[0], f"V31 · transfer matrix A[i][j] — {arms[0]}",
                "task j (evaluated)", "task i (after training)")
    plt.colorbar(im, ax=ax[0], fraction=.046)

    w = 0.36
    xs = np.arange(len(arms))
    ax[1].bar(xs - w / 2, [results[a]["bwt"] for a in arms], w, color=C_STAB, label="BWT")
    ax[1].bar(xs + w / 2, [results[a]["fwt"] for a in arms], w, color=C_OK, label="FWT")
    ax[1].axhline(0, color=C_MUTE, lw=.6)
    ax[1].set_xticks(xs)
    ax[1].set_xticklabels(arms, fontsize=6.5, rotation=12)
    style.panel(ax[1], "V32 · backward / forward transfer", "", "transfer")
    ax[1].legend()

    vals = [results[a]["plasticity_index"] for a in arms]
    ax[2].bar(xs, vals, color=[ARM_COLOR.get(a.split("_")[0], C_MUTE) for a in arms])
    ax[2].axhline(1.0, ls="--", color=C_MUTE, label="no plasticity loss")
    ax[2].set_xticks(xs)
    ax[2].set_xticklabels(arms, fontsize=6.5, rotation=12)
    style.panel(ax[2], "V33 · plasticity index (last / first tasks)", "", "ratio")
    ax[2].legend()
    for i, v in enumerate(vals):
        ax[2].text(i, v, f"{v:.3f}", ha="center", va="bottom", fontsize=7, color=C_MUTE)

    fig.tight_layout()
    return style.save(fig, out_dir, "V31", "continual_summary", prov)


def v34_metrics_table(metrics: dict, out_dir: Path, prov: str = "") -> Path:
    """Headline ranking metrics per arm, before and after continual training."""
    style.apply_style()
    arms = list(metrics)
    keys = ["rank1", "mrr", "recall@3", "auc", "stale_above_current"]
    fig, ax = plt.subplots(1, len(keys), figsize=(13, 3.0))
    for a, key in zip(ax, keys):
        vals = [metrics[arm].get(key, np.nan) for arm in arms]
        a.bar(range(len(arms)), vals,
              color=[ARM_COLOR.get(x.split("_")[0], C_MUTE) for x in arms])
        a.set_xticks(range(len(arms)))
        a.set_xticklabels(arms, fontsize=6, rotation=35, ha="right")
        style.panel(a, key + (" ↓ lower is better" if "stale" in key else ""), "", "")
        a.set_ylim(0, 1.05)
        for i, v in enumerate(vals):
            if not np.isnan(v):
                a.text(i, v, f"{v:.2f}", ha="center", va="bottom", fontsize=6, color=C_MUTE)
    fig.suptitle("V34 · ranking metrics by arm", fontsize=9, color=C_MUTE)
    fig.tight_layout()
    return style.save(fig, out_dir, "V34", "metrics_by_arm", prov)


_ = C_PLAS
