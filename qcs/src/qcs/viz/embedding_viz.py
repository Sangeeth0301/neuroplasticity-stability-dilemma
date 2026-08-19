"""Phase 2 figures: V05 embedding geometry, V06 whitening spectrum,
V07 projection retention, V08 cosine separation, V13 feature-block balance.
"""

from __future__ import annotations

from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np

from . import style
from .style import C_MUTE, C_OK, C_PLAS, C_STAB, C_WARN


def v05_embedding_geometry(E: np.ndarray, kinds: list[str], users: list[str],
                           turn_idx: list[int], out_dir: Path, prov: str = "") -> Path:
    """2-D projection of the raw embedding space, coloured three ways.

    PASS: turn-kind structure visible (facts separate from filler).
    FAIL: only turn-index structure — the encoder captured position, not meaning.
    """
    style.apply_style()
    from sklearn.decomposition import PCA

    n = min(4000, len(E))
    sel = np.random.default_rng(0).choice(len(E), n, replace=False)
    Z = PCA(n_components=2).fit_transform(E[sel])
    k = np.array(kinds)[sel]
    u = np.array(users)[sel]
    t = np.array(turn_idx)[sel]

    fig, ax = plt.subplots(1, 3, figsize=(12, 3.4))
    kc = {"fact": C_PLAS, "update": C_WARN, "probe": C_STAB, "distractor": C_OK, "filler": C_MUTE}
    for kind, c in kc.items():
        m = k == kind
        if m.any():
            ax[0].scatter(Z[m, 0], Z[m, 1], s=2, c=c, alpha=.5, label=kind, linewidths=0)
    style.panel(ax[0], "V05 · embedding space by turn kind", "PC1", "PC2")
    ax[0].legend(markerscale=4, ncol=2)

    for i, uu in enumerate(sorted(set(u))):
        m = u == uu
        ax[1].scatter(Z[m, 0], Z[m, 1], s=2, alpha=.5, linewidths=0,
                      c=style.CYCLE[i % len(style.CYCLE)], label=uu)
    style.panel(ax[1], "by speaker", "PC1", "PC2")
    ax[1].legend(markerscale=4, ncol=2)

    sc = ax[2].scatter(Z[:, 0], Z[:, 1], s=2, c=t, cmap="viridis", alpha=.6, linewidths=0)
    style.panel(ax[2], "by turn index", "PC1", "PC2")
    plt.colorbar(sc, ax=ax[2], fraction=.046)

    fig.tight_layout()
    return style.save(fig, out_dir, "V05", "embedding_geometry", prov)


def v06_whitening_spectrum(whitener, E: np.ndarray, out_dir: Path, prov: str = "") -> Path:
    """Eigenvalue spectrum before and after whitening.

    PASS: flattened spectrum, condition number down by orders of magnitude.
    FAIL: a few dominant directions surviving — they will swallow the amplitude mass
    and leave most of the qubit register in vacuum.
    """
    style.apply_style()
    W = whitener.transform(E)
    s_raw = np.linalg.svd(E - E.mean(0), compute_uv=False) ** 2
    s_wht = np.linalg.svd(W - W.mean(0), compute_uv=False) ** 2
    c_raw, c_wht = whitener.condition_number(E)

    fig, ax = plt.subplots(1, 3, figsize=(12, 3.2))
    ax[0].semilogy(s_raw / s_raw.sum(), color=C_STAB, label="raw")
    ax[0].semilogy(s_wht / s_wht.sum(), color=C_PLAS, label="whitened")
    style.panel(ax[0], "V06 · variance spectrum", "component", "share of variance")
    ax[0].legend()
    style.note(ax[0], f"condition number\nraw {c_raw:.1f} → whitened {c_wht:.2f}")

    ax[1].plot(np.cumsum(s_raw) / s_raw.sum(), color=C_STAB, label="raw")
    ax[1].plot(np.cumsum(s_wht) / s_wht.sum(), color=C_PLAS, label="whitened")
    style.panel(ax[1], "cumulative variance", "component", "cumulative share")
    ax[1].legend()

    ax[2].hist(np.abs(E).ravel()[::37], bins=60, color=C_STAB, alpha=.6, label="raw", density=True)
    ax[2].hist(np.abs(W).ravel()[::37], bins=60, color=C_PLAS, alpha=.6, label="whitened",
               density=True)
    style.panel(ax[2], "component magnitude distribution", "|value|", "density")
    ax[2].legend()

    fig.tight_layout()
    return style.save(fig, out_dir, "V06", "whitening_spectrum", prov)


def v07_projection_retention(curves: dict[str, list[float]], ks: list[int],
                             out_dir: Path, prov: str = "") -> Path:
    """kNN retrieval accuracy at each representation stage.

    PASS: projected within ~2 points of whitened at the same k.
    FAIL: a bigger drop — the projection is discarding signal the scorer will need.
    """
    style.apply_style()
    fig, ax = plt.subplots(1, 2, figsize=(9.5, 3.2))
    colors = {"raw": C_MUTE, "whitened": C_STAB, "projected": C_PLAS}
    for name, ys in curves.items():
        ax[0].plot(ks, ys, marker="o", ms=3, color=colors.get(name), label=name)
    style.panel(ax[0], "V07 · probe→fact recall@k by representation", "k", "recall")
    ax[0].legend()
    ax[0].set_ylim(0, 1)

    names = list(curves)
    vals = [curves[n][-1] for n in names]
    ax[1].bar(names, vals, color=[colors.get(n, C_OK) for n in names])
    style.panel(ax[1], f"recall@{ks[-1]}", "", "recall")
    for i, v in enumerate(vals):
        ax[1].text(i, v, f"{v:.3f}", ha="center", va="bottom", fontsize=7, color=C_MUTE)

    fig.tight_layout()
    return style.save(fig, out_dir, "V07", "projection_retention", prov)


def v08_cosine_separation(pos: np.ndarray, hard: np.ndarray, rand: np.ndarray,
                          out_dir: Path, prov: str = "") -> Path:
    """Similarity of a probe to its required fact vs distractors vs random turns.

    PASS: visible separation between positives and hard negatives.
    FAIL: overlap — no scorer, quantum or classical, can fix an unseparable space.
    """
    style.apply_style()
    fig, ax = plt.subplots(1, 2, figsize=(9.5, 3.2))
    bins = np.linspace(-1, 1, 60)
    ax[0].hist(rand, bins=bins, color=C_MUTE, alpha=.6, density=True, label="random turn")
    ax[0].hist(hard, bins=bins, color=C_WARN, alpha=.7, density=True, label="hard distractor")
    ax[0].hist(pos, bins=bins, color=C_PLAS, alpha=.8, density=True, label="required fact")
    style.panel(ax[0], "V08 · probe↔candidate cosine similarity", "cosine", "density")
    ax[0].legend()

    def d(a, b):
        return (np.mean(a) - np.mean(b)) / np.sqrt(0.5 * (np.var(a) + np.var(b)) + 1e-12)

    labels = ["fact vs hard", "fact vs random"]
    vals = [d(pos, hard), d(pos, rand)]
    ax[1].bar(labels, vals, color=[C_WARN, C_MUTE])
    style.panel(ax[1], "separation (Cohen's d)", "", "d")
    for i, v in enumerate(vals):
        ax[1].text(i, v, f"{v:.2f}", ha="center", va="bottom", fontsize=8, color=C_MUTE)

    fig.tight_layout()
    return style.save(fig, out_dir, "V08", "cosine_separation", prov)


def v13_feature_blocks(X: np.ndarray, shares: dict[str, float], targets: list[float],
                       pr: float, out_dir: Path, prov: str = "") -> Path:
    """The assembled 1024-vector: block shares, an example vector, register map, PR.

    PASS: realised shares near their targets, participation ratio in the hundreds.
    FAIL: one block dominating and PR in the tens — the circuit is scoring on one
    feature family and ignoring the rest (the failure the pre-build demo exposed).
    """
    style.apply_style()
    fig, ax = plt.subplots(2, 2, figsize=(11, 6.0))
    names = list(shares)

    a = ax[0, 0]
    w = 0.38
    a.bar(np.arange(len(names)) - w / 2, [shares[n] for n in names], w,
          color=C_PLAS, label="realised")
    a.bar(np.arange(len(names)) + w / 2, targets, w, color=C_STAB, label="target")
    a.set_xticks(range(len(names)))
    a.set_xticklabels(names, fontsize=7)
    style.panel(a, "V13 · share of ‖x‖² per block", "", "share")
    a.legend()

    a = ax[0, 1]
    x0 = X[0]
    b = len(x0) // 4
    for i, (name, c) in enumerate(zip(names, [C_STAB, C_OK, C_PLAS, C_WARN])):
        a.plot(np.arange(i * b, (i + 1) * b), x0[i * b:(i + 1) * b], lw=.6, color=c, label=name)
    style.panel(a, "one assembled state |ψ₀⟩", "basis state index j", "amplitude")
    a.legend(ncol=2)
    style.note(a, f"‖x‖₂ = {np.linalg.norm(x0):.6f}")

    a = ax[1, 0]
    side = int(np.sqrt(len(x0)))
    im = a.imshow(x0.reshape(side, side), cmap="PuOr", aspect="auto",
                  vmin=-np.abs(x0).max(), vmax=np.abs(x0).max())
    style.panel(a, "register map (high qubits = rows)", "low qubits", "high qubits")
    plt.colorbar(im, ax=a, fraction=.046)

    a = ax[1, 1]
    prs = [1.0 / ((x**2 / (x**2).sum()) ** 2).sum() for x in X[:400]]
    a.hist(prs, bins=40, color=C_PLAS)
    style.panel(a, "participation ratio across triples", "effective basis states used", "triples")
    style.note(a, f"mean PR = {pr:.0f} / {len(x0)}\n"
                  f"({100 * pr / len(x0):.1f}% of the register)")

    fig.tight_layout()
    return style.save(fig, out_dir, "V13", "feature_blocks", prov)
