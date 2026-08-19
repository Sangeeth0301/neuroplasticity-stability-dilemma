"""Phase 4 figures: V14 amplitudes, V15 register map, V16 basis probabilities,
V17 gradient variance vs depth, V18 entanglement, V19 gradient agreement,
V20 readout separation.

These are the "what does my dataset look like as qubits" figures — drawn from real
probe/candidate pairs, not synthetic vectors.
"""

from __future__ import annotations

from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np

from . import style
from .style import C_MUTE, C_OK, C_PLAS, C_STAB, C_WARN


def v14_v16_encoding(X: np.ndarray, y: np.ndarray, n_qubits: int,
                     out_dir: Path, prov: str = "") -> Path:
    """One real (probe, required-fact) pair as a quantum state, plus the ensemble view."""
    style.apply_style()
    pos = int(np.flatnonzero(y > 0)[0])
    x = X[pos]
    side = int(np.sqrt(len(x)))

    fig, ax = plt.subplots(2, 2, figsize=(11, 6.0))

    a = ax[0, 0]
    a.plot(x, lw=.5, color=C_PLAS)
    style.panel(a, f"V14 · |ψ₀⟩ for a real probe→fact pair ({n_qubits} qubits)",
                "basis state j", "amplitude xⱼ")
    style.note(a, f"‖x‖₂ = {np.linalg.norm(x):.6f}\n"
                  f"1024 features → 1024 amplitudes, lossless")

    a = ax[0, 1]
    im = a.imshow(x.reshape(side, side), cmap="PuOr", aspect="auto",
                  vmin=-np.abs(x).max(), vmax=np.abs(x).max())
    style.panel(a, "V15 · register map", "low qubits", "high qubits")
    plt.colorbar(im, ax=a, fraction=.046)

    a = ax[1, 0]
    p = x**2
    top = np.argsort(p)[::-1][:40]
    a.bar(range(40), p[top], color=C_PLAS)
    style.panel(a, "V16 · measurement probabilities |xⱼ|² (top 40)", "rank", "P(j)")
    style.note(a, f"Σp = {p.sum():.6f}\nmax P(j) = {p.max():.4f}\n"
                  f"participation ratio = {1 / (p**2).sum():.0f} / {len(x)}")

    a = ax[1, 1]
    prs = 1.0 / ((X**2 / (X**2).sum(axis=1, keepdims=True)) ** 2).sum(axis=1)
    a.hist(prs[y == 0], bins=40, color=C_STAB, alpha=.7, label="irrelevant candidates")
    a.hist(prs[y > 0], bins=40, color=C_PLAS, alpha=.85, label="required facts")
    style.panel(a, "participation ratio across candidates", "effective basis states", "count")
    a.legend()

    fig.tight_layout()
    return style.save(fig, out_dir, "V14", "encoding_real_pair", prov)


def v17_grad_variance(var_local: dict[int, float], var_global: dict[int, float],
                      light_cone: int, params_per_layer: int,
                      out_dir: Path, prov: str = "") -> Path:
    """Var[∂⟨O⟩/∂θ] vs depth — the barren-plateau guard, on real inputs.

    PASS: local observable flat above the light-cone depth.
    FAIL: exponential decay ⇒ cap the depth there.
    """
    style.apply_style()
    depths = sorted(var_local)
    fig, ax = plt.subplots(1, 2, figsize=(10, 3.4))

    a = ax[0]
    a.semilogy(depths, [max(var_local[d], 1e-35) for d in depths], marker="o", ms=4,
               color=C_PLAS, label="local ⟨Z_q⟩ (default)")
    a.semilogy(depths, [max(var_global[d], 1e-35) for d in depths], marker="s", ms=4,
               color=C_STAB, label="global ⟨Z⊗…⊗Z⟩")
    a.axvline(light_cone, color=C_WARN, ls=":", lw=1)
    a.text(light_cone, a.get_ylim()[1] * 1e-2, f" light cone = {light_cone}",
           fontsize=6.5, color=C_WARN)
    style.panel(a, "V17 · gradient variance vs circuit depth", "depth L", "Var[∂⟨O⟩/∂θ₀]")
    a.legend()
    style.note(a, "below the light cone the gradient is exactly zero,\n"
                  "not merely small — those parameters cannot train", y=0.30)

    a = ax[1]
    a.plot(depths, [d * params_per_layer for d in depths], marker="o", ms=4, color=C_WARN)
    style.panel(a, "parameter count vs depth", "depth L", "|θ|")
    for d in depths:
        a.annotate(f"{d * params_per_layer}", (d, d * params_per_layer), fontsize=6,
                   color=C_MUTE, xytext=(0, 5), textcoords="offset points", ha="center")

    fig.tight_layout()
    return style.save(fig, out_dir, "V17", "grad_variance_vs_depth", prov)


def v18_entanglement(ent: dict[int, np.ndarray], page: list[float], n_qubits: int,
                     out_dir: Path, prov: str = "") -> Path:
    """Entanglement entropy across every bipartition, by layer.

    PASS: rising toward the Page value with depth — the ansatz is genuinely entangling.
    FAIL: flat curves ⇒ a pairing bug leaving qubits disconnected.
    """
    style.apply_style()
    cuts = list(range(1, n_qubits))
    fig, ax = plt.subplots(1, 2, figsize=(10, 3.4))

    for i, (layer, vals) in enumerate(sorted(ent.items())):
        ax[0].plot(cuts, vals, marker="o", ms=3, lw=1,
                   color=plt.cm.plasma(i / max(len(ent) - 1, 1)), label=f"after layer {layer}")
    ax[0].plot(cuts, page, ls="--", lw=.9, color=C_MUTE, label="Page value (max)")
    style.panel(ax[0], "V18 · entanglement entropy vs bipartition cut", "cut position",
                "S(ρ_A)  [bits]")
    ax[0].legend(ncol=2, fontsize=6)

    mid = n_qubits // 2 - 1
    layers = sorted(ent)
    ax[1].plot(layers, [ent[layer][mid] for layer in layers], marker="o", ms=4, color=C_PLAS)
    ax[1].axhline(page[mid], ls="--", color=C_MUTE, label="Page value")
    style.panel(ax[1], "mid-cut entropy vs depth", "layer", "S(ρ_A)  [bits]")
    ax[1].legend()

    fig.tight_layout()
    return style.save(fig, out_dir, "V18", "entanglement_vs_depth", prov)


def v19_gradient_agreement(pairs_exp: list, pairs_prod: list, out_dir: Path,
                           prov: str = "") -> Path:
    """Autodiff vs parameter-shift, for both gate forms.

    PASS: the product form sits on the diagonal (<1e-5). The exponential form does NOT,
    and that is a mathematical fact about non-commuting generators, not a bug — it is
    recorded here so nobody "fixes" it later by changing the training path.
    """
    style.apply_style()
    fig, ax = plt.subplots(1, 2, figsize=(9.5, 3.4))
    for a, pairs, name, c in ((ax[0], pairs_prod, "product form (shift rule exact)", C_PLAS),
                              (ax[1], pairs_exp, "exponential form (shift rule invalid)", C_WARN)):
        ad = [p[1] for p in pairs]
        ps = [p[2] for p in pairs]
        lim = max(1e-3, max(abs(v) for v in ad + ps) * 1.2)
        a.plot([-lim, lim], [-lim, lim], ls="--", lw=.8, color=C_MUTE)
        a.scatter(ad, ps, s=28, color=c)
        style.panel(a, f"V19 · {name}", "autodiff ∂⟨Z⟩/∂θ", "parameter shift")
        err = max(abs(x - y) for x, y in zip(ad, ps))
        style.note(a, f"max |Δ| = {err:.2e}")
    fig.tight_layout()
    return style.save(fig, out_dir, "V19", "gradient_agreement", prov)


def v20_readout_separation(z_pos: np.ndarray, z_neg: np.ndarray, z_stale: np.ndarray | None,
                           title_suffix: str, out_dir: Path, prov: str = "",
                           fig_id: str = "V20") -> Path:
    """⟨Z⟩ for required facts vs irrelevant candidates (and stale facts).

    PASS at init: overlapping (d ≈ 0) — an untrained circuit knows nothing.
    PASS after training: separated (d > 1), with stale facts pushed to the negative side.
    """
    style.apply_style()
    fig, ax = plt.subplots(1, 2, figsize=(9.5, 3.2))
    bins = np.linspace(-1, 1, 45)
    ax[0].hist(z_neg, bins=bins, color=C_STAB, alpha=.75, density=True, label="irrelevant")
    if z_stale is not None and len(z_stale):
        ax[0].hist(z_stale, bins=bins, color=C_WARN, alpha=.75, density=True, label="stale fact")
    ax[0].hist(z_pos, bins=bins, color=C_PLAS, alpha=.85, density=True, label="required fact")
    style.panel(ax[0], f"{fig_id} · ⟨Z⟩ readout separation {title_suffix}",
                "⟨Z⟩ → r = σ(β⟨Z⟩+b)", "density")
    ax[0].legend()

    def d(a, b):
        return float((np.mean(a) - np.mean(b)) /
                     np.sqrt(0.5 * (np.var(a) + np.var(b)) + 1e-12))

    labels = ["fact vs irrelevant"]
    vals = [d(z_pos, z_neg)]
    if z_stale is not None and len(z_stale):
        labels.append("fact vs stale")
        vals.append(d(z_pos, z_stale))
    ax[1].bar(labels, vals, color=[C_STAB, C_WARN][: len(vals)])
    style.panel(ax[1], "separation (Cohen's d)", "", "d")
    for i, v in enumerate(vals):
        ax[1].text(i, v, f"{v:.2f}", ha="center", va="bottom", fontsize=8, color=C_MUTE)
    ax[1].axhline(0, color=C_MUTE, lw=.6)

    fig.tight_layout()
    return style.save(fig, out_dir, fig_id, "readout_separation", prov)
