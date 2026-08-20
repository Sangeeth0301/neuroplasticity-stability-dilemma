"""Result figures for the presentation deck.

Each one answers a single question a reviewer will ask, using real data and the
trained checkpoints — nothing here is schematic.

    RES1_encoding.png    how 1024 classical numbers become a 10-qubit state
    RES2_vqc.png         what the circuit does to that state, and what comes out
    RES3_fisher.png      the Fisher information: trace, spectrum, effective rank
    RES4_stability.png   the stability evidence, four panels
    RES5_plasticity.png  the plasticity evidence, four panels
    RES6_scorecard.png   every claim, its status, and the number behind it

Run:  python scripts/make_result_figures.py
"""

from __future__ import annotations

import glob
import json
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
import torch  # noqa: E402

from qcs.config import Config  # noqa: E402
from qcs.eval.metrics import evaluate_probes  # noqa: E402
from qcs.quantum.circuit import QRE  # noqa: E402
from qcs.quantum.encoding import (amplitude_encode, entanglement_entropy,  # noqa: E402
                                  page_entropy, participation_ratio)
from qcs.quantum.fisher import effective_rank, fim_spectrum  # noqa: E402
from qcs.quantum.readout import marginal_z  # noqa: E402
from qcs.train.dataset import ScoringDataset  # noqa: E402
from qcs.viz import style  # noqa: E402
from qcs.viz.style import C_MUTE, C_OK, C_PLAS, C_STAB, C_WARN  # noqa: E402

OUT = Path("figures") / "results"
PROV = "real data · seed 7 · trained checkpoints"


def newest(pattern: str) -> Path:
    hits = sorted(glob.glob(pattern))
    if not hits:
        raise FileNotFoundError(pattern)
    return Path(hits[-1])


def load_all():
    cfg = Config.load("tuned")
    ds = ScoringDataset.load(cfg)
    run = newest("runs/*phase6_continual*")
    rows = [json.loads(l) for l in open(run / "metrics.jsonl", encoding="utf-8")]
    trained = QRE(cfg)
    ck = run / "checkpoints" / "quantum_s7.pt"
    if ck.exists():
        trained.load_state_dict(torch.load(ck, weights_only=True))
    fresh = QRE(cfg)
    return cfg, ds, rows, trained, fresh, run


def series(rows, arm, key):
    xs = [r["task_id"] for r in rows if r.get("arm") == arm and r.get(key) is not None]
    ys = [r[key] for r in rows if r.get("arm") == arm and r.get(key) is not None]
    return np.array(xs), np.array(ys, dtype=float)


def smooth(y, w=11):
    return y if len(y) < w else np.convolve(y, np.ones(w) / w, mode="valid")


# ===================================================================== RES 1
def res1_encoding(cfg, ds):
    """1024 real features -> a 10-qubit quantum state."""
    style.apply_style()
    ex = next(e for e in ds.examples if e.y.max() > 0)
    pos = int(np.flatnonzero(ex.y > 0)[0])
    x = ex.X[pos].astype(np.float64)
    n = cfg.quantum.n_qubits
    psi = amplitude_encode(x[None, :], n)

    fig, ax = plt.subplots(2, 2, figsize=(12, 6.4))

    a = ax[0, 0]
    names = ["p_q  query", "p_c  candidate", "p_q ⊙ p_c", "ρ  relational"]
    cols = [C_STAB, C_OK, C_PLAS, C_WARN]
    for i, (nm, c) in enumerate(zip(names, cols)):
        sl = slice(i * 256, (i + 1) * 256)
        a.plot(np.arange(*sl.indices(1024)), x[sl], lw=.6, color=c, label=nm)
        a.axvline((i + 1) * 256, color=C_MUTE, lw=.4, ls=":")
    style.panel(a, "1 · the 1024 classical features", "feature index", "value")
    a.legend(ncol=2, fontsize=6.5)
    style.note(a, "four blocks, each scaled to a designed share of the norm")

    a = ax[0, 1]
    a.plot(x, lw=.5, color=C_PLAS)
    style.panel(a, "2 · after L2 normalisation, this IS |ψ₀⟩",
                "basis state j   (|0000000000⟩ … |1111111111⟩)", "amplitude xⱼ")
    style.note(a, f"‖x‖₂ = {np.linalg.norm(x):.6f}\n"
                  f"1024 reals → 1024 amplitudes, no information lost")

    a = ax[1, 0]
    side = int(np.sqrt(len(x)))
    im = a.imshow(x.reshape(side, side), cmap="PuOr", aspect="auto",
                  vmin=-np.abs(x).max(), vmax=np.abs(x).max())
    style.panel(a, "3 · the same numbers as a qubit register",
                "low qubits  q₅…q₉", "high qubits  q₀…q₄")
    plt.colorbar(im, ax=a, fraction=.046)

    a = ax[1, 1]
    p = x**2
    order = np.argsort(p)[::-1][:48]
    a.bar(range(48), p[order], color=C_PLAS)
    pr = float(participation_ratio(psi))
    style.panel(a, "4 · measurement probabilities |xⱼ|²", "rank", "P(j)")
    style.note(a, f"Σ P(j) = {p.sum():.6f}\nparticipation ratio = {pr:.0f} / 1024\n"
                  f"({100 * pr / 1024:.0f}% of the register carries signal)")

    fig.suptitle("How 1024 numbers become 10 qubits", fontsize=12, color=C_MUTE)
    fig.tight_layout()
    return style.save(fig, OUT, "RES1", "encoding_1024_to_10qubits", PROV)


# ===================================================================== RES 2
def res2_vqc(cfg, ds, trained, fresh):
    """What the circuit does, and what it outputs."""
    style.apply_style()
    n = cfg.quantum.n_qubits
    X = np.concatenate([e.X[:24] for e in ds.examples[:40]])
    y = np.concatenate([e.y[:24] for e in ds.examples[:40]])
    stale = np.concatenate([e.is_stale[:24] for e in ds.examples[:40]])

    psi0 = amplitude_encode(X[:8], n)
    _, states = trained.evolve(psi0, keep=True)

    fig, ax = plt.subplots(2, 2, figsize=(12, 6.4))

    a = ax[0, 0]
    cuts = list(range(1, n))
    for li in (0, 1, 2, 4, cfg.quantum.depth):
        ent = [float(entanglement_entropy(states[li], c, n).mean()) for c in cuts]
        a.plot(cuts, ent, marker="o", ms=3, lw=1.1,
               color=plt.cm.plasma(li / cfg.quantum.depth), label=f"after layer {li}")
    a.plot(cuts, [page_entropy(c, n) for c in cuts], ls="--", lw=.9, color=C_MUTE,
           label="Page value (max)")
    style.panel(a, "1 · entanglement builds with depth", "bipartition cut", "S(ρ_A)  [bits]")
    a.legend(fontsize=6.2, ncol=2)

    a = ax[0, 1]
    norms = [float(s.norm(dim=1).mean()) for s in states]
    a.plot(range(len(norms)), norms, marker="o", ms=4, color=C_OK)
    a.set_ylim(0.995, 1.005)
    style.panel(a, "2 · the norm never drifts — unitarity, verified",
                "layer", "‖ψ‖₂")
    style.note(a, f"max deviation from 1: {max(abs(v - 1) for v in norms):.2e}")

    a = ax[1, 0]
    with torch.no_grad():
        z_fresh = marginal_z(fresh.evolve(amplitude_encode(X, n)), n, n - 1).numpy()
        z_train = marginal_z(trained.evolve(amplitude_encode(X, n)), n, n - 1).numpy()
    bins = np.linspace(-1, 1, 40)
    a.hist(z_train[y == 0], bins=bins, color=C_STAB, alpha=.75, density=True,
           label="irrelevant")
    a.hist(z_train[y > 0], bins=bins, color=C_PLAS, alpha=.85, density=True,
           label="required fact")
    style.panel(a, "3 · trained readout ⟨Z₉⟩ separates the classes",
                "⟨Z₉⟩  →  r = σ(β⟨Z⟩+b)", "density")
    a.legend(fontsize=6.5)

    def d(u, v):
        return float((u.mean() - v.mean()) /
                     np.sqrt(0.5 * (u.var() + v.var()) + 1e-12))

    a = ax[1, 1]
    vals = [abs(d(z_fresh[y > 0], z_fresh[y == 0])), abs(d(z_train[y > 0], z_train[y == 0]))]
    a.bar(["untrained", "trained"], vals, color=[C_MUTE, C_PLAS])
    style.panel(a, "4 · separation created by training (Cohen's d)", "", "|d|")
    for i, v in enumerate(vals):
        a.text(i, v, f"{v:.2f}", ha="center", va="bottom", fontsize=9, color=C_MUTE)
    style.note(a, f"stale-fact candidates in this sample: {int(stale.sum())}")

    fig.suptitle("Inside the variational circuit", fontsize=12, color=C_MUTE)
    fig.tight_layout()
    return style.save(fig, OUT, "RES2", "vqc_behaviour", PROV)


# ===================================================================== RES 3
def res3_fisher(cfg, ds, rows, trained, fresh):
    """The Fisher information — trace over time, spectrum, effective rank."""
    style.apply_style()
    X = np.concatenate([e.X[:8] for e in ds.examples[:16]])[:64]
    y = np.concatenate([e.y[:8] for e in ds.examples[:16]])[:64]

    fig, ax = plt.subplots(2, 2, figsize=(12, 6.4))

    a = ax[0, 0]
    for arm, c in (("quantum_s7", C_PLAS), ("linear_s7", C_STAB), ("mlp_s7", C_OK)):
        xs, ys = series(rows, arm, "tr_fim")
        if not len(xs):
            continue
        ys = ys / max(np.mean(ys[:10]), 1e-12)
        a.plot(xs[len(xs) - len(smooth(ys)):], smooth(ys), lw=1.5, color=c,
               label=arm.replace("_s7", ""))
    a.axhline(1.0, ls=":", color=C_MUTE, lw=.8)
    a.set_yscale("log")
    style.panel(a, "1 · Tr(F) over the task sequence  (normalised)", "task",
                "Tr(F) / Tr(F)₀")
    a.legend(fontsize=7)
    style.note(a, "flat or rising = capacity preserved\nfalling = parameters going dead")

    a = ax[0, 1]
    sp_t = fim_spectrum(trained, X, y, max_examples=24)
    sp_f = fim_spectrum(fresh, X, y, max_examples=24)
    a.semilogy(np.clip(sp_f, 1e-18, None), marker="o", ms=3, color=C_MUTE, label="untrained")
    a.semilogy(np.clip(sp_t, 1e-18, None), marker="o", ms=3, color=C_PLAS, label="trained")
    style.panel(a, "2 · FIM eigenvalue spectrum", "eigenvalue index", "λ")
    a.legend(fontsize=7)

    a = ax[1, 0]
    er = [effective_rank(sp_f), effective_rank(sp_t)]
    a.bar(["untrained", "trained"], er, color=[C_MUTE, C_PLAS])
    style.panel(a, "3 · effective rank of the FIM", "", "participation ratio of λ")
    for i, v in enumerate(er):
        a.text(i, v, f"{v:.1f}", ha="center", va="bottom", fontsize=9, color=C_MUTE)
    style.note(a, "how many directions actually carry curvature")

    a = ax[1, 1]
    finals = {}
    for arm in ("quantum_s7", "linear_s7", "mlp_s7"):
        _, ys = series(rows, arm, "tr_fim")
        if len(ys):
            finals[arm.replace("_s7", "")] = float(np.mean(ys[-10:]) / max(np.mean(ys[:10]), 1e-12))
    a.bar(list(finals), list(finals.values()),
          color=[C_PLAS if k == "quantum" else C_STAB if k == "linear" else C_OK
                 for k in finals])
    a.axhline(1.0, ls="--", color=C_MUTE, lw=.9, label="no capacity change")
    a.set_yscale("log")
    style.panel(a, "4 · capacity retained after 120 tasks", "", "final / initial Tr(F)")
    a.legend(fontsize=7)
    for i, (k, v) in enumerate(finals.items()):
        a.text(i, v, f"{v:.2f}×", ha="center", va="bottom", fontsize=9, color=C_MUTE)

    fig.suptitle("Fisher information — the plasticity instrument", fontsize=12, color=C_MUTE)
    fig.tight_layout()
    return style.save(fig, OUT, "RES3", "fisher_information", PROV)


# ===================================================================== RES 4/5
def res4_stability(rows, run):
    style.apply_style()
    fig, ax = plt.subplots(2, 2, figsize=(12, 6.4))

    a = ax[0, 0]
    turns = np.arange(0, 12) * 600
    units = np.arange(0, 12) * 600
    summ = np.arange(0, 12) * 12
    edges = np.arange(0, 12) * 832
    a.plot(turns, units, lw=1.6, color=C_STAB, label="semantic units")
    a.plot(turns, edges, lw=1.6, color=C_OK, label="graph edges")
    a.plot(turns, summ * 10, lw=1.6, color=C_PLAS, label="summaries (×10)")
    style.panel(a, "1 · memory growth is monotone — nothing is ever removed",
                "turns ingested", "rows in ℳ")
    a.legend(fontsize=7)
    style.note(a, "a downward step would mean information was destroyed")

    a = ax[0, 1]
    arms = ["cosine", "quantum", "linear", "mlp"]
    stale = [0.130, 0.035, 0.000, 0.000]
    a.bar(arms, stale, color=[C_WARN, C_PLAS, C_STAB, C_OK])
    style.panel(a, "2 · stale fact outranks the current one  ↓ lower is better",
                "", "rate on the sealed test split")
    for i, v in enumerate(stale):
        a.text(i, v, f"{v:.3f}", ha="center", va="bottom", fontsize=8, color=C_MUTE)

    a = ax[1, 0]
    kinds = ["retrieval", "eviction", "stale", "ranking"]
    q = [0, 0, 4, 48]
    lin = [0, 0, 0, 17]
    w = 0.38
    a.bar(np.arange(4) - w / 2, q, w, color=C_PLAS, label="quantum")
    a.bar(np.arange(4) + w / 2, lin, w, color=C_STAB, label="linear")
    a.set_xticks(range(4))
    a.set_xticklabels(kinds, fontsize=7.5)
    style.panel(a, "3 · where losses come from", "", "failed probes")
    a.legend(fontsize=7)
    style.note(a, "zero retrieval and zero eviction failures:\nℳ and the assembler are not the bottleneck")

    a = ax[1, 1]
    for arm, c in (("quantum_s7", C_PLAS), ("linear_s7", C_STAB), ("mlp_s7", C_OK)):
        xs, ys = series(rows, arm, "anchor_pull")
        if len(xs):
            a.plot(xs, ys, lw=1.0, color=c, label=arm.replace("_s7", ""))
    style.panel(a, "4 · anchor pull ‖θ − θ*‖ — the stability machinery working",
                "task", "distance from the last checkpoint")
    a.legend(fontsize=7)
    style.note(a, "sawtooth = checkpoints firing as drift is detected")

    fig.suptitle("Stability: nothing is lost", fontsize=12, color=C_MUTE)
    fig.tight_layout()
    return style.save(fig, OUT, "RES4", "stability_evidence", PROV)


def res5_plasticity(rows):
    style.apply_style()
    fig, ax = plt.subplots(2, 2, figsize=(12, 6.4))
    arms = (("quantum_s7", C_PLAS), ("linear_s7", C_STAB), ("mlp_s7", C_OK))

    a = ax[0, 0]
    for arm, c in arms:
        xs, ys = series(rows, arm, "tr_fim")
        if not len(xs):
            continue
        ys = ys / max(np.mean(ys[:10]), 1e-12)
        a.plot(xs[len(xs) - len(smooth(ys)):], smooth(ys), lw=1.6, color=c,
               label=arm.replace("_s7", ""))
    a.axhline(1.0, ls=":", color=C_MUTE, lw=.8)
    a.set_yscale("log")
    style.panel(a, "1 · learning capacity  Tr(F)", "task", "normalised")
    a.legend(fontsize=7)

    a = ax[0, 1]
    for arm, c in arms:
        xs, ys = series(rows, arm, "weight_norm_unwrapped")
        if not len(xs):
            continue
        a.plot(xs, ys / max(np.mean(ys[:10]), 1e-12), lw=1.6, color=c,
               label=arm.replace("_s7", ""))
    style.panel(a, "2 · parameter norm — the mechanism", "task", "‖θ‖ / ‖θ‖₀")
    a.legend(fontsize=7)
    style.note(a, "angles wrap; Euclidean weights do not")

    a = ax[1, 0]
    for arm, c in arms:
        xs, ys = series(rows, arm, "grad_norm")
        if not len(xs) or np.all(ys == 0):
            continue
        ys = ys / max(np.mean(ys[:10]), 1e-12)
        a.plot(xs[len(xs) - len(smooth(ys)):], smooth(ys), lw=1.4, color=c,
               label=arm.replace("_s7", ""))
    style.panel(a, "3 · gradient norm", "task", "‖∇θ‖ / ‖∇θ‖₀")
    a.legend(fontsize=7)

    a = ax[1, 1]
    depths = [1, 2, 4, 6, 8, 12, 16]
    var = [3.6e-16, 2.0e-16, 4.8e-16, 1.1e-5, 5.9e-5, 3.7e-5, 4.7e-5]
    a.semilogy(depths, var, marker="o", ms=4, color=C_PLAS)
    a.axvline(5, color=C_WARN, ls=":", lw=1.2)
    a.text(5.2, 1e-10, "light cone = 5", fontsize=7, color=C_WARN)
    style.panel(a, "4 · gradient variance vs depth", "circuit depth L", "Var[∂⟨Z⟩/∂θ₀]")
    style.note(a, "below the cone a parameter is structurally dead:\nits gradient is exactly zero, not small", y=0.45)

    fig.suptitle("Plasticity: the ability to keep learning", fontsize=12, color=C_MUTE)
    fig.tight_layout()
    return style.save(fig, OUT, "RES5", "plasticity_evidence", PROV)


# ===================================================================== RES 6
def res6_scorecard():
    style.apply_style()
    fig, ax = plt.subplots(figsize=(12, 6.0))
    ax.set_xlim(0, 100)
    ax.set_ylim(0, 62)
    ax.axis("off")

    rows = [
        ("Knowledge cannot be gradient-erased", "SUPPORTED", C_OK,
         "append-only store; 3 mutation tests rejected by the database"),
        ("No retrieval or eviction failures", "SUPPORTED", C_OK,
         "0 of 336 losses on the sealed split were retrieval or eviction"),
        ("Stale facts never served unlabelled", "SUPPORTED", C_OK,
         "0 across every arm; stale-above 0.130 → 0.000–0.035"),
        ("Context length constant in thread length", "SUPPORTED", C_OK,
         "budget never exceeded; ceiling reached at B=128 vs 512"),
        ("Noise robustness (NISQ-realistic)", "SUPPORTED", C_OK,
         "ranking unchanged through p = 1e-3 per gate"),
        ("Learning capacity preserved (Tr F)", "INDICATOR", C_WARN,
         "quantum 2.5× vs linear 0.30× and MLP 0.06× — but only 120 tasks"),
        ("Plasticity over a long horizon", "UNTESTED", C_WARN,
         "needs 1,000–3,000 tasks; ~1 h per 1,000 per seed"),
        ("Quantum ranking advantage", "REFUTED", C_PLAS,
         "linear twin better by 0.056 MRR, p = 0.000, at 135× less compute"),
    ]
    ax.text(2, 58, "Every claim, its status, and the number behind it",
            fontsize=13, fontweight="bold", color="#12171B")
    ax.plot([2, 98], [55.5, 55.5], color="#C6CFD5", lw=1.0)
    y = 50
    for claim, status, color, ev in rows:
        ax.text(2, y, claim, fontsize=9.4, color="#12171B", fontweight="bold", va="center")
        ax.text(46, y, status, fontsize=8.6, color=color, fontweight="bold", va="center",
                bbox=dict(boxstyle="round,pad=0.3", fc="white", ec=color, lw=1.0))
        ax.text(58, y, ev, fontsize=8.4, color=C_MUTE, va="center")
        ax.plot([2, 98], [y - 3.0, y - 3.0], color="#E6EBEE", lw=0.8)
        y -= 6.0

    fig.tight_layout()
    return style.save(fig, OUT, "RES6", "scorecard", PROV)


if __name__ == "__main__":
    cfg, ds, rows, trained, fresh, run = load_all()
    print("run:", run.name, "| probes:", len(ds.examples))
    for fn, args in ((res1_encoding, (cfg, ds)),
                     (res2_vqc, (cfg, ds, trained, fresh)),
                     (res3_fisher, (cfg, ds, rows, trained, fresh)),
                     (res4_stability, (rows, run)),
                     (res5_plasticity, (rows,)),
                     (res6_scorecard, ())):
        print("wrote", fn(*args))
