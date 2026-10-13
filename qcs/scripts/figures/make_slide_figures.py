"""Result charts sized for the final deck (large type, drawn at the size they are shown).

    python scripts/figures/make_slide_figures.py  ->  figures/final/*.png
"""
from __future__ import annotations

import json
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

ROOT = Path(__file__).resolve().parents[2]
RAW = ROOT / "results" / "raw"
FINAL, EVID, FAIR = RAW / "final", RAW / "evidence", RAW / "fairness"
OUT = ROOT / "figures" / "final"
OUT.mkdir(parents=True, exist_ok=True)

ARMS = ["multi", "quantum", "linear", "mlp"]
LABEL = {"multi": "Quantum, 10-qubit readout", "quantum": "Quantum, 1-qubit readout",
         "linear": "Linear", "mlp": "MLP (65k parameters)"}
COLOR = {"multi": "#A3226A", "quantum": "#D98FBB", "linear": "#0B5C73", "mlp": "#7A8791"}
LS = {"multi": "-", "quantum": "--", "linear": "-", "mlp": "-"}
plt.rcParams.update({"font.family": "DejaVu Sans", "font.size": 13, "axes.titlesize": 14,
                     "axes.labelsize": 13, "legend.fontsize": 12, "xtick.labelsize": 12,
                     "ytick.labelsize": 12, "axes.spines.top": False, "axes.spines.right": False,
                     "axes.grid": True, "grid.alpha": 0.25, "grid.linewidth": 0.8})


def load(src):
    d = {a: [json.load(open(f)) for f in sorted(src.glob(f"{a}_s*.json"))] for a in ARMS}
    return {a: v for a, v in d.items() if v}


F, E = load(FINAL), load(EVID)


def save(fig, name):
    fig.savefig(OUT / f"{name}.png", dpi=200, bbox_inches="tight", facecolor="white")
    plt.close(fig)


def band(ax, x, vals, a, marker="o"):
    v = np.asarray(vals, float)
    m, s = v.mean(0), v.std(0)
    ax.plot(x, m, LS[a], color=COLOR[a], lw=2.6, marker=marker, ms=6, label=LABEL[a])
    ax.fill_between(x, m - s, m + s, color=COLOR[a], alpha=0.14, lw=0)


def roll(x, k=6):
    return np.convolve(np.asarray(x, float), np.ones(k) / k, mode="valid")


# ---- 1. memory over distance + old vs new facts ---------------------------------------
fig, ax = plt.subplots(1, 2, figsize=(13, 4.7), gridspec_kw={"width_ratios": [1.25, 1]})
bands = E["multi"][0]["bands"]
names = [f"{lo}-{hi}" if hi < 9999 else f"over {lo}" for lo, hi in bands]
x = np.arange(len(bands))
for a, runs in E.items():
    ax[0].plot(x, np.mean([r["by_distance"]["model"] for r in runs], 0), LS[a], color=COLOR[a],
               lw=2.6, marker="o", ms=6, label=LABEL[a])
win = np.mean([r["by_distance"]["window"] for a in E for r in E[a]], 0)
ax[0].plot(x, win, color="black", ls="--", lw=2.2, marker="s", ms=6, label="No memory (last 100 turns only)")
ax[0].set_xticks(x, names)
ax[0].set(xlabel="How many turns ago the needed fact was stated", ylabel="MRR (test split)",
          ylim=(-0.03, 1.06), title="Recalling facts from far back")
ax[0].legend(frameon=False, loc="lower left", fontsize=10.5)
w = 0.3
for i, a in enumerate(E):
    for j in (0, 1):
        key = ("unchanged", "updated")[j]
        v = [r["split"][key]["mrr"] for r in E[a]]
        xx = i + (j - 0.5) * 0.34
        ax[1].bar(xx, np.mean(v), w, yerr=np.std(v), capsize=2.5, color=COLOR[a], alpha=0.5 if j == 0 else 1)
        ax[1].text(xx, np.mean(v) + 0.045, f"{np.mean(v):.2f}", ha="center", fontsize=9.5)
ax[1].set_xticks(range(len(E)), ["Quantum\n10-qubit", "Quantum\n1-qubit", "Linear", "MLP"])
ax[1].set(ylim=(0, 1.12), ylabel="MRR (test split)",
          title="Without (light) and with (dark) an outdated\nfact among the candidates")
fig.tight_layout()
save(fig, "memory_distance")

# ---- 2. forgetting matrices ---------------------------------------------------------
fig = plt.figure(figsize=(13, 5.9))
gs = fig.add_gridspec(2, 4, height_ratios=[1.1, 1], hspace=0.62, wspace=0.3)
short = {"multi": "Quantum, 10-qubit", "quantum": "Quantum, 1-qubit", "linear": "Linear", "mlp": "MLP"}
for i, a in enumerate(E):
    ax = fig.add_subplot(gs[0, i])
    mat = E[a][0]["matrix"]
    ts = sorted(int(k) for k in mat)
    M = np.full((len(ts), 120), np.nan)
    for r, t in enumerate(ts):
        M[r, :t + 1] = mat[str(t)]
    im = ax.imshow(M, aspect="auto", cmap="viridis", vmin=0.5, vmax=1.0, extent=[0, 120, len(ts), 0],
                   interpolation="nearest")
    ax.set_yticks(np.arange(len(ts))[::2] + 0.5, [t + 1 for t in ts][::2], fontsize=10)
    ax.set_xticks([0, 40, 80, 120])
    ax.set_title(short[a], fontsize=13)
    ax.set_xlabel("task re-tested", fontsize=11)
    if i == 0:
        ax.set_ylabel("tasks learned", fontsize=11)
    ax.grid(False)
cax = fig.add_axes([0.925, 0.60, 0.012, 0.28])
fig.colorbar(im, cax=cax).set_label("MRR", fontsize=11)
ax = fig.add_subplot(gs[1, :])
for a, runs in E.items():
    ts = sorted(int(k) for k in runs[0]["matrix"])
    band(ax, [t + 1 for t in ts], [[np.mean(r["matrix"][str(t)]) for t in ts] for r in runs], a)
ax.set(xlabel="tasks learned so far", ylabel="mean MRR, all past tasks", ylim=(0.78, 1.01))
ax.legend(frameon=False, ncol=4, loc="lower center", bbox_to_anchor=(0.5, 1.0), fontsize=11)
save(fig, "forgetting_matrix")

# ---- 3. learning new things -------------------------------------------------------------
fig, ax = plt.subplots(1, 2, figsize=(13, 4.6), gridspec_kw={"width_ratios": [1.45, 1]})
for a, runs in E.items():
    pre = np.mean([roll(r["pre"]) for r in runs], 0)
    ax[0].plot(np.arange(len(pre)) + 3, pre, LS[a], color=COLOR[a], lw=2.5, label=LABEL[a])
ax[0].set(xlabel="task index", ylabel="MRR on a task not yet trained on", ylim=(0, 1.03),
          title="Score on each task before training on it")
ax[0].legend(frameon=False, loc="lower right", fontsize=10.5)
for i, a in enumerate(E):
    v = [r["split"]["updated"]["mrr"] for r in E[a]]
    ax[1].bar(i, np.mean(v), 0.62, yerr=np.std(v), capsize=3, color=COLOR[a])
    ax[1].text(i, np.mean(v) - 0.09, f"{np.mean(v):.2f}", ha="center", fontsize=12, fontweight="bold",
               color="#222" if a == "quantum" else "white")
cosv = np.mean([r["split"]["updated"]["cosine_mrr"] for a in E for r in E[a]])
ax[1].axhline(cosv, color="black", ls=":", lw=1.4)
ax[1].text(-0.45, 1.065, f"dotted line: plain similarity search, {cosv:.2f}", ha="left", fontsize=10.5, color="#333")
ax[1].set_xticks(range(len(E)), ["Quantum\n10-qubit", "Quantum\n1-qubit", "Linear", "MLP"])
ax[1].set(ylim=(0, 1.14), ylabel="MRR, probes with an outdated fact", title="Outdated fact among the candidates")
fig.tight_layout()
save(fig, "learning_new")

# ---- 4. plasticity (reversal learning) -------------------------------------------------
CK = [0, 29, 59, 89, 119]
fig, ax = plt.subplots(1, 2, figsize=(13, 4.6), gridspec_kw={"width_ratios": [1.15, 1]})
for a, runs in F.items():
    band(ax[0], [c + 1 for c in CK], [[r["reversal"][str(c)]["40"] for c in CK] for r in runs], a)
ax[0].set(xlabel="tasks learned so far", ylabel="MRR on the reversed rule", ylim=(0, 1.02),
          title="Learning a reversed rule, 40 gradient steps")
ax[0].legend(frameon=False, loc="lower left", fontsize=10.5)
steps = [0, 10, 20, 40]
for a, runs in F.items():
    v = np.array([[r["reversal"]["119"][str(k)] for k in steps] for r in runs])
    ax[1].errorbar(steps, v.mean(0), v.std(0), color=COLOR[a], ls=LS[a], lw=2.5, marker="o", capsize=3)
ax[1].set(xlabel="gradient steps on the reversed rule", ylabel="MRR", ylim=(0, 1.02),
          title="Adaptation speed after 120 tasks")
fig.tight_layout()
save(fig, "plasticity")

# ---- 5. both together --------------------------------------------------------------------
def S_P(runs, c):
    return (np.array([r["retention_early"][str(c)] for r in runs]),
            np.array([r["reversal"][str(c)]["40"] for r in runs]))


fig, ax = plt.subplots(1, 2, figsize=(13, 5.0), gridspec_kw={"width_ratios": [1.2, 1]})
for a, runs in F.items():
    pts = np.array([[S_P(runs, c)[0].mean(), S_P(runs, c)[1].mean()] for c in CK[1:]])
    ax[0].plot(pts[:, 0], pts[:, 1], LS[a], color=COLOR[a], lw=1.8, alpha=0.85)
    for i in range(len(pts) - 1):
        ax[0].annotate("", xy=pts[i + 1], xytext=pts[i], arrowprops=dict(arrowstyle="->", color=COLOR[a], lw=1.8))
    ax[0].scatter(pts[:, 0], pts[:, 1], s=36, color=COLOR[a], zorder=3)
    ax[0].scatter(*pts[-1], s=190, color=COLOR[a], edgecolor="black", zorder=4, label=LABEL[a])
ax[0].scatter([1.0], [1.0], marker="*", s=330, color="#E0A020", zorder=5)
ax[0].text(0.85, 0.94, "goal: high on both", color="#9A6B00", fontsize=11.5)
ax[0].set(xlabel="Stability: MRR on the first 30 tasks", ylabel="Plasticity: MRR on the reversed rule",
          xlim=(0.83, 1.02), ylim=(0, 1.04), title="Path of each model over 120 tasks")
ax[0].legend(frameon=False, loc="lower left", fontsize=10)
w = 0.2
for i, a in enumerate(F):
    H = []
    for c in CK[1:]:
        S, P = S_P(F[a], c)
        H.append((2 * S * P / (S + P + 1e-9)).mean())
    ax[1].bar(np.arange(len(H)) + i * w - 0.3, H, w, color=COLOR[a])
ax[1].set_xticks(range(4), [f"{c + 1}" for c in CK[1:]])
ax[1].set(xlabel="tasks learned so far", ylabel="Combined score", ylim=(0, 1.02),
          title="Combined: harmonic mean of the two")
fig.tight_layout()
save(fig, "dilemma")

# ---- 6. fairness -------------------------------------------------------------------------
G = {a: [json.load(open(f)) for f in sorted(FAIR.glob(f"{a}_s*.json"))] for a in ["multi", "linear", "mlp"]}
fig, ax = plt.subplots(figsize=(7.2, 4.6))
nm = {"multi": "Quantum\n10-qubit", "linear": "Linear", "mlp": "MLP"}
for i, a in enumerate(G):
    for j, key in enumerate(("best40", "best80")):
        v = np.array([r[key] for r in G[a]])
        ax.bar(i + (j - 0.5) * 0.36, v.mean(), 0.34, yerr=v.std(), capsize=3, color=COLOR[a],
               alpha=0.55 if j == 0 else 1)
        ax.text(i + (j - 0.5) * 0.36, v.mean() + v.std() + 0.025, f"{v.mean():.2f}", ha="center", fontsize=11)
ax.set_xticks(range(3), [nm[a] for a in G])
ax.set(ylim=(0, 1.2), ylabel="MRR on the reversed rule", title="Best learning rate for each model",
       xlabel="light bars: 40 steps      dark bars: 80 steps")
fig.tight_layout()
save(fig, "fairness")
print("ok")
