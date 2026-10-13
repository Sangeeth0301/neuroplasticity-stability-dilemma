"""Figures for 'learns new things / remembers old things'. Reads results/raw/evidence/*.json.

    python scripts/figures/make_evidence_figures.py
"""
from __future__ import annotations

import json
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

ROOT = Path(__file__).resolve().parents[2]
SRC, OUT = ROOT / "results" / "raw" / "evidence", ROOT / "figures" / "analysis"
ARMS = ["multi", "quantum", "linear", "mlp"]
LABEL = {"multi": "Quantum (multi-qubit)", "quantum": "Quantum (single-qubit)",
         "linear": "Linear", "mlp": "MLP (65k)"}
COLOR = {"multi": "#a3226a", "quantum": "#d98fbb", "linear": "#0b5c73", "mlp": "#7a8791"}
LS = {"multi": "-", "quantum": "--", "linear": "-", "mlp": "-"}
plt.rcParams.update({"font.size": 11, "axes.spines.top": False, "axes.spines.right": False,
                     "axes.grid": True, "grid.alpha": 0.25, "figure.dpi": 150})

R = {a: [json.load(open(f)) for f in sorted(SRC.glob(f"{a}_s*.json"))] for a in ARMS}
R = {a: v for a, v in R.items() if v}


def save(fig, name):
    fig.savefig(OUT / f"{name}.png", bbox_inches="tight", facecolor="white")
    fig.savefig(OUT / f"{name}.svg", bbox_inches="tight", facecolor="white")
    plt.close(fig)


def roll(x, k=6):
    x = np.asarray(x, float)
    return np.convolve(x, np.ones(k) / k, mode="valid")


# ---- EV1: learns NEW things ---------------------------------------------------------
fig, ax = plt.subplots(1, 2, figsize=(14, 4.8), gridspec_kw={"width_ratios": [1.4, 1]})
for a, runs in R.items():
    pre = np.mean([roll(r["pre"]) for r in runs], axis=0)
    ax[0].plot(np.arange(len(pre)) + 3, pre, LS[a], color=COLOR[a], lw=2.2, label=LABEL[a])
ax[0].set(xlabel="task index (the task the model is about to meet)",
          ylabel="MRR on a task it has NEVER trained on",
          title="Learning new things: score on each task BEFORE training on it",
          ylim=(0, 1.03))
ax[0].legend(frameon=False, fontsize=9, loc="lower right")
ax[0].annotate("starts untrained,\nlearns within ~15 tasks", (6, 0.2), (28, 0.3), fontsize=9,
               arrowprops=dict(arrowstyle="->", color="#555"))
w = 0.2
for i, a in enumerate(R):
    v = [r["split"]["updated"]["mrr"] for r in R[a]]
    ax[1].bar(i, np.mean(v), 0.6, yerr=np.std(v), capsize=3, color=COLOR[a])
    ax[1].text(i, np.mean(v) + 0.03, f"{np.mean(v):.2f}", ha="center", fontsize=9)
cosv = np.mean([r["split"]["updated"]["cosine_mrr"] for a in R for r in R[a]])
ax[1].axhline(cosv, color="k", ls=":", lw=1.2)
ax[1].text(len(R) - 0.5, cosv - 0.07, f"no-learning cosine {cosv:.2f}", ha="right", fontsize=9)
ax[1].set_xticks(range(len(R)), [LABEL[a].replace(" (", "\n(") for a in R], fontsize=9)
ax[1].set(ylim=(0, 1.1), ylabel="MRR on probes with an outdated fact (test split)",
          title="Probes with an outdated fact among the candidates")
save(fig, "EV1_learns_new")

# ---- EV2: remembers OLD things --------------------------------------------------------
fig = plt.figure(figsize=(17, 8.6))
gs = fig.add_gridspec(2, 4, height_ratios=[1.15, 1], hspace=0.38, wspace=0.28)
for i, a in enumerate(R):
    ax = fig.add_subplot(gs[0, i])
    mat = R[a][0]["matrix"]
    ts = sorted(int(k) for k in mat)
    M = np.full((len(ts), 120), np.nan)
    for r, t in enumerate(ts):
        M[r, :t + 1] = mat[str(t)]
    im = ax.imshow(M, aspect="auto", cmap="viridis", vmin=0.5, vmax=1.0,
                   extent=[0, 120, len(ts), 0], interpolation="nearest")
    ax.set_yticks(np.arange(len(ts)) + 0.5, [t + 1 for t in ts], fontsize=8)
    ax.set(xlabel="task being re-tested (0 = oldest)", title=LABEL[a] + "  (seed 7)")
    if i == 0:
        ax.set_ylabel("tasks learned so far")
    ax.grid(False)
cax = fig.add_axes([0.915, 0.57, 0.01, 0.3])
fig.colorbar(im, cax=cax, label="MRR")
ax = fig.add_subplot(gs[1, :])
for a, runs in R.items():
    ts = sorted(int(k) for k in runs[0]["matrix"])
    vals = np.array([[np.mean(r["matrix"][str(t)]) for t in ts] for r in runs])
    m, s = vals.mean(0), vals.std(0)
    ax.plot([t + 1 for t in ts], m, LS[a], color=COLOR[a], lw=2.2, marker="o", label=LABEL[a])
    ax.fill_between([t + 1 for t in ts], m - s, m + s, color=COLOR[a], alpha=0.15, lw=0)
ax.set(xlabel="tasks learned so far", ylabel="mean MRR over ALL past tasks",
       title="Remembering the old: average score on everything learned before "
             "(every past task re-tested, mean ± sd over 3 seeds)", ylim=(0.7, 1.01))
ax.legend(frameon=False, ncol=4, loc="lower right", fontsize=9)
save(fig, "EV2_remembers_old")

# ---- EV3: memory over distance + old-vs-new knowledge ------------------------------
fig, ax = plt.subplots(1, 2, figsize=(14.5, 5), gridspec_kw={"width_ratios": [1.2, 1]})
bands = R[next(iter(R))][0]["bands"]
names = [f"{lo}–{hi}" if hi < 9999 else f">{lo}" for lo, hi in bands]
x = np.arange(len(bands))
for a, runs in R.items():
    m = np.mean([r["by_distance"]["model"] for r in runs], axis=0)
    ax[0].plot(x, m, LS[a], color=COLOR[a], lw=2.4, marker="o", label=LABEL[a])
win = np.mean([r["by_distance"]["window"] for a in R for r in R[a]], axis=0)
cosd = np.mean([r["by_distance"]["cosine"] for a in R for r in R[a]], axis=0)
W = R[next(iter(R))][0]["window"]
ax[0].plot(x, win, "k--", lw=2, marker="s", label=f"no memory: last {W} turns only")
ax[0].plot(x, cosd, color="#999", ls=":", lw=2, marker="^", label="memory + cosine (no learning)")
ax[0].set_xticks(x, names)
ax[0].set(xlabel="how many turns ago the needed fact was stated", ylabel="MRR (sealed test split)",
          title="Remembering the old: facts from hundreds of turns back", ylim=(-0.03, 1.05))
ax[0].legend(frameon=False, fontsize=8.5, loc="lower left")
w = 0.19
for i, a in enumerate(R):
    for j, (k, col) in enumerate((("unchanged", COLOR[a]), ("updated", COLOR[a]))):
        v = [r["split"][k]["mrr"] for r in R[a]]
        ax[1].bar(i + (j - 0.5) * w * 1.05, np.mean(v), w, yerr=np.std(v), capsize=2,
                  color=col, alpha=0.55 if j == 0 else 1.0)
        ax[1].text(i + (j - 0.5) * w * 1.05, np.mean(v) + 0.025, f"{np.mean(v):.2f}", ha="center", fontsize=8)
ax[1].set_xticks(range(len(R)), [LABEL[a].replace(" (", "\n(") for a in R], fontsize=9)
ax[1].set(ylim=(0, 1.12), ylabel="MRR (sealed test split)",
          title="Without (light) and with (dark) an outdated fact\namong the candidates")
save(fig, "EV3_memory_over_distance")

# ---- table ----------------------------------------------------------------------------
rows = []
for a, runs in R.items():
    g = lambda f: (np.mean([f(r) for r in runs]), np.std([f(r) for r in runs]))
    rows.append((a, g(lambda r: r["split"]["unchanged"]["mrr"]), g(lambda r: r["split"]["updated"]["mrr"]),
                 g(lambda r: r["split"]["updated"]["stale_above"]),
                 g(lambda r: r["by_distance"]["model"][-1]),
                 g(lambda r: np.mean(r["matrix"]["119"])), g(lambda r: np.mean(r["pre"][20:]))))
print("arm | old-memory MRR | updated-fact MRR | stale-above | MRR at >400 turns | all-past-tasks MRR@120 | "
      "unseen-task MRR (task>=20)")
for a, *v in rows:
    print(a, " | ".join(f"{m:.3f}±{s:.3f}" for m, s in v))
print("window baseline by band:", np.round(win, 3), "cosine memory:", np.round(cosd, 3))
json.dump({a: {"unchanged": v[0], "updated": v[1], "stale_above": v[2], "far": v[3],
               "all_past": v[4], "unseen": v[5]} for a, *v in rows},
          open(ROOT / "results" / "evidence_summary.json", "w"), indent=1)
