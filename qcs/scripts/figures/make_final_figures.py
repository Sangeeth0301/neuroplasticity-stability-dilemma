"""Figures + summary table for the final stability / plasticity comparison.

    python scripts/figures/make_final_figures.py   (reads results/raw/final/*.json)
"""
from __future__ import annotations

import json
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

ROOT = Path(__file__).resolve().parents[2]
SRC = ROOT / "results" / "raw" / "final"
OUT = ROOT / "figures" / "analysis"
OUT.mkdir(parents=True, exist_ok=True)

ARMS = ["multi", "quantum", "linear", "mlp"]
LABEL = {"multi": "Quantum (multi-qubit readout)", "quantum": "Quantum (single-qubit readout)",
         "linear": "Linear (param-matched)", "mlp": "MLP (65k params)"}
COLOR = {"multi": "#a3226a", "quantum": "#d98fbb", "linear": "#0b5c73", "mlp": "#7a8791"}
STYLE = {"multi": "-", "quantum": "--", "linear": "-", "mlp": "-"}
CK = [0, 29, 59, 89, 119]

plt.rcParams.update({"font.size": 11, "axes.spines.top": False, "axes.spines.right": False,
                     "axes.grid": True, "grid.alpha": 0.25, "figure.dpi": 150})

R = {a: [json.load(open(f)) for f in sorted(SRC.glob(f"{a}_s*.json"))] for a in ARMS}
R = {a: v for a, v in R.items() if v}


def ms(vals):
    v = np.asarray(vals, float)
    return v.mean(0), v.std(0)


def band(ax, x, vals, a):
    m, s = ms(vals)
    ax.plot(x, m, STYLE[a], color=COLOR[a], lw=2.2, marker="o", ms=5, label=LABEL[a])
    ax.fill_between(x, m - s, m + s, color=COLOR[a], alpha=0.15, lw=0)


def save(fig, name):
    fig.savefig(OUT / f"{name}.png", bbox_inches="tight", facecolor="white")
    fig.savefig(OUT / f"{name}.svg", bbox_inches="tight", facecolor="white")
    plt.close(fig)


def S_P(runs, c):
    S = np.array([r["retention_early"][str(c)] for r in runs])
    P = np.array([r["reversal"][str(c)]["40"] for r in runs])
    return S, P


# ---- FIN1 stability ---------------------------------------------------------
fig, ax = plt.subplots(1, 2, figsize=(13, 4.6))
x = [c + 1 for c in CK[1:]]
for a, runs in R.items():
    band(ax[0], x, [[r["retention_early"][str(c)] for c in CK[1:]] for r in runs], a)
ax[0].set(xlabel="tasks learned so far", ylabel="MRR on the FIRST 30 tasks",
          title="Stability: is old knowledge still there?", ylim=(0.7, 1.01))
ax[0].legend(frameon=False, fontsize=9, loc="lower right")
for a, runs in R.items():
    prof = np.array([r["final_profile"] for r in runs])
    k = 8
    sm = np.array([np.convolve(p, np.ones(k) / k, mode="valid") for p in prof])
    m = sm.mean(0)
    ax[1].plot(np.arange(len(m)) + k // 2, m, STYLE[a], color=COLOR[a], lw=2.2, label=LABEL[a])
ax[1].set(xlabel="task index (0 = oldest)", ylabel="MRR of the FINAL model on that task",
          title="Final model, every past task (8-task moving mean)", ylim=(0.7, 1.01))
save(fig, "FIN1_stability")

# ---- FIN2 plasticity --------------------------------------------------------
fig, ax = plt.subplots(1, 3, figsize=(17, 4.6))
for a, runs in R.items():
    band(ax[0], [c + 1 for c in CK], [[r["reversal"][str(c)]["40"] for c in CK] for r in runs], a)
ax[0].set(xlabel="tasks learned so far", ylabel="reversal MRR after 40 gradient steps",
          title="Plasticity: can it still learn a NEW rule?", ylim=(0, 1))
for a, runs in R.items():
    tr = np.array([np.array(r["trF"]) / r["trF"][0] for r in runs])
    m = tr.mean(0)
    ax[1].plot(np.arange(len(m)) + 1, m, STYLE[a], color=COLOR[a], lw=2.2)
ax[1].set(yscale="log", xlabel="task", ylabel="Tr(F) / Tr(F) at task 0",
          title="Learning capacity Tr(F) — noisy, secondary metric")
ax[1].axhline(1, color="k", lw=0.8, ls=":")
steps = [0, 10, 20, 40]
for a, runs in R.items():
    m, s = ms([[r["reversal"]["119"][str(k)] for k in steps] for r in runs])
    ax[2].errorbar(steps, m, s, color=COLOR[a], ls=STYLE[a], lw=2.2, marker="o", capsize=3)
ax[2].set(xlabel="gradient steps on the new rule", ylabel="reversal MRR",
          title="After 120 tasks: adaptation speed", ylim=(0, 1))
h, l = ax[0].get_legend_handles_labels()
fig.legend(h, l, loc="lower center", ncol=4, frameon=False, fontsize=10, bbox_to_anchor=(0.5, -0.06))
save(fig, "FIN2_plasticity")

# ---- FIN3 the dilemma, combined ---------------------------------------------
fig, ax = plt.subplots(1, 2, figsize=(14, 5.6), gridspec_kw={"width_ratios": [1.25, 1]})
for a, runs in R.items():
    pts = np.array([[S_P(runs, c)[0].mean(), S_P(runs, c)[1].mean()] for c in CK[1:]])
    ax[0].plot(pts[:, 0], pts[:, 1], STYLE[a], color=COLOR[a], lw=1.6, alpha=0.8)
    for i in range(len(pts) - 1):
        ax[0].annotate("", xy=pts[i + 1], xytext=pts[i],
                       arrowprops=dict(arrowstyle="->", color=COLOR[a], lw=1.6))
    ax[0].scatter(pts[:, 0], pts[:, 1], s=40, color=COLOR[a], zorder=3)
    ax[0].scatter(*pts[-1], s=170, color=COLOR[a], edgecolor="k", zorder=4, label=LABEL[a])
ax[0].scatter([1], [1], marker="*", s=300, color="#e0a020", zorder=5)
ax[0].annotate("ideal: stable AND plastic", (1, 1), (0.84, 0.93), fontsize=10, color="#9a6b00")
ax[0].set(xlabel="STABILITY  — MRR on the first 30 tasks",
          ylabel="PLASTICITY — reversal MRR after 40 steps",
          title="The stability–plasticity dilemma: path of each arm over 120 tasks\n"
                "(small dots = after 30, 60, 90 tasks; large dot = after 120)",
          xlim=(0.82, 1.02), ylim=(0, 1.02))
ax[0].legend(frameon=False, fontsize=9, loc="lower left")
w = 0.2
for i, a in enumerate(R):
    H = []
    for c in CK[1:]:
        S, P = S_P(R[a], c)
        H.append((2 * S * P / (S + P + 1e-9)).mean())
    ax[1].bar(np.arange(len(H)) + i * w - 0.3, H, w, color=COLOR[a], label=LABEL[a])
ax[1].set_xticks(range(len(CK[1:])), [f"{c + 1} tasks" for c in CK[1:]])
ax[1].set(ylabel="combined = harmonic mean(stability, plasticity)",
          title="Combined score over training", ylim=(0, 1))
save(fig, "FIN3_dilemma_combined")

# ---- summary table ----------------------------------------------------------
rows, summ = [], {}
for a, runs in R.items():
    def g(f):
        return np.array([f(r) for r in runs])
    S, P = S_P(runs, 119)
    d = {
        "train_mrr": g(lambda r: r["train"]["mrr"]), "test_mrr": g(lambda r: r["test"]["mrr"]),
        "test_rank1": g(lambda r: r["test"]["rank1"]),
        "test_stale": g(lambda r: r["test"]["stale_above_current"]),
        "bwt": g(lambda r: r["bwt"]), "retention_end": S, "reversal_end": P,
        "reversal_start": g(lambda r: r["reversal"]["0"]["40"]),
        "trF_ratio": g(lambda r: r["trF"][-1] / r["trF"][0]),
        "combined_end": 2 * S * P / (S + P + 1e-9),
    }
    summ[a] = {k: [float(v.mean()), float(v.std())] for k, v in d.items()}
    summ[a].update(n_seeds=len(runs), params=runs[0]["params"])
    rows.append([LABEL[a]] + [f"{summ[a][k][0]:.3f} ± {summ[a][k][1]:.3f}" for k in
                              ("test_mrr", "retention_end", "reversal_end", "combined_end")] +
                [f"{summ[a]['trF_ratio'][0]:.2f}x"])
json.dump(summ, open(ROOT / "results" / "final_summary.json", "w"), indent=1)

fig, ax = plt.subplots(figsize=(14, 2.6 + 0.55 * len(rows)))
ax.axis("off")
tb = ax.table(cellText=rows, colLabels=["arm (mean ± sd over seeds)", "test MRR\n(sealed split)",
              "STABILITY\nold-task MRR", "PLASTICITY\nreversal MRR", "COMBINED\n(harmonic mean)",
              "capacity\nTr(F) end/start"], loc="center", cellLoc="center")
tb.auto_set_font_size(False)
tb.set_fontsize(10.5)
tb.scale(1, 2.1)
for (r, c), cell in tb.get_celld().items():
    cell.set_edgecolor("#dddddd")
    if r == 0:
        cell.set_facecolor("#eef1f3")
        cell.set_text_props(weight="bold")
ax.set_title("Final scorecard — all numbers measured after 120 tasks", loc="left", fontweight="bold")
save(fig, "FIN4_scorecard")

print("\n".join(" | ".join(r) for r in rows))
