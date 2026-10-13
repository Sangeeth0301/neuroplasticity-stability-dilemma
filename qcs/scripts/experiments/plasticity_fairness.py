"""Fairness check for the reversal-learning result.

    python scripts/experiments/plasticity_fairness.py ARM SEED

The headline plasticity test fine-tunes every arm at its own training learning rate. A
critic could say the classical arms simply need a different step size to re-learn. So:
train the arm normally for 120 tasks, then fine-tune a copy on the reversal rule at EVERY
learning rate in a grid and keep the BEST one per arm. If the classical arms still cannot
recover, the loss of plasticity is not a learning-rate artefact.
"""
from __future__ import annotations

import copy
import json
import sys
from dataclasses import replace
from pathlib import Path

import numpy as np
import torch

from qcs import paths
from qcs.classical.scorers import build_scorer
from qcs.config import Config
from qcs.eval.metrics import evaluate_probes
from qcs.logging_ import RunLogger
from qcs.quantum.circuit import QRE
from qcs.quantum.hybrid import MultiReadoutQRE
from qcs.seeding import seed_everything
from qcs.train.dataset import ScoringDataset
from qcs.train.loop import TaskStream, _flatten_task, run_continual

arm, seed = sys.argv[1], int(sys.argv[2])
OUT = Path(__file__).resolve().parents[2] / "results" / "raw" / "fairness"
OUT.mkdir(parents=True, exist_ok=True)
cfg = Config.load("tuned")
cfg.seed = seed
cfg.validate()
seed_everything(seed)
kind = "quantum" if arm in ("quantum", "multi") else arm
if arm == "quantum":
    model = QRE(cfg, seed=seed)
elif arm == "multi":
    model = MultiReadoutQRE(cfg, seed=seed)
else:
    model = build_scorer(arm, cfg, seed=seed)

ds = ScoringDataset.load(cfg)
stream = TaskStream(ds.examples, 6)
log = RunLogger.create(cfg, f"fair_{arm}_s{seed}", root=paths.RUNS / "scratch")
run_continual(model, stream, cfg, log, arm=f"{arm}_s{seed}", kind=kind, seed=seed, verbose=False)

rev = [replace(e, y=e.is_stale.copy(), is_stale=e.y.copy())
       for e in ds.examples if e.is_stale.sum() > 0 and e.y.sum() > 0]
fit, held = rev[0::2], rev[1::2]
Xr, yr = _flatten_task(fit, 8, np.random.default_rng(seed))


def mrr_of(m):
    return evaluate_probes(lambda ex: m.score(ex.X), held, ks=(1,))["mrr"]


GRID = [0.001, 0.003, 0.01, 0.03, 0.1, 0.3]
out = {"arm": arm, "seed": seed, "before": mrr_of(model), "by_lr": {}}
for lr in GRID:
    m = copy.deepcopy(model)
    opt = torch.optim.Adam(m.parameters(), lr=lr)
    r = np.random.default_rng(seed + 1)
    curve = {}
    for step in range(1, 81):
        idx = r.choice(len(Xr), 32, replace=False)
        opt.zero_grad()
        loss = torch.nn.functional.binary_cross_entropy(
            m(Xr[idx]).clamp(1e-6, 1 - 1e-6), torch.as_tensor(yr[idx], dtype=torch.float32))
        loss.backward()
        opt.step()
        m.wrap_angles()
        if step in (40, 80):
            curve[step] = mrr_of(m)
    out["by_lr"][str(lr)] = curve
out["best40"] = max(v[40] for v in out["by_lr"].values())
out["best80"] = max(v[80] for v in out["by_lr"].values())
json.dump(out, open(OUT / f"{arm}_s{seed}.json", "w"))
print("DONE", arm, seed, "best@40", round(out["best40"], 3), "best@80", round(out["best80"], 3), flush=True)
