"""Final comparison: stability, plasticity (reversal learning) and generalisation.

    python scripts/experiments/final_eval.py ARM SEED          ARM in {quantum, multi, linear, mlp}

Measures, for ONE arm and ONE seed, over the same 120-task stream:

  stability   retention  — MRR of the model on the FIRST 30 tasks, measured at
              checkpoints t = 0, 30, 60, 90, 119 (does old knowledge survive?)
  plasticity  reversal   — at the same checkpoints, a COPY of the model is fine-tuned on a
              rule it has never seen (the stale fact, not the current one, is now
              correct: reversal learning) and scored on held-out reversal probes after
              0 / 10 / 20 / 40 gradient steps (can it still learn something new?)
  capacity    Tr(F)      — already logged by run_continual
  generalise  sealed test split MRR, per-probe, for the paired bootstrap
"""
from __future__ import annotations

import copy
import json
import sys
import time
from dataclasses import replace
from pathlib import Path

import numpy as np
import torch

from qcs import paths
from qcs.classical.scorers import build_scorer
from qcs.config import Config
from qcs.eval.metrics import evaluate_probes, mrr
from qcs.logging_ import RunLogger
from qcs.quantum.circuit import QRE
from qcs.quantum.hybrid import MultiReadoutQRE
from qcs.seeding import seed_everything
from qcs.train.dataset import ProbeExample, ScoringDataset
from qcs.train.loop import TaskStream, _flatten_task, run_continual

arm, seed = sys.argv[1], int(sys.argv[2])
OUT = Path(__file__).resolve().parents[2] / "results" / "raw" / "final"
OUT.mkdir(parents=True, exist_ok=True)

cfg = Config.load("tuned")
cfg.seed = seed
cfg.validate()
seed_everything(seed)
kind = "quantum" if arm in ("quantum", "multi") else arm
lr = cfg.train.lr_by_arm[kind]


def make():
    if arm == "quantum":
        return QRE(cfg, seed=seed)
    if arm == "multi":
        return MultiReadoutQRE(cfg, seed=seed)
    return build_scorer(arm, cfg, seed=seed)


def score_mrr(model, exs):
    return evaluate_probes(lambda ex: model.score(ex.X), exs, ks=(1,))["mrr"] if exs else float("nan")


ds = ScoringDataset.load(cfg)
stream = TaskStream(ds.examples, 6)
tasks = stream.tasks()
early = [e for t in tasks[:30] for e in t]

# reversal task: the superseded fact is now the correct answer
rev = []
for e in ds.examples:
    if e.is_stale.sum() > 0 and e.y.sum() > 0:
        rev.append(replace(e, y=e.is_stale.copy(), is_stale=e.y.copy()))
rev_fit, rev_test = rev[0::2], rev[1::2]
rng = np.random.default_rng(seed)
Xr, yr = _flatten_task(rev_fit, 8, rng)

CHECK = {0, 29, 59, 89, 119}
retention, reversal = {}, {}


def reversal_curve(model):
    m = copy.deepcopy(model)
    opt = torch.optim.Adam(m.parameters(), lr=lr)
    bce = torch.nn.BCELoss()
    out = {0: score_mrr(m, rev_test)}
    r = np.random.default_rng(seed + 1)
    for step in range(1, 41):
        idx = r.choice(len(Xr), 32, replace=False)
        opt.zero_grad()
        loss = bce(m(Xr[idx]).clamp(1e-6, 1 - 1e-6), torch.as_tensor(yr[idx], dtype=torch.float32))
        loss.backward()
        opt.step()
        m.wrap_angles()
        if step in (10, 20, 40):
            out[step] = score_mrr(m, rev_test)
    return out


def hook(t, model):
    if t in CHECK:
        retention[t] = score_mrr(model, early)
        reversal[t] = reversal_curve(model)
        print(f"[{arm} s{seed}] t={t:3d} early-task MRR {retention[t]:.3f}  "
              f"reversal@0/10/20/40 {[round(v, 3) for v in reversal[t].values()]}", flush=True)


model = make()
log = RunLogger.create(cfg, f"final_{arm}_s{seed}", root=paths.RUNS / "scratch")
t0 = time.time()
res = run_continual(model, stream, cfg, log, arm=f"{arm}_s{seed}", kind=kind, seed=seed,
                    verbose=False, on_task_end=hook)

# final retention profile: every task re-scored by the FINAL model
profile = [score_mrr(model, t) for t in tasks]
train_metrics = evaluate_probes(lambda ex: model.score(ex.X), ds.examples)

test_path = paths.DATA_PROCESSED / "scoring_test.npz"
test_ds = ScoringDataset.load(cfg, split="test")
test_metrics = evaluate_probes(lambda ex: model.score(ex.X), test_ds.examples)
per_probe = {}
for ex in test_ds.examples:
    per_probe.setdefault(ex.thread_id, []).append(float(mrr(model.score(ex.X), ex.y)))

hist = res.history
out = {
    "arm": arm, "seed": seed, "params": int(model.n_circuit_params),
    "train": train_metrics, "test": test_metrics,
    "bwt": res.bwt(), "final_avg": res.final_average(),
    "plasticity_index": res.plasticity_index(),
    "trF": [h["tr_fim"] for h in hist],
    "acc_diag": [h["acc"] for h in hist],
    "retention_early": retention, "reversal": reversal,
    "final_profile": profile, "A": res.A.tolist(),
    "test_per_probe": per_probe, "seconds": time.time() - t0,
}
json.dump(out, open(OUT / f"{arm}_s{seed}.json", "w"))
print("DONE", arm, seed, f"test mrr {test_metrics['mrr']:.4f}", flush=True)
