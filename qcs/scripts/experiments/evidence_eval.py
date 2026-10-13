"""Direct evidence that the model LEARNS NEW things and REMEMBERS OLD ones.

    python scripts/experiments/evidence_eval.py ARM SEED          ARM in {quantum, multi, linear, mlp}

Records, over the same 120-task stream used everywhere else:

  NEW  (plasticity)  for every task t:  score on t BEFORE training on it (the model has
                     never seen it)  and  AFTER training on it  -> how much it learns, and
                     whether that ability holds at task 100 as well as at task 5.
  OLD  (stability)   the full forgetting matrix: after every 10th task, the model is
                     re-scored on EVERY task seen so far (no carried-forward values).
  KNOWLEDGE (sealed test split)
       memory over distance   MRR as a function of how many turns ago the needed fact was
                              stated, plus a no-memory "context window" baseline
       updated vs unchanged   MRR on probes whose fact was later UPDATED (new information
                              must win) vs probes whose fact never changed (old memory).
"""
from __future__ import annotations

import json
import sys
import time
from pathlib import Path

import numpy as np
import torch

from qcs import paths
from qcs.classical.scorers import CosineScorer, build_scorer
from qcs.config import Config
from qcs.eval.metrics import evaluate_probes, mrr, stale_above_current
from qcs.ingest.loader import load_sus
from qcs.logging_ import RunLogger
from qcs.quantum.circuit import QRE
from qcs.quantum.hybrid import MultiReadoutQRE
from qcs.seeding import seed_everything
from qcs.train.dataset import ScoringDataset
from qcs.train.loop import TaskStream, run_continual

arm, seed = sys.argv[1], int(sys.argv[2])
OUT = Path(__file__).resolve().parents[2] / "results" / "raw" / "evidence"
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


def score(m, exs):
    return evaluate_probes(lambda ex: m.score(ex.X), exs, ks=(1,))["mrr"]


ds = ScoringDataset.load(cfg)
stream = TaskStream(ds.examples, 6)
tasks = stream.tasks()
T = len(tasks)

pre = [float("nan")] * T
post = [float("nan")] * T
matrix = {}                       # t -> [score on task j for j <= t]
pre[0] = score(model, tasks[0])   # an untrained model on the first task


def hook(t, m):
    post[t] = score(m, tasks[t])
    if t + 1 < T:
        pre[t + 1] = score(m, tasks[t + 1])        # not yet trained on: genuinely unseen
    if t % 10 == 9:
        matrix[t] = [score(m, tasks[j]) for j in range(t + 1)]
        print(f"[{arm} s{seed}] t={t + 1:3d} mean on all past tasks "
              f"{np.mean(matrix[t]):.3f}  (this task: pre {pre[t]:.2f} -> post {post[t]:.2f})",
              flush=True)


log = RunLogger.create(cfg, f"evidence_{arm}_s{seed}", root=paths.RUNS / "scratch")
t0 = time.time()
run_continual(model, stream, cfg, log, arm=f"{arm}_s{seed}", kind=kind, seed=seed,
              verbose=False, on_task_end=hook)

# ---------------- knowledge-level evidence on the SEALED test split --------------------
test = ScoringDataset.load(cfg, split="test")
turn_of = {s.su_id: s.turn_idx for s in load_sus("test", cfg.data.su_target_tokens)}
BANDS = [(0, 50), (50, 200), (200, 400), (400, 10_000)]
WINDOW = 100                                       # a "context window" of the last 100 turns
cos = CosineScorer(cfg.encode.feature_dim)


def distance(ex):
    g = int(np.argmax(ex.y))
    return ex.probe_turn - turn_of[ex.su_ids[g]]


def window_mrr(ex):
    """No memory: only candidates within the last WINDOW turns are visible."""
    d = np.array([ex.probe_turn - turn_of[i] for i in ex.su_ids])
    vis = d <= WINDOW
    if not vis[int(np.argmax(ex.y))]:
        return 0.0
    s = np.where(vis, cos.score(ex.X), -1e9)
    return float(mrr(s, ex.y))


by_dist = {"model": [], "window": [], "cosine": [], "n": []}
for lo, hi in BANDS:
    sub = [e for e in test.examples if lo <= distance(e) < hi]
    by_dist["n"].append(len(sub))
    by_dist["model"].append(score(model, sub) if sub else float("nan"))
    by_dist["cosine"].append(float(np.mean([mrr(cos.score(e.X), e.y) for e in sub])) if sub else float("nan"))
    by_dist["window"].append(float(np.mean([window_mrr(e) for e in sub])) if sub else float("nan"))

upd = [e for e in test.examples if e.is_stale.sum() > 0]       # the fact was UPDATED later
unc = [e for e in test.examples if e.is_stale.sum() == 0]      # the fact never changed
split = {}
for name, exs in (("updated", upd), ("unchanged", unc)):
    m = evaluate_probes(lambda ex: model.score(ex.X), exs, ks=(1,))
    c = evaluate_probes(lambda ex: cos.score(ex.X), exs, ks=(1,))
    split[name] = {"n": len(exs), "mrr": m["mrr"], "rank1": m["rank1"],
                   "stale_above": m["stale_above_current"], "cosine_mrr": c["mrr"],
                   "cosine_stale_above": c["stale_above_current"]}

out = {"arm": arm, "seed": seed, "pre": pre, "post": post,
       "matrix": {str(k): v for k, v in matrix.items()},
       "by_distance": by_dist, "bands": BANDS, "window": WINDOW, "split": split,
       "seconds": time.time() - t0}
json.dump(out, open(OUT / f"{arm}_s{seed}.json", "w"))
print("DONE", arm, seed, f"updated {split['updated']['mrr']:.3f} unchanged {split['unchanged']['mrr']:.3f}",
      flush=True)
