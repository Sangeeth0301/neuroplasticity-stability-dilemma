"""Single-variant training run on the TRAIN stream; writes one JSON line per variant.

    python scripts/experiments/sweep_quantum.py NAME [--depth 8] [--epochs 3] [--lr 0.02] [--multi] ...

Selection uses the train stream only. The sealed test split is opened by run_eval.py.
"""
from __future__ import annotations

import argparse
import json
import time

import numpy as np

from qcs import paths
from qcs.config import Config
from qcs.eval.metrics import evaluate_probes
from qcs.logging_ import RunLogger
from qcs.quantum.circuit import QRE
from qcs.quantum.hybrid import MultiReadoutQRE
from qcs.seeding import seed_everything
from qcs.train.dataset import ScoringDataset
from qcs.train.loop import TaskStream, run_continual

ap = argparse.ArgumentParser()
ap.add_argument("name")
ap.add_argument("--seed", type=int, default=7)
ap.add_argument("--depth", type=int, default=None)
ap.add_argument("--epochs", type=int, default=None)
ap.add_argument("--lr", type=float, default=0.02)
ap.add_argument("--beta", type=float, default=None)
ap.add_argument("--lam", type=float, default=None)
ap.add_argument("--replay", type=float, default=None)
ap.add_argument("--delta", type=float, default=None)
ap.add_argument("--bs", type=int, default=None)
ap.add_argument("--multi", action="store_true")
a = ap.parse_args()

cfg = Config.load("tuned")
cfg.seed = a.seed
if a.depth: cfg.quantum.depth = a.depth
if a.epochs: cfg.train.epochs_per_task = a.epochs
if a.beta: cfg.quantum.beta_init = a.beta
if a.lam is not None: cfg.train.lambda_omega = a.lam
if a.replay is not None: cfg.train.replay_ratio = a.replay
if a.delta: cfg.train.trust_region_delta = a.delta
if a.bs: cfg.train.batch_size = a.bs
cfg.train.lr_by_arm["quantum"] = a.lr
cfg.validate()
seed_everything(a.seed)
log = RunLogger.create(cfg, f"sweep_{a.name}", root=paths.RUNS / "scratch")
ds = ScoringDataset.load(cfg)
stream = TaskStream(ds.examples, 6)
model = (MultiReadoutQRE if a.multi else QRE)(cfg, seed=a.seed)
t0 = time.time()
res = run_continual(model, stream, cfg, log, arm=f"{a.name}_s{a.seed}", kind="quantum",
                    seed=a.seed, verbose=False)
m = evaluate_probes(lambda ex: model.score(ex.X), ds.examples)
rows = [r for r in res.history]
out = {"name": a.name, "seed": a.seed, "mrr": m["mrr"], "rank1": m["rank1"],
       "stale": m["stale_above_current"], "bwt": res.bwt(), "final_avg": res.final_average(),
       "plasticity": res.plasticity_index(), "trF_ratio": rows[-1]["tr_fim"] / max(rows[0]["tr_fim"], 1e-12),
       "params": model.n_circuit_params, "sec": time.time() - t0}
print(json.dumps(out))
RESULTS = paths.ROOT / "results" / "exploration" / "sweep_results.jsonl"
RESULTS.parent.mkdir(parents=True, exist_ok=True)
open(RESULTS, "a").write(json.dumps(out) + "\n")
