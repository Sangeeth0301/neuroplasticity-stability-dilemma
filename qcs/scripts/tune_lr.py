"""Phase 5 · give every arm its own best learning rate before comparing them.

A quantum-vs-classical result where the quantum arm ran at lr 0.03 and the classical arm
at 1e-3 is not a comparison, it is a handicap — and it is the first thing a reviewer will
attack. This script trains each arm on a prefix of the task stream at several learning
rates and scores it on the NEXT tasks (never seen), then writes the winners into
`configs/tuned.yaml` for the real run to load.
"""

from __future__ import annotations

import argparse
import json

import numpy as np
import torch
import yaml

from qcs import paths
from qcs.classical.scorers import build_scorer
from qcs.config import Config
from qcs.eval.metrics import evaluate_probes
from qcs.quantum.circuit import QRE
from qcs.seeding import seed_everything
from qcs.train.dataset import ScoringDataset
from qcs.train.loop import TaskStream, _flatten_task


def train_prefix(model, tasks, lr, cfg, rng, epochs: int) -> None:
    opt = torch.optim.Adam(model.parameters(), lr=lr)
    bce = torch.nn.BCELoss()
    for task in tasks:
        X, y = _flatten_task(task, cfg.train.n_negatives, rng)
        for _ in range(epochs):
            idx = rng.permutation(len(X))
            for i in range(0, len(X), cfg.train.batch_size):
                j = idx[i:i + cfg.train.batch_size]
                opt.zero_grad(set_to_none=True)
                loss = bce(model(X[j]).clamp(1e-6, 1 - 1e-6),
                           torch.as_tensor(y[j], dtype=torch.float32))
                loss.backward()
                opt.step()
        model.wrap_angles()


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--arms", nargs="*", default=["quantum", "linear", "mlp"])
    ap.add_argument("--lrs", nargs="*", type=float, default=[0.001, 0.005, 0.02, 0.05, 0.1])
    ap.add_argument("--train-tasks", type=int, default=12)
    ap.add_argument("--eval-tasks", type=int, default=8)
    ap.add_argument("--probes-per-task", type=int, default=6)
    args = ap.parse_args()

    cfg = Config.load()
    seed_everything(cfg.seed)
    ds = ScoringDataset.load(cfg)
    tasks = TaskStream(ds.examples, args.probes_per_task).tasks()
    train_tasks = tasks[: args.train_tasks]
    eval_tasks = tasks[args.train_tasks: args.train_tasks + args.eval_tasks]
    eval_examples = [e for t in eval_tasks for e in t]
    print(f"tuning on {len(train_tasks)} tasks, scoring on {len(eval_examples)} held-out probes")

    best: dict[str, float] = {}
    table: dict[str, dict[str, float]] = {}
    for arm in args.arms:
        table[arm] = {}
        for lr in args.lrs:
            rng = np.random.default_rng(cfg.seed)
            model = QRE(cfg, seed=cfg.seed) if arm == "quantum" else build_scorer(arm, cfg, cfg.seed)
            train_prefix(model, train_tasks, lr, cfg, rng, cfg.train.epochs_per_task)
            m = evaluate_probes(lambda ex, mm=model: mm.score(ex.X), eval_examples, ks=(1,))
            table[arm][str(lr)] = m["mrr"]
            print(f"  {arm:8s} lr={lr:<6} held-out MRR {m['mrr']:.4f}  rank1 {m['rank1']:.3f}")
        best[arm] = float(max(table[arm], key=lambda k: table[arm][k]))
        print(f"  -> {arm}: best lr = {best[arm]}  (MRR {max(table[arm].values()):.4f})")

    out = paths.CONFIGS / "tuned.yaml"
    payload = {"train": {"lr_by_arm": best}}
    out.write_text(yaml.safe_dump(payload, sort_keys=False), encoding="utf-8")
    print("\nwrote", out)
    print(json.dumps(payload, indent=2))


if __name__ == "__main__":
    main()
