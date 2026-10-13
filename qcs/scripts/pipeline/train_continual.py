"""Phases 5-6 · baselines, the parameter-matched twin, and the continual run.

    python scripts/pipeline/train_continual.py --threads 12 --probes-per-task 4 --arms quantum linear

Emits V21-V34. This is the experiment; everything before it was preparation.
"""

from __future__ import annotations

import argparse
import json
import time

import numpy as np
import torch

from qcs import paths
from qcs.classical.scorers import OracleScorer, build_scorer
from qcs.config import Config
from qcs.eval.metrics import evaluate_probes
from qcs.logging_ import RunLogger, read_jsonl
from qcs.quantum.circuit import QRE
from qcs.quantum.hybrid import HybridQRE
from qcs.seeding import seed_everything
from qcs.train.dataset import ScoringDataset
from qcs.train.loop import TaskStream, run_continual
from qcs.viz import training_viz


def make_model(name: str, cfg: Config, seed: int):
    if name == "quantum":
        return QRE(cfg, seed=seed)
    if name == "hybrid_quantum":
        return HybridQRE(cfg, seed=seed)
    return build_scorer(name, cfg, seed=seed)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", nargs="*", default=[])
    ap.add_argument("--threads", type=int, default=12)
    ap.add_argument("--probes", type=int, default=None)
    ap.add_argument("--probes-per-task", type=int, default=4)
    ap.add_argument("--arms", nargs="*", default=["quantum", "linear", "mlp"])
    ap.add_argument("--static-arms", nargs="*", default=["cosine", "recency", "random"])
    ap.add_argument("--seed", type=int, default=None)
    ap.add_argument("--rebuild", action="store_true")
    ap.add_argument("--candidates", type=int, default=64)
    ap.add_argument("--tag", default="phase6_continual")
    args = ap.parse_args()

    cfg = Config.load(*args.config)
    cfg.validate()
    if args.seed is not None:
        cfg.seed = args.seed
    seed_everything(cfg.seed)
    log = RunLogger.create(cfg, args.tag)
    prov = f"{log.run_dir.name} · seed {cfg.seed}"

    cache = paths.DATA_PROCESSED / "scoring_train.npz"
    if args.rebuild or not cache.exists():
        print("building scorer dataset…")
        ds = ScoringDataset(cfg).build(n_threads=args.threads, max_probes_per_thread=args.probes,
                                      max_candidates=args.candidates)
        ds.save()
    else:
        ds = ScoringDataset.load(cfg)
        print(f"loaded cached dataset: {len(ds.examples)} probes")
    print(json.dumps(ds.stats(), indent=2))

    stream = TaskStream(ds.examples, args.probes_per_task)
    n_tasks = len(stream.tasks())
    print(f"task stream: {n_tasks} tasks × {args.probes_per_task} probes")

    # ---------------- static baselines (no learning) ----------------------
    metrics: dict[str, dict] = {}
    for name in args.static_arms:
        m = build_scorer(name, cfg)
        metrics[name] = evaluate_probes(lambda ex, m=m: m.score(ex.X), ds.examples)
        print(f"[{name:8s}] rank1 {metrics[name]['rank1']:.3f}  mrr {metrics[name]['mrr']:.3f}  "
              f"stale-above {metrics[name]['stale_above_current']:.3f}")
    metrics["oracle"] = evaluate_probes(lambda ex: OracleScorer.score_with_labels(ex.y),
                                        ds.examples)

    # ---------------- continual arms --------------------------------------
    results: dict[str, dict] = {}
    for name in args.arms:
        model = make_model(name, cfg, cfg.seed)
        arm = f"{name}_s{cfg.seed}"
        print(f"\n=== {arm} · |θ| = {model.n_circuit_params} ===")
        t0 = time.time()
        res = run_continual(model, stream, cfg, log, arm=arm, seed=cfg.seed)
        dt = time.time() - t0
        results[arm] = {
            "A": res.A, "bwt": res.bwt(), "fwt": res.fwt(),
            "plasticity_index": res.plasticity_index(),
            "final_average": res.final_average(),
            "n_params": model.n_circuit_params, "seconds": dt,
        }
        metrics[arm] = evaluate_probes(lambda ex, m=model: m.score(ex.X), ds.examples)
        print(f"  BWT {res.bwt():+.4f}   FWT {res.fwt():+.4f}   "
              f"plasticity {res.plasticity_index():.4f}   final avg {res.final_average():.4f}")
        print(f"  rank1 {metrics[arm]['rank1']:.3f}  mrr {metrics[arm]['mrr']:.3f}  "
              f"stale-above {metrics[arm]['stale_above_current']:.3f}  ({dt:.0f}s)")
        torch.save(model.state_dict(), log.run_dir / "checkpoints" / f"{arm}.pt")

    rows = read_jsonl(log.run_dir / "metrics.jsonl")
    arms = list(results)
    figs = [
        training_viz.v21_v25_training(rows, arms, log.figs, prov),
        training_viz.v26_v29_stability(rows, arms, log.figs, prov),
        training_viz.v31_v33_continual(results, log.figs, prov),
        training_viz.v34_metrics_table(metrics, log.figs, prov),
    ]

    log.note("phase6_summary", {
        "n_tasks": n_tasks, "probes_per_task": args.probes_per_task,
        "metrics": {k: {kk: float(vv) for kk, vv in v.items()} for k, v in metrics.items()},
        "continual": {k: {kk: (float(vv) if not isinstance(vv, np.ndarray) else None)
                          for kk, vv in v.items()} for k, v in results.items()},
    })
    for f in figs:
        print("  ", f)
    print("run dir:", log.run_dir)


if __name__ == "__main__":
    main()
