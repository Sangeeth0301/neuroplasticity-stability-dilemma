"""Phase 8 · the full evaluation: seeds, ablations, noise, the sealed test split,
significance and failure analysis.

    python scripts/run_eval.py --seeds 7 17 27 --tasks 120

Emits V35-V40 and V46. Every claim made in the blueprint should end up with a figure
here — including the ones that did not go our way.
"""

from __future__ import annotations

import argparse
import json
import time

import numpy as np
import torch

from qcs import paths
from qcs.classical.scorers import build_scorer
from qcs.config import Config
from qcs.eval.failure_analysis import analyse
from qcs.eval.metrics import evaluate_probes, mrr
from qcs.eval.significance import paired_bootstrap, per_thread_metric
from qcs.logging_ import RunLogger
from qcs.quantum.circuit import QRE
from qcs.quantum.noise import noisy_score
from qcs.seeding import seed_everything
from qcs.train.dataset import ScoringDataset
from qcs.train.loop import TaskStream, run_continual
from qcs.viz import eval_viz

# Feature-block masks for the ablations that remove information rather than machinery.
SALIENCE = slice(768, 832)      # ρ starts at 768; salience is its first 64 dims
GRAPH = slice(848, 880)         # graph features


def mask_features(X: np.ndarray, blocks: list[slice]) -> np.ndarray:
    """Zero a feature block and renormalise — the vector must stay a valid quantum state."""
    Z = X.copy()
    for b in blocks:
        Z[:, b] = 0.0
    return Z / (np.linalg.norm(Z, axis=1, keepdims=True) + 1e-8)


def make_model(name: str, cfg: Config, seed: int):
    return QRE(cfg, seed=seed) if name == "quantum" else build_scorer(name, cfg, seed=seed)


def train_arm(name: str, cfg: Config, stream: TaskStream, log: RunLogger, seed: int,
              tag: str | None = None):
    model = make_model(name, cfg, seed)
    run_continual(model, stream, cfg, log, arm=f"{tag or name}_s{seed}", kind=name,
                  seed=seed, verbose=False)
    return model


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", nargs="*", default=["tuned"])
    ap.add_argument("--seeds", nargs="*", type=int, default=[7, 17, 27])
    ap.add_argument("--arms", nargs="*", default=["quantum", "linear", "mlp"])
    ap.add_argument("--probes-per-task", type=int, default=6)
    ap.add_argument("--tasks", type=int, default=None)
    ap.add_argument("--ablation-tasks", type=int, default=60)
    ap.add_argument("--test-threads", type=int, default=6)
    ap.add_argument("--ablations-only", action="store_true")
    ap.add_argument("--tag", default="phase8_eval")
    args = ap.parse_args()

    cfg = Config.load(*args.config)
    cfg.validate()
    seed_everything(cfg.seed)
    log = RunLogger.create(cfg, args.tag)
    prov = f"{log.run_dir.name} · seeds {args.seeds}"
    t_start = time.time()

    ds = ScoringDataset.load(cfg)
    stream = TaskStream(ds.examples, args.probes_per_task)
    tasks = stream.tasks()
    if args.tasks:
        tasks = tasks[: args.tasks]
        stream = TaskStream([e for t in tasks for e in t], args.probes_per_task)
    print(f"train stream: {len(stream.tasks())} tasks, {len(stream.examples)} probes")

    # ---------------- A · multi-seed comparison ---------------------------
    per_seed: dict[str, list[dict]] = {a: [] for a in args.arms}
    trained: dict[str, object] = {}
    for seed in ([] if args.ablations_only else args.seeds):
        for arm in args.arms:
            t0 = time.time()
            model = train_arm(arm, cfg, stream, log, seed)
            m = evaluate_probes(lambda ex, mm=model: mm.score(ex.X), stream.examples)
            per_seed[arm].append(m)
            if seed == args.seeds[0]:
                trained[arm] = model
                torch.save(model.state_dict(), log.run_dir / "checkpoints" / f"{arm}.pt")
            print(f"  [seed {seed}] {arm:8s} mrr {m['mrr']:.4f}  rank1 {m['rank1']:.4f}  "
                  f"stale {m['stale_above_current']:.4f}  ({time.time() - t0:.0f}s)")

    keys = ["rank1", "mrr", "auc", "stale_above_current"]
    headline = {arm: {k: (float(np.nanmean([r[k] for r in rs])),
                          float(np.nanstd([r[k] for r in rs]))) for k in keys}
                for arm, rs in per_seed.items()}
    print("\nheadline (mean ± std over seeds):")
    for arm, row in headline.items():
        print("  " + arm.ljust(9) + "  ".join(f"{k} {v[0]:.3f}±{v[1]:.3f}" for k, v in row.items()))

    # ---------------- B · ablations (quantum arm) -------------------------
    ab_stream = TaskStream(stream.examples[: args.ablation_tasks * args.probes_per_task],
                           args.probes_per_task)
    seed0 = args.seeds[0]
    base_model = train_arm("quantum", cfg, ab_stream, log, seed0, tag="abl_full")
    base_mrr = evaluate_probes(lambda ex: base_model.score(ex.X), ab_stream.examples)["mrr"]
    ablations = {}

    variants = {
        "no_anchor (λ_Ω=0)": {"lambda_omega": 0.0},
        "no_replay": {"replay_ratio": 0.0},
        "no_trust_region": {"trust_region_delta": 1e9},
    }
    for label, over in variants.items():
        c = Config.load(*args.config)
        for k, v in over.items():
            setattr(c.train, k, v)
        m = train_arm("quantum", c, ab_stream, log, seed0, tag=f"abl_{label.split()[0]}")
        ablations[label] = evaluate_probes(lambda ex, mm=m: mm.score(ex.X), ab_stream.examples)["mrr"]

    c = Config.load(*args.config)
    c.quantum.observable = "global_z"
    m = train_arm("quantum", c, ab_stream, log, seed0, tag="abl_global")
    ablations["global observable"] = evaluate_probes(
        lambda ex, mm=m: mm.score(ex.X), ab_stream.examples)["mrr"]

    for label, blocks in (("no salience block", [SALIENCE]), ("no graph block", [GRAPH]),
                          ("no ρ block at all", [slice(768, 1024)])):
        ablations[label] = evaluate_probes(
            lambda ex: base_model.score(mask_features(ex.X, blocks)), ab_stream.examples)["mrr"]

    print("\nablations (Δ from full system):")
    for k, v in sorted(ablations.items(), key=lambda kv: kv[1]):
        print(f"  {k:24s} {v:.4f}  ({v - base_mrr:+.4f})")

    # ---------------- C · noise -------------------------------------------
    q = trained["quantum"]
    noise_curve = {}
    for p in (0.0, 1e-4, 5e-4, 1e-3, 5e-3):
        sub = stream.examples[:40]
        m = evaluate_probes(lambda ex: noisy_score(q, ex.X, p, trajectories=6), sub)
        noise_curve[p] = m
        print(f"  noise p={p:<7} mrr {m['mrr']:.4f}  rank1 {m['rank1']:.4f}")

    # ---------------- D · sealed test split -------------------------------
    print("\nopening the sealed test split…")
    test_ds = ScoringDataset(cfg, split="test").build(
        n_threads=args.test_threads, max_candidates=64, verbose=False)
    test_metrics = {}
    for arm, model in trained.items():
        test_metrics[arm] = evaluate_probes(lambda ex, mm=model: mm.score(ex.X),
                                            test_ds.examples)
        print(f"  [TEST] {arm:8s} mrr {test_metrics[arm]['mrr']:.4f}  "
              f"rank1 {test_metrics[arm]['rank1']:.4f}  "
              f"stale {test_metrics[arm]['stale_above_current']:.4f}")
    for name in ("cosine", "recency"):
        sm = build_scorer(name, cfg)
        test_metrics[name] = evaluate_probes(lambda ex, mm=sm: mm.score(ex.X),
                                             test_ds.examples)
        print(f"  [TEST] {name:8s} mrr {test_metrics[name]['mrr']:.4f}")

    # per-regime on the test split (regimes cycle with thread index)
    regimes = cfg.data.regimes
    by_regime: dict[str, dict[str, float]] = {}
    for arm, model in trained.items():
        by_regime[arm] = {}
        for i, reg in enumerate(regimes):
            sub = [e for e in test_ds.examples if int(e.thread_id.split("-")[1]) % len(regimes) == i]
            if sub:
                by_regime[arm][reg] = evaluate_probes(
                    lambda ex, mm=model: mm.score(ex.X), sub)["mrr"]

    # ---------------- E · significance and failures -----------------------
    tests = {}
    arm_names = list(trained)
    for i in range(len(arm_names)):
        for j in range(i + 1, len(arm_names)):
            a, b = arm_names[i], arm_names[j]
            pa = per_thread_metric(lambda ex, mm=trained[a]: mm.score(ex.X),
                                   test_ds.examples, mrr)
            pb = per_thread_metric(lambda ex, mm=trained[b]: mm.score(ex.X),
                                   test_ds.examples, mrr)
            tests[f"{a} − {b}"] = paired_bootstrap(pa, pb, seed=cfg.seed)
            r = tests[f"{a} − {b}"]
            print(f"  {a} − {b}: Δ {r['difference']:+.4f} "
                  f"[{r['ci_low']:+.4f}, {r['ci_high']:+.4f}] p={r['p_value']:.3f} "
                  f"{'SIGNIFICANT' if r['significant'] else 'not significant'}")

    analyses = {arm: analyse(lambda ex, mm=model: mm.score(ex.X), test_ds.examples)
                for arm, model in trained.items()}
    analyses["cosine"] = analyse(
        lambda ex, mm=build_scorer("cosine", cfg): mm.score(ex.X), test_ds.examples)
    print("\nfailure taxonomy (test split):")
    for arm, a in analyses.items():
        print(f"  {arm:9s} acc {a['accuracy']:.3f}  {a['counts']}")

    stale_rates = {arm: m["stale_above_current"] for arm, m in test_metrics.items()}

    figs = [
        eval_viz.v46_headline(headline, log.figs, prov),
        eval_viz.v35_ablations(base_mrr, ablations, log.figs, prov),
        eval_viz.v36_v37_regimes(by_regime, stale_rates, log.figs, prov),
        eval_viz.v38_noise(noise_curve, log.figs, prov),
        eval_viz.v39_failures(analyses, log.figs, prov),
        eval_viz.v40_significance(tests, log.figs, prov),
    ]

    log.note("phase8_summary", {
        "seeds": args.seeds, "n_tasks": len(stream.tasks()),
        "headline": {a: {k: list(v) for k, v in r.items()} for a, r in headline.items()},
        "ablations": {"full_system": base_mrr, **ablations},
        "noise": {str(p): {k: float(v) for k, v in m.items()} for p, m in noise_curve.items()},
        "test_split": {a: {k: float(v) for k, v in m.items()} for a, m in test_metrics.items()},
        "by_regime": by_regime, "significance": tests, "failures": analyses,
        "minutes": (time.time() - t_start) / 60,
    })
    print(f"\ntotal {(time.time() - t_start) / 60:.1f} min")
    for f in figs:
        print("  ", f)
    print("run dir:", log.run_dir)
    _ = paths


if __name__ == "__main__":
    main()
