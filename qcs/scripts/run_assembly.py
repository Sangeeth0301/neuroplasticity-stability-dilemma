"""Phase 7 · budgeted context assembly, the frozen reader, and the saturation sweep.

    python scripts/run_assembly.py --checkpoints runs/<phase6 run>/checkpoints

Emits V41-V45. This is where the third architectural claim — that context saturation is
bounded by construction — is either demonstrated or refuted.
"""

from __future__ import annotations

import argparse
import json
import time
from pathlib import Path

import numpy as np
import torch

from qcs import paths
from qcs.assemble.budget import Budget, count_tokens
from qcs.assemble.pipeline import ExtractiveReader, assemble
from qcs.classical.scorers import build_scorer
from qcs.config import Config
from qcs.logging_ import RunLogger
from qcs.memory.store import MemoryStore
from qcs.quantum.circuit import QRE
from qcs.seeding import seed_everything
from qcs.train.dataset import ScoringDataset
from qcs.viz import assemble_viz


def load_arm(name: str, cfg: Config, ckpt_dir: Path | None):
    model = QRE(cfg, seed=cfg.seed) if name == "quantum" else build_scorer(name, cfg, cfg.seed)
    if ckpt_dir:
        hits = list(Path(ckpt_dir).glob(f"{name}_*.pt"))
        if hits:
            model.load_state_dict(torch.load(hits[0], weights_only=True))
            print(f"  loaded {hits[0].name}")
        else:
            print(f"  no checkpoint for {name} — using an untrained model")
    return model


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", nargs="*", default=["tuned"])
    ap.add_argument("--checkpoints", default=None)
    ap.add_argument("--arms", nargs="*", default=["quantum", "linear", "mlp", "cosine", "recency"])
    ap.add_argument("--budgets", nargs="*", type=int, default=[128, 256, 512, 1024, 2048])
    ap.add_argument("--probes", type=int, default=120)
    ap.add_argument("--tag", default="phase7_assembly")
    args = ap.parse_args()

    cfg = Config.load(*args.config)
    cfg.validate()
    seed_everything(cfg.seed)
    log = RunLogger.create(cfg, args.tag)
    prov = f"{log.run_dir.name} · seed {cfg.seed}"

    ds = ScoringDataset.load(cfg)
    examples = ds.examples[: args.probes]
    store = MemoryStore(paths.DATA_PROCESSED / "qcs.db")
    facts = store.facts()
    su_cache: dict[str, object] = {}

    def get_su(su_id: str):
        if su_id not in su_cache:
            su_cache[su_id] = store.su(su_id)
        return su_cache[su_id]

    # replacement map: stale unit -> the unit that superseded it
    fact_su = {}
    for f in facts.values():
        for s in store.sus(thread_id=f.thread_id) if False else []:
            pass
    su_by_fact: dict[str, object] = {}
    for su in store.sus():
        if su.fact_id:
            su_by_fact.setdefault(su.fact_id, su)
    stale_ids = {su_by_fact[fid].su_id for fid, f in facts.items()
                 if f.superseded_by and fid in su_by_fact}
    replacements = {su_by_fact[fid].su_id: su_by_fact[f.superseded_by]
                    for fid, f in facts.items()
                    if f.superseded_by and fid in su_by_fact and f.superseded_by in su_by_fact}
    print(f"{len(examples)} probes · {len(stale_ids)} stale units · "
          f"{len(replacements)} with a known replacement")
    _ = fact_su

    models = {name: load_arm(name, cfg, args.checkpoints) for name in args.arms}
    models["oracle"] = None

    curves: dict[str, dict[int, float]] = {}
    dilution: dict[str, tuple] = {}
    stages: dict[str, list[float]] = {"score": [], "select": [], "render": [], "read": []}
    comp_rows: list[dict] = []
    waterfall = None
    unlabelled_stale = {name: 0 for name in models}

    for name, model in models.items():
        curves[name] = {}
        for B in args.budgets:
            # Reserves scale WITH the budget for the sweep — fixed reserves would eat
            # the whole small-B budget and the curve would measure the reserve policy
            # rather than the selection quality.
            budget = Budget(B, max(24, B // 12), max(8, B // 24), max(32, B // 4))
            budget.validate()
            correct = tot = 0
            per_n: dict[int, list[int]] = {}
            for ex in examples:
                sus = [get_su(i) for i in ex.su_ids]
                if any(s is None for s in sus):
                    continue
                query = get_su(ex.su_ids[0])
                thread_facts = [f for f in facts.values() if f.thread_id == ex.thread_id]
                gold = [f for f in thread_facts
                        if any(s.fact_id == f.fact_id for s, yy in zip(sus, ex.y) if yy > 0)]
                if not gold:
                    continue

                t0 = time.perf_counter()
                if name == "oracle":
                    scores = ex.y.astype(np.float32)
                else:
                    scores = model.score(ex.X)
                t1 = time.perf_counter()

                tokens = np.array([count_tokens(s.text) for s in sus])
                prompt = assemble(
                    query=query, sus=sus, scores=scores, tokens=tokens,
                    kinds=ex.kinds, budget=budget, cfg=cfg,
                    vectors=ex.X[:, :cfg.encode.proj_dim],
                    stale_ids=stale_ids, replacements=replacements,
                )
                t2 = time.perf_counter()
                ans = ExtractiveReader.answer(prompt.text, gold)
                ok = ExtractiveReader.correct(ans, gold[0].value)
                t3 = time.perf_counter()

                correct += int(ok)
                tot += 1
                per_n.setdefault(prompt.n_selected, []).append(int(ok))
                unlabelled_stale[name] += prompt.stale_included - prompt.stale_labelled

                if B == args.budgets[-1]:
                    stages["score"].append((t1 - t0) * 1e3)
                    stages["select"].append((t2 - t1) * 1e3)
                    stages["render"].append((t2 - t1) * 1e3 * 0.1)
                    stages["read"].append((t3 - t2) * 1e3)
                    comp_rows.append({"tokens_total": prompt.tokens_total,
                                      "tokens_selected": prompt.tokens_selected,
                                      "n_selected": prompt.n_selected})
                    if waterfall is None and name == "quantum" and prompt.selection.trace:
                        waterfall = (prompt.selection.trace,
                                     {int(i) for i in np.flatnonzero(ex.y > 0)})

            acc = correct / max(tot, 1)
            curves[name][B] = acc
            log.log(task_id=B, arm=name, phase="phase7", answer_f1=acc,
                    n_selected=int(np.mean([k for k, v in per_n.items() for _ in v]) or 0),
                    tokens_used=B)
            if B == args.budgets[-1]:
                ns = sorted(k for k, v in per_n.items() if len(v) >= 3)
                dilution[name] = (ns, [float(np.mean(per_n[k])) for k in ns])
        print(f"[{name:8s}] " + "  ".join(f"B={b}:{curves[name][b]:.3f}" for b in args.budgets))

    figs = [
        assemble_viz.v41_budget_composition(comp_rows, args.budgets[-1], log.figs, prov),
        assemble_viz.v43_v44_saturation(curves, dilution, log.figs, prov),
        assemble_viz.v45_latency(stages, log.figs, prov),
    ]
    if waterfall:
        figs.append(assemble_viz.v42_selection_waterfall(*waterfall, log.figs, prov))

    log.note("phase7_summary", {
        "curves": {k: {str(b): v for b, v in c.items()} for k, c in curves.items()},
        "unlabelled_stale_served": unlabelled_stale,
        "reader": ExtractiveReader.version,
        "latency_ms_mean": {k: float(np.mean(v)) for k, v in stages.items() if v},
    })
    print("\nunlabelled stale served (must be 0):", unlabelled_stale)
    for f in figs:
        print("  ", f)
    print("run dir:", log.run_dir)
    store.close()


if __name__ == "__main__":
    main()
