"""Phase 1 · generate the corpus, validate it, and emit V01-V04.

    python scripts/gen_dataset.py [--config quantum_qre] [--tag phase1]
"""

from __future__ import annotations

import argparse
import json

from qcs import paths
from qcs.config import Config
from qcs.datagen.generate import write_corpus
from qcs.datagen.stats import corpus_stats, load_split
from qcs.datagen.validate import oracle_vs_random, validate_corpus
from qcs.logging_ import RunLogger, read_jsonl
from qcs.seeding import seed_everything
from qcs.viz import dataset_viz


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", nargs="*", default=[])
    ap.add_argument("--tag", default="phase1_dataset")
    args = ap.parse_args()

    cfg = Config.load(*args.config)
    cfg.validate()
    seed_everything(cfg.seed)
    paths.ensure_dirs()

    log = RunLogger.create(cfg, args.tag)
    prov = f"{log.run_dir.name} · seed {cfg.seed}"

    print("generating corpus…")
    counts = write_corpus(cfg)
    print(json.dumps(counts, indent=2))

    print("validating…")
    report = validate_corpus(probe_distances=cfg.data.probe_distances)
    calib = oracle_vs_random(seed=cfg.seed)
    report.update(calib)
    print(json.dumps(report, indent=2))
    assert calib["oracle_recall"] == 1.0, "oracle must score 1.0 — corpus is not a valid instrument"
    assert calib["random_recall"] < 0.25, "random retriever too strong — task is trivial"

    print("plotting…")
    stats = corpus_stats()
    turns = load_split("train")
    facts = read_jsonl(paths.DATA_RAW / "fact_bank.jsonl")
    # one V01 per regime — regimes cycle in thread order, so the first three cover all
    figs = [
        dataset_viz.v01_thread_structure(turns, f"train-{i:03d}", log.figs, prov)
        for i in range(len(cfg.data.regimes))
    ]
    figs += [
        dataset_viz.v02_probe_distances(turns, log.figs, cfg.data.probe_distances, prov),
        dataset_viz.v03_fact_ownership(facts, log.figs, prov),
        dataset_viz.v04_corpus_stats(stats, turns, log.figs, prov),
    ]

    log.note("dataset_report", {"counts": counts, "validation": report, "stats": stats})
    log.log(task_id=0, phase="phase1", split="train",
            extra={"probes": report["probes"], "facts": report["facts"]})

    print("\nfigures:")
    for f in figs:
        print("  ", f)
    print("\nrun dir:", log.run_dir)


if __name__ == "__main__":
    main()
