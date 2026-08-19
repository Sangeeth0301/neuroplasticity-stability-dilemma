"""Corpus generation entry point: plans -> threads -> JSONL on disk.

Outputs
    data/raw/fact_bank.jsonl      every fact, with its supersession links
    data/raw/threads_train.jsonl  training threads (turns)
    data/raw/threads_test.jsonl   SEALED test split — opened once, at Phase 8
"""

from __future__ import annotations

from pathlib import Path

from .. import paths
from ..config import Config
from ..logging_ import write_jsonl
from ..seeding import child_rng
from ..types import ThreadPlan
from .thread_planner import plan_thread


def generate_split(cfg: Config, split: str, n_threads: int) -> list[ThreadPlan]:
    regimes = cfg.data.regimes
    plans: list[ThreadPlan] = []
    for i in range(n_threads):
        regime = regimes[i % len(regimes)]
        thread_id = f"{split}-{i:03d}"
        rng = child_rng(cfg.seed, split, i)
        plans.append(
            plan_thread(
                thread_id=thread_id,
                regime=regime,
                n_turns=cfg.data.turns_per_thread,
                n_users=cfg.data.n_users,
                n_facts=cfg.data.n_facts_per_thread,
                probe_distances=cfg.data.probe_distances,
                supersede_ratio=cfg.data.supersede_ratio,
                distractor_ratio=cfg.data.distractor_ratio,
                rng=rng,
            )
        )
    return plans


def write_corpus(cfg: Config, out_dir: Path | None = None) -> dict[str, int]:
    out_dir = Path(out_dir or paths.DATA_RAW)
    out_dir.mkdir(parents=True, exist_ok=True)

    counts: dict[str, int] = {}
    all_facts = []
    for split, n in (("train", cfg.data.n_threads_train), ("test", cfg.data.n_threads_test)):
        plans = generate_split(cfg, split, n)
        turns = [t.to_dict() for p in plans for t in p.turns]
        counts[f"{split}_turns"] = write_jsonl(out_dir / f"threads_{split}.jsonl", turns)
        counts[f"{split}_threads"] = len(plans)
        counts[f"{split}_probes"] = sum(t["kind"] == "probe" for t in turns)
        all_facts += [f.to_dict() for p in plans for f in p.facts]

    counts["facts"] = write_jsonl(out_dir / "fact_bank.jsonl", all_facts)
    return counts
