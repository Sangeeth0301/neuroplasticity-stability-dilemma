"""Corpus statistics — the numbers V01-V04 are drawn from."""

from __future__ import annotations

from collections import Counter
from pathlib import Path

import numpy as np

from .. import paths
from ..logging_ import read_jsonl


def load_split(split: str, raw_dir: Path | None = None) -> list[dict]:
    raw_dir = Path(raw_dir or paths.DATA_RAW)
    return read_jsonl(raw_dir / f"threads_{split}.jsonl")


def corpus_stats(raw_dir: Path | None = None) -> dict:
    raw_dir = Path(raw_dir or paths.DATA_RAW)
    facts = read_jsonl(raw_dir / "fact_bank.jsonl")
    out: dict = {
        "n_facts": len(facts),
        "n_superseded": sum(1 for f in facts if f.get("superseded_by")),
        "attributes": dict(Counter(f["attribute"] for f in facts)),
        "regimes": dict(Counter(f["regime"] for f in facts)),
    }
    for split in ("train", "test"):
        turns = load_split(split, raw_dir)
        kinds = Counter(t["kind"] for t in turns)
        lengths = [len(t["text"].split()) for t in turns]
        dists = [t["probe_distance"] for t in turns if t["kind"] == "probe"]
        out[split] = {
            "n_turns": len(turns),
            "n_threads": len({t["thread_id"] for t in turns}),
            "kinds": dict(kinds),
            "words_per_turn_mean": float(np.mean(lengths)),
            "words_per_turn_p95": float(np.percentile(lengths, 95)),
            "probe_distance_mean": float(np.mean(dists)) if dists else 0.0,
            "probe_distance_max": int(np.max(dists)) if dists else 0,
            "speakers": dict(Counter(t["user_id"] for t in turns)),
        }
    return out
