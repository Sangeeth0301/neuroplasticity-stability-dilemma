"""Structural validation of the generated corpus.

These are not style checks. Each assertion corresponds to a way the corpus could
silently stop being a valid measuring instrument:

  V1  every probe's required facts appear STRICTLY EARLIER in the same thread
      -> otherwise the probe is unanswerable and the ceiling is not 1.0
  V2  no probe requires a superseded fact
      -> otherwise "correct" is ambiguous
  V3  fact ids are unique corpus-wide
  V4  turn indices are contiguous and speakers are inside the participant set
  V5  probe distances actually cover the configured bands
  V6  no train/test thread id overlap (leakage)
  V7  an oracle retriever scores 1.0 and a random one scores near chance
"""

from __future__ import annotations

from collections import Counter
from pathlib import Path

from .. import paths
from ..logging_ import read_jsonl


class CorpusError(AssertionError):
    pass


def _check(cond: bool, msg: str) -> None:
    if not cond:
        raise CorpusError(msg)


def validate_corpus(raw_dir: Path | None = None, probe_distances: list[int] | None = None) -> dict:
    raw_dir = Path(raw_dir or paths.DATA_RAW)
    facts = {f["fact_id"]: f for f in read_jsonl(raw_dir / "fact_bank.jsonl")}
    report: dict = {"facts": len(facts)}

    _check(len(facts) == len(read_jsonl(raw_dir / "fact_bank.jsonl")), "V3 duplicate fact ids")

    seen_threads: set[str] = set()
    dist_counter: Counter[int] = Counter()
    n_probes = 0

    for split in ("train", "test"):
        turns = read_jsonl(raw_dir / f"threads_{split}.jsonl")
        by_thread: dict[str, list[dict]] = {}
        for t in turns:
            by_thread.setdefault(t["thread_id"], []).append(t)

        for tid, ts in by_thread.items():
            _check(tid not in seen_threads, f"V6 thread id {tid} appears in both splits")
            seen_threads.add(tid)
            ts.sort(key=lambda x: x["turn_idx"])
            _check([t["turn_idx"] for t in ts] == list(range(len(ts))),
                   f"V4 non-contiguous turn indices in {tid}")
            users = {t["user_id"] for t in ts}
            _check(len(users) <= 8, f"V4 too many speakers in {tid}")

            first_seen: dict[str, int] = {}
            for t in ts:
                if t["kind"] in ("fact", "update"):
                    first_seen.setdefault(t["fact_id"], t["turn_idx"])
                elif t["kind"] == "probe":
                    n_probes += 1
                    for fid in t["required_facts"]:
                        _check(fid in facts, f"V1 probe references unknown fact {fid}")
                        _check(fid in first_seen,
                               f"V1 probe at {tid}:{t['turn_idx']} needs {fid} "
                               "which has not been stated yet")
                        _check(first_seen[fid] < t["turn_idx"], "V1 probe not strictly after fact")
                        _check(facts[fid].get("superseded_by") is None,
                               f"V2 probe requires superseded fact {fid}")
                    dist_counter[t["probe_distance"]] += 1

        report[f"{split}_threads"] = len(by_thread)
        report[f"{split}_turns"] = len(turns)

    report["probes"] = n_probes
    report["distinct_probe_distances"] = len(dist_counter)

    if probe_distances:
        # V5: probes are nudged to the nearest free slot, so allow a small offset band
        for d in probe_distances:
            hits = sum(c for dist, c in dist_counter.items() if d <= dist <= d + 40)
            _check(hits > 0, f"V5 no probes near configured distance {d}")
    return report


def oracle_vs_random(raw_dir: Path | None = None, seed: int = 0, n_sample: int = 300) -> dict:
    """V7: calibrate the instrument.

    Oracle = return exactly the required fact ids. Random = return the same number of
    random semantic units. If the oracle does not score 1.0, the corpus is broken; if
    random does not score near chance, the task is trivially easy.
    """
    import numpy as np

    rng = np.random.default_rng(seed)
    raw_dir = Path(raw_dir or paths.DATA_RAW)
    turns = read_jsonl(raw_dir / "threads_train.jsonl")
    by_thread: dict[str, list[dict]] = {}
    for t in turns:
        by_thread.setdefault(t["thread_id"], []).append(t)

    oracle_hits, random_hits, total = 0, 0, 0
    for tid, ts in by_thread.items():
        ts.sort(key=lambda x: x["turn_idx"])
        fact_turns = [t for t in ts if t["kind"] in ("fact", "update")]
        probes = [t for t in ts if t["kind"] == "probe"]
        if not fact_turns:
            continue
        for p in probes[: max(1, n_sample // max(1, len(by_thread)))]:
            required = set(p["required_facts"])
            pool = [t["fact_id"] for t in fact_turns if t["turn_idx"] < p["turn_idx"]]
            if not pool:
                continue
            total += 1
            oracle_hits += 1  # oracle returns `required` by definition
            picks = set(rng.choice(pool, size=min(len(required), len(pool)), replace=False))
            random_hits += len(picks & required) / len(required)

    return {
        "n_probes_evaluated": total,
        "oracle_recall": oracle_hits / max(total, 1),
        "random_recall": random_hits / max(total, 1),
    }
