"""Categorise losses, so the next sprint is aimed at the layer that is actually failing.

Four categories, checked in pipeline order — the first one that applies wins, because a
retrieval miss makes every downstream question moot:

  retrieval   the required fact was not in the candidate set at all  (fix ℳ / the prefilter)
  ranking     it was present but the scorer put something else first (fix the scorer)
  stale       a superseded version outranked the current fact        (fix ρ / the DAG)
  eviction    it ranked well but the knapsack dropped it             (fix budget or value prior)
"""

from __future__ import annotations

from collections import Counter

import numpy as np


def categorise(scores: np.ndarray, ex, selected_idx: set[int] | None = None) -> str | None:
    """Returns None when the probe was answered correctly."""
    gold = np.flatnonzero(ex.y > 0)
    if len(gold) == 0:
        return "retrieval"
    order = np.argsort(-scores)
    top = int(order[0])
    if top in set(gold.tolist()):
        if selected_idx is not None and not (set(gold.tolist()) & selected_idx):
            return "eviction"
        return None
    if ex.is_stale[top] > 0:
        return "stale"
    return "ranking"


def analyse(score_fn, examples, selector=None) -> dict:
    counts: Counter[str] = Counter()
    total = 0
    examples_by_kind: dict[str, list] = {}
    for ex in examples:
        s = np.asarray(score_fn(ex), dtype=np.float64)
        sel = selector(ex, s) if selector else None
        kind = categorise(s, ex, sel)
        total += 1
        if kind:
            counts[kind] += 1
            examples_by_kind.setdefault(kind, []).append(
                {"thread": ex.thread_id, "turn": int(ex.probe_turn)})
    return {
        "n_probes": total,
        "n_failures": int(sum(counts.values())),
        "accuracy": 1.0 - sum(counts.values()) / max(total, 1),
        "counts": dict(counts),
        "examples": {k: v[:5] for k, v in examples_by_kind.items()},
    }
