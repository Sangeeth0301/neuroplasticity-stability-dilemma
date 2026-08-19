"""How much room is there for a learned scorer at all?

Phase 3 measured prefilter recall (the ceiling). This measures the opposite quantity:
what pure cosine similarity CANNOT do, and therefore what the learned scorer — quantum
or classical — has to earn.

The decisive number is `stale_outranks_current`. When a fact is superseded, the old and
new statements are near-identical text about the same entity, so semantic similarity is
blind between them. Only the relational block (recency, supersession edges) can break
the tie. If this number is high for cosine and low for the scorer, the architecture is
doing what it claims.
"""

from __future__ import annotations

import numpy as np

from ..types import Fact, SemanticUnit


def cosine_headroom(
    sus: list[SemanticUnit],
    proj: dict[str, np.ndarray],
    required_of: dict[tuple[str, int], list[str]],
    facts: dict[str, Fact],
) -> dict:
    """Rank-1 / recall@5 for cosine, overall, by regime, and on supersession cases."""
    by_thread: dict[str, list[SemanticUnit]] = {}
    for s in sus:
        by_thread.setdefault(s.thread_id, []).append(s)

    tot = {"n": 0, "r1": 0, "r5": 0}
    by_regime: dict[str, list[int]] = {}
    by_band: dict[int, list[int]] = {}
    stale_cases = stale_wins = 0

    for tid, S in by_thread.items():
        S = sorted(S, key=lambda x: (x.turn_idx, x.su_idx))
        V = np.stack([proj[s.su_id] for s in S])
        V = V / (np.linalg.norm(V, axis=1, keepdims=True) + 1e-8)
        pos = {s.su_id: i for i, s in enumerate(S)}
        thread_facts = [f for f in facts.values() if f.thread_id == tid]
        regime = thread_facts[0].regime if thread_facts else "unknown"

        for s in S:
            if s.kind != "probe":
                continue
            gold = set(required_of.get((s.thread_id, s.turn_idx), []))
            cands = [c for c in S if c.turn_idx < s.turn_idx]
            if not gold or not cands:
                continue
            sims = V[[pos[c.su_id] for c in cands]] @ V[pos[s.su_id]]
            ranked = [cands[i] for i in np.argsort(-sims)]
            hit1 = ranked[0].fact_id in gold
            hit5 = any(c.fact_id in gold for c in ranked[:5])
            tot["n"] += 1
            tot["r1"] += int(hit1)
            tot["r5"] += int(hit5)
            by_regime.setdefault(regime, []).append(int(hit1))

            gold_turns = [c.turn_idx for c in cands if c.fact_id in gold]
            if gold_turns:
                band = s.turn_idx - max(gold_turns)
                by_band.setdefault(_band_of(band), []).append(int(hit1))

            old_ids = {facts[g].supersedes for g in gold
                       if g in facts and facts[g].supersedes}
            if old_ids:
                stale_cases += 1
                first_gold = min(i for i, c in enumerate(ranked) if c.fact_id in gold)
                stale_pos = [i for i, c in enumerate(ranked) if c.fact_id in old_ids]
                if stale_pos and min(stale_pos) < first_gold:
                    stale_wins += 1

    return {
        "n_probes": tot["n"],
        "rank1": tot["r1"] / max(tot["n"], 1),
        "recall5": tot["r5"] / max(tot["n"], 1),
        "rank1_by_regime": {k: float(np.mean(v)) for k, v in by_regime.items()},
        "n_by_regime": {k: len(v) for k, v in by_regime.items()},
        "rank1_by_band": {str(k): float(np.mean(v)) for k, v in sorted(by_band.items())},
        "supersession_cases": stale_cases,
        "stale_outranks_current": stale_wins / max(stale_cases, 1),
        "headroom": 1.0 - tot["r1"] / max(tot["n"], 1),
    }


def _band_of(d: int) -> int:
    for b in (10, 50, 200, 500):
        if d <= b * 1.6:
            return b
    return 1000
