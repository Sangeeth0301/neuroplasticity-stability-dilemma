"""The metric suite. Defined once, computed identically for every arm.

Ranking metrics are computed per probe and then averaged over probes — never pooled over
candidates, which would let probes with many candidates dominate.
"""

from __future__ import annotations

import numpy as np


def rank_of_first_positive(scores: np.ndarray, y: np.ndarray) -> int | None:
    order = np.argsort(-scores)
    hits = np.flatnonzero(y[order] > 0)
    return int(hits[0]) if len(hits) else None


def recall_at_k(scores: np.ndarray, y: np.ndarray, k: int) -> float:
    if y.sum() == 0:
        return float("nan")
    top = np.argsort(-scores)[:k]
    return float(y[top].sum() / y.sum())


def mrr(scores: np.ndarray, y: np.ndarray) -> float:
    r = rank_of_first_positive(scores, y)
    return 0.0 if r is None else 1.0 / (r + 1)


def auc(scores: np.ndarray, y: np.ndarray) -> float:
    """Rank-based AUC (Mann-Whitney), tie-safe."""
    pos, neg = scores[y > 0], scores[y == 0]
    if len(pos) == 0 or len(neg) == 0:
        return float("nan")
    order = np.argsort(np.concatenate([pos, neg]))
    ranks = np.empty(len(order), float)
    ranks[order] = np.arange(1, len(order) + 1)
    r_pos = ranks[: len(pos)].sum()
    return float((r_pos - len(pos) * (len(pos) + 1) / 2) / (len(pos) * len(neg)))


def stale_above_current(scores: np.ndarray, y: np.ndarray, stale: np.ndarray) -> float | None:
    """Does a superseded fact outrank the current one? The measurement that motivates ρ."""
    if stale.sum() == 0 or y.sum() == 0:
        return None
    order = np.argsort(-scores)
    first_gold = np.flatnonzero(y[order] > 0)
    first_stale = np.flatnonzero(stale[order] > 0)
    if not len(first_gold) or not len(first_stale):
        return None
    return float(first_stale[0] < first_gold[0])


def evaluate_probes(score_fn, examples, ks=(1, 3, 5, 10)) -> dict:
    """score_fn(example) -> scores aligned with example.y."""
    acc: dict[str, list[float]] = {f"recall@{k}": [] for k in ks}
    acc.update({"mrr": [], "auc": [], "rank1": []})
    stale_flags: list[float] = []

    for ex in examples:
        s = np.asarray(score_fn(ex), dtype=np.float64)
        for k in ks:
            acc[f"recall@{k}"].append(recall_at_k(s, ex.y, k))
        acc["mrr"].append(mrr(s, ex.y))
        a = auc(s, ex.y)
        if not np.isnan(a):
            acc["auc"].append(a)
        acc["rank1"].append(float(rank_of_first_positive(s, ex.y) == 0))
        sa = stale_above_current(s, ex.y, ex.is_stale)
        if sa is not None:
            stale_flags.append(sa)

    out = {k: float(np.nanmean(v)) if v else float("nan") for k, v in acc.items()}
    out["stale_above_current"] = float(np.mean(stale_flags)) if stale_flags else float("nan")
    out["n_probes"] = len(examples)
    out["n_stale_cases"] = len(stale_flags)
    return out


def persona_bleed(score_fn, examples, owner_of: dict[str, str],
                  su_owner: dict[str, str]) -> float:
    """Fraction of probes whose top-ranked candidate was authored by a participant other
    than the owner of the required fact."""
    bad = tot = 0
    for ex in examples:
        gold = np.flatnonzero(ex.y > 0)
        if not len(gold):
            continue
        gold_owner = su_owner.get(ex.su_ids[gold[0]])
        top = ex.su_ids[int(np.argmax(np.asarray(score_fn(ex))))]
        if gold_owner is None:
            continue
        tot += 1
        bad += int(su_owner.get(top) != gold_owner)
    _ = owner_of
    return bad / max(tot, 1)
