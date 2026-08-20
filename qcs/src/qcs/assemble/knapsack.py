"""Budgeted, redundancy-aware selection — the layer that actually defeats saturation.

    maximise  Σ_{c∈S} r_c · v_c − λ · redundancy(S)
    s.t.      Σ_{c∈S} tokens(c) ≤ B

The objective is monotone submodular (a modular gain minus a max-similarity penalty), so
the greedy gain-per-token rule carries the standard (1 − 1/e) guarantee against the best
possible selection. That matters here because the alternative — "take the top-k scores" —
happily spends the whole budget on five paraphrases of the same fact.
"""

from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np


@dataclass
class Selection:
    idx: list[int] = field(default_factory=list)
    tokens_used: int = 0
    trace: list[dict] = field(default_factory=list)   # the V42 waterfall

    def __len__(self) -> int:
        return len(self.idx)


def greedy_knapsack(
    scores: np.ndarray,
    tokens: np.ndarray,
    budget: int,
    vectors: np.ndarray | None = None,
    value: np.ndarray | None = None,
    redundancy_lambda: float = 0.35,
    max_items: int | None = None,
) -> Selection:
    """Greedy selection by marginal gain per token."""
    n = len(scores)
    value = np.ones(n, dtype=np.float32) if value is None else np.asarray(value, np.float32)
    sel = Selection()
    chosen = np.zeros(n, dtype=bool)
    sims = None
    if vectors is not None:
        V = np.asarray(vectors, np.float32)
        V = V / (np.linalg.norm(V, axis=1, keepdims=True) + 1e-8)
        sims = V @ V.T

    while True:
        best_i, best_gain = -1, 0.0
        for i in range(n):
            if chosen[i] or sel.tokens_used + tokens[i] > budget:
                continue
            red = 0.0
            if sims is not None and sel.idx:
                red = float(sims[i, sel.idx].max())
            gain = scores[i] * value[i] - redundancy_lambda * red
            per_token = gain / max(int(tokens[i]), 1)
            if per_token > best_gain:
                best_gain, best_i = per_token, i
        if best_i < 0:
            break
        chosen[best_i] = True
        sel.idx.append(best_i)
        sel.tokens_used += int(tokens[best_i])
        sel.trace.append({
            "rank": len(sel.idx), "idx": int(best_i), "score": float(scores[best_i]),
            "value": float(value[best_i]), "tokens": int(tokens[best_i]),
            "gain_per_token": float(best_gain),
        })
        if max_items and len(sel.idx) >= max_items:
            break

    assert sel.tokens_used <= budget, "knapsack exceeded the token budget"
    return sel


def topk_baseline(scores: np.ndarray, tokens: np.ndarray, budget: int) -> Selection:
    """What everyone does: take the highest scores until the budget fills.

    Kept as an explicit comparison so the redundancy penalty has to prove it earns its
    place rather than being assumed useful.
    """
    sel = Selection()
    for i in np.argsort(-scores):
        if sel.tokens_used + tokens[i] > budget:
            continue
        sel.idx.append(int(i))
        sel.tokens_used += int(tokens[i])
    return sel


def value_prior(kinds: list[str], prior: dict[str, float]) -> np.ndarray:
    """Facts are worth more than filler at equal relevance."""
    default = prior.get("filler", 0.35)
    return np.array([prior.get("fact" if k in ("fact", "update") else k, default)
                     for k in kinds], dtype=np.float32)
