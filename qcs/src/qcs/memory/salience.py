"""Per-participant salience: what each user's own turns are about.

This is what makes the scorer participant-aware and is the direct countermeasure to
persona bleed (attributing user A's stated preference to user B). It is an EMA over the
projected embeddings of a participant's own turns, so it tracks a drifting interest
without storing anything that could be forgotten — it is recomputable from ℳ at any time.
"""

from __future__ import annotations

import numpy as np

from ..types import SemanticUnit
from .store import MemoryStore

SALIENCE_DIM = 64


def compress(vec: np.ndarray, dim: int = SALIENCE_DIM) -> np.ndarray:
    """Fold a projected vector down to the 64 dims ρ reserves for salience.

    Deterministic block-mean folding rather than a learned map: this must stay frozen
    and inspectable, and a learned compressor here would be one more thing that drifts.
    """
    v = np.asarray(vec, dtype=np.float32).ravel()
    reps = int(np.ceil(len(v) / dim))
    padded = np.pad(v, (0, reps * dim - len(v)))
    return padded.reshape(reps, dim).mean(axis=0)


class SalienceTracker:
    def __init__(self, alpha: float = 0.05, dim: int = SALIENCE_DIM):
        self.alpha = alpha
        self.dim = dim
        self.vecs: dict[tuple[str, str], np.ndarray] = {}
        self.counts: dict[tuple[str, str], int] = {}

    def update(self, su: SemanticUnit, proj_vec: np.ndarray) -> np.ndarray:
        key = (su.thread_id, su.user_id)
        v = self.vecs.get(key, np.zeros(self.dim, dtype=np.float32))
        new = (1 - self.alpha) * v + self.alpha * compress(proj_vec, self.dim)
        self.vecs[key] = new.astype(np.float32)
        self.counts[key] = self.counts.get(key, 0) + 1
        return self.vecs[key]

    def get(self, thread_id: str, user_id: str) -> np.ndarray:
        return self.vecs.get((thread_id, user_id), np.zeros(self.dim, dtype=np.float32))

    def replay_thread(self, sus: list[SemanticUnit], proj: dict[str, np.ndarray],
                      up_to_turn: int | None = None) -> None:
        for su in sorted(sus, key=lambda s: (s.turn_idx, s.su_idx)):
            if up_to_turn is not None and su.turn_idx >= up_to_turn:
                break
            if su.su_id in proj:
                self.update(su, proj[su.su_id])

    def persist(self, store: MemoryStore) -> None:
        for (tid, uid), v in self.vecs.items():
            store.set_salience(tid, uid, v, self.counts[(tid, uid)])

    def separation(self) -> float:
        """Mean cosine distance between different users' salience vectors in a thread.

        If this is ~0 the salience block carries no participant information and persona
        bleed cannot be reduced by it — the V12 pass/fail number.
        """
        by_thread: dict[str, list[np.ndarray]] = {}
        for (tid, _), v in self.vecs.items():
            by_thread.setdefault(tid, []).append(v)
        dists = []
        for vs in by_thread.values():
            for i in range(len(vs)):
                for j in range(i + 1, len(vs)):
                    a, b = vs[i], vs[j]
                    na, nb = np.linalg.norm(a), np.linalg.norm(b)
                    if na > 0 and nb > 0:
                        dists.append(1.0 - float(a @ b / (na * nb)))
        return float(np.mean(dists)) if dists else 0.0
