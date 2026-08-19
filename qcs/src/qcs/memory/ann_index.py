"""The candidate prefilter: cheap cosine retrieval that hands the scorer M candidates.

This layer sets a HARD CEILING on the whole system — a fact that never enters the
candidate set cannot be selected no matter how good the quantum scorer is. So it is
measured explicitly (figure V10) and the exit criterion for Phase 3 is recall ≥ 0.95 at
M = 256.

Exact inner-product search is used deliberately: at ~15k units per corpus it costs
microseconds, and an approximate index would confound "the scorer missed it" with "the
index never showed it". `use_ivf` switches to an approximate index for the scaling study
in Phase 9.
"""

from __future__ import annotations

import numpy as np

try:
    import faiss
    HAVE_FAISS = True
except ImportError:  # pragma: no cover - environment dependent
    HAVE_FAISS = False


class ANNIndex:
    def __init__(self, dim: int, use_ivf: bool = False, nlist: int = 64):
        self.dim = dim
        self.use_ivf = use_ivf
        self.nlist = nlist
        self.ids: list[str] = []
        self._vecs: np.ndarray | None = None
        self._index = None

    @staticmethod
    def _norm(X: np.ndarray) -> np.ndarray:
        X = np.ascontiguousarray(np.asarray(X, dtype=np.float32))
        return X / (np.linalg.norm(X, axis=1, keepdims=True) + 1e-8)

    def add(self, ids: list[str], vecs: np.ndarray) -> None:
        V = self._norm(vecs)
        self.ids.extend(ids)
        self._vecs = V if self._vecs is None else np.concatenate([self._vecs, V])
        self._index = None  # rebuilt lazily

    def build(self) -> None:
        if not HAVE_FAISS or self._vecs is None:
            return
        if self.use_ivf and len(self._vecs) > 8 * self.nlist:
            quant = faiss.IndexFlatIP(self.dim)
            idx = faiss.IndexIVFFlat(quant, self.dim, self.nlist, faiss.METRIC_INNER_PRODUCT)
            idx.train(self._vecs)
            idx.add(self._vecs)
            idx.nprobe = max(1, self.nlist // 8)
        else:
            idx = faiss.IndexFlatIP(self.dim)
            idx.add(self._vecs)
        self._index = idx

    def search(self, query: np.ndarray, k: int, mask: np.ndarray | None = None
               ) -> tuple[list[str], np.ndarray]:
        """Top-k by cosine. `mask` (bool over rows) restricts the searchable set — used to
        enforce causality: a candidate from the future must never be retrievable."""
        q = self._norm(np.atleast_2d(query))[0]
        if mask is not None:
            rows = np.flatnonzero(mask)
            sims = self._vecs[rows] @ q
            k = min(k, len(rows))
            top = rows[np.argpartition(-sims, k - 1)[:k]] if k > 0 else np.array([], int)
            order = top[np.argsort(-(self._vecs[top] @ q))]
            return [self.ids[i] for i in order], self._vecs[order] @ q
        if self._index is None:
            self.build()
        if self._index is not None:
            sims, idx = self._index.search(q[None, :], min(k, len(self.ids)))
            return [self.ids[i] for i in idx[0]], sims[0]
        sims = self._vecs @ q
        k = min(k, len(sims))
        top = np.argpartition(-sims, k - 1)[:k]
        order = top[np.argsort(-sims[top])]
        return [self.ids[i] for i in order], sims[order]

    def __len__(self) -> int:
        return len(self.ids)
