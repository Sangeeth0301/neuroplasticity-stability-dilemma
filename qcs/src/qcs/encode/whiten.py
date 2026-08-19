"""PCA whitening — fit ONCE, on held-out data, then frozen.

Why it matters here more than in an ordinary NLP pipeline: raw sentence embeddings are
strongly anisotropic (a few directions carry most of the variance). Amplitude encoding
turns vector components directly into quantum amplitudes, so an anisotropic input means
a handful of basis states absorb almost all the probability mass and most of the
register is left in vacuum. Whitening is what makes the qubits carry signal.

Refitting during a continual run would be a silent, untracked distribution shift, so
`transform()` refuses to run against an unfitted object and `fit()` refuses to run twice.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np


class Whitener:
    def __init__(self, n_components: int, eps: float = 1e-6):
        self.n_components = n_components
        self.eps = eps
        self.mu: np.ndarray | None = None
        self.W: np.ndarray | None = None
        self.explained: np.ndarray | None = None
        self.raw_spectrum: np.ndarray | None = None

    @property
    def fitted(self) -> bool:
        return self.W is not None

    def fit(self, X: np.ndarray) -> "Whitener":
        if self.fitted:
            raise RuntimeError("Whitener already fitted — refitting mid-run is forbidden")
        X = np.asarray(X, dtype=np.float64)
        self.mu = X.mean(axis=0)
        Xc = X - self.mu
        # SVD rather than an explicit covariance: better conditioned for n < d
        U, S, Vt = np.linalg.svd(Xc, full_matrices=False)
        k = min(self.n_components, Vt.shape[0])
        var = (S**2) / max(len(X) - 1, 1)
        self.raw_spectrum = var
        self.explained = var[:k] / var.sum()
        self.W = (Vt[:k] / np.sqrt(var[:k] + self.eps)[:, None]).astype(np.float32)
        return self

    def transform(self, X: np.ndarray) -> np.ndarray:
        if not self.fitted:
            raise RuntimeError("Whitener not fitted")
        return ((np.asarray(X, dtype=np.float32) - self.mu.astype(np.float32)) @ self.W.T)

    def condition_number(self, X: np.ndarray) -> tuple[float, float]:
        """Spectrum condition number before and after — the V06 pass/fail number."""
        def cond(M: np.ndarray) -> float:
            s = np.linalg.svd(M - M.mean(0), compute_uv=False)
            return float(s[0] / max(s[min(len(s) - 1, self.n_components - 1)], 1e-12))
        return cond(np.asarray(X, dtype=np.float64)), cond(self.transform(X).astype(np.float64))

    def save(self, path: str | Path) -> None:
        Path(path).parent.mkdir(parents=True, exist_ok=True)
        np.savez(path, mu=self.mu, W=self.W, explained=self.explained,
                 raw_spectrum=self.raw_spectrum, n_components=self.n_components)

    @classmethod
    def load(cls, path: str | Path) -> "Whitener":
        blob = np.load(path)
        w = cls(int(blob["n_components"]))
        w.mu, w.W = blob["mu"], blob["W"]
        w.explained, w.raw_spectrum = blob["explained"], blob["raw_spectrum"]
        return w
