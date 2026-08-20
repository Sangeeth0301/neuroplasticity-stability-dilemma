"""The classical arms — every one implements the same interface as the QRE.

    score(X: (M, 1024) unit-norm) -> r: (M,) in [0,1]

That identical signature is what makes the quantum claim falsifiable: an A/B between
`QRE` and `MLPScorer` changes exactly one component and nothing else in the pipeline.

On parameter matching. The QRE at depth 8 has |θ| = 1,080. A classical network consuming
the same 1024-dimensional input needs ≥1024 weights for a single linear layer, so the
honest parameter-matched twin is a linear scorer (1,025 params) — the closest classical
model of equal size. A wider MLP is ALSO reported as a capacity reference, clearly
labelled, because hiding it would be the kind of handicapped baseline that makes quantum
papers unconvincing.
"""

from __future__ import annotations

import numpy as np
import torch
from torch import nn


class BaseScorer(nn.Module):
    name = "base"
    trainable = True

    def forward(self, x) -> torch.Tensor:      # pragma: no cover - interface
        raise NotImplementedError

    @torch.no_grad()
    def score(self, X: np.ndarray) -> np.ndarray:
        return self.forward(X).cpu().numpy()

    @property
    def n_circuit_params(self) -> int:
        return sum(p.numel() for p in self.parameters())

    def wrap_angles(self) -> None:
        """No-op: Euclidean parameters have nothing to wrap. That asymmetry IS the
        experiment — the classical arm's weights are free to grow without bound."""

    @torch.no_grad()
    def theta_norm(self, wrapped: bool = True) -> float:
        return float(torch.sqrt(sum((p.detach() ** 2).sum() for p in self.parameters())))

    def grad_norm(self) -> float:
        return float(torch.sqrt(sum((p.grad**2).sum() for p in self.parameters()
                                    if p.grad is not None)))

    @staticmethod
    def _t(x) -> torch.Tensor:
        return torch.as_tensor(np.ascontiguousarray(x), dtype=torch.float32) \
            if isinstance(x, np.ndarray) else x.float()


class LinearScorer(BaseScorer):
    """Parameter-matched twin: 1024 → 1, 1,025 parameters vs the QRE's 1,080."""

    name = "linear"

    def __init__(self, in_dim: int = 1024, seed: int = 0):
        super().__init__()
        torch.manual_seed(seed)
        self.fc = nn.Linear(in_dim, 1)
        nn.init.normal_(self.fc.weight, 0.0, 0.02)
        nn.init.zeros_(self.fc.bias)
        # `theta` alias so the training loop, anchor and Fisher code are arm-agnostic
        self.theta = self.fc.weight

    def forward(self, x) -> torch.Tensor:
        return torch.sigmoid(self.fc(self._t(x)).squeeze(-1))


class MLPScorer(BaseScorer):
    """Capacity reference: 1024 → h → 1 with ReLU. Reported separately, never as 'the'
    classical arm."""

    name = "mlp"

    def __init__(self, in_dim: int = 1024, hidden: int = 64, seed: int = 0):
        super().__init__()
        torch.manual_seed(seed)
        self.net = nn.Sequential(nn.Linear(in_dim, hidden), nn.ReLU(), nn.Linear(hidden, 1))
        self.theta = self.net[0].weight

    def forward(self, x) -> torch.Tensor:
        return torch.sigmoid(self.net(self._t(x)).squeeze(-1))


class CosineScorer(BaseScorer):
    """B1 — no learning at all. Reads the query·candidate similarity straight out of the
    interaction block, which is exactly `p_q ⊙ p_c` summed."""

    name = "cosine"
    trainable = False

    def __init__(self, feature_dim: int = 1024):
        super().__init__()
        self.block = slice(feature_dim // 2, 3 * feature_dim // 4)

    def forward(self, x) -> torch.Tensor:
        x = self._t(x)
        # gain 20 saturated sigmoid to exactly 1.0 for most candidates, producing
        # ties that make AUC meaningless; 6 keeps the ordering informative.
        return torch.sigmoid(6.0 * x[:, self.block].sum(dim=1))


class RecencyScorer(BaseScorer):
    """B0 — 'just use the most recent turns'. The truncation floor, expressed as a scorer
    so it runs through the identical harness."""

    name = "recency"
    trainable = False

    def __init__(self, feature_dim: int = 1024):
        super().__init__()
        # ρ starts at 3/4 of the vector; its second slot is the sharp 1/(1+gap) feature
        self.idx = 3 * feature_dim // 4 + 65

    def forward(self, x) -> torch.Tensor:
        return torch.sigmoid(6.0 * self._t(x)[:, self.idx])


class RandomScorer(BaseScorer):
    """Chance floor."""

    name = "random"
    trainable = False

    def __init__(self, seed: int = 0):
        super().__init__()
        self.seed = seed

    def forward(self, x) -> torch.Tensor:
        g = torch.Generator().manual_seed(self.seed)
        return torch.rand(len(self._t(x)), generator=g)


class OracleScorer(BaseScorer):
    """Upper bound: given the labels. Not a model — the ceiling any scorer is measured
    against."""

    name = "oracle"
    trainable = False

    def forward(self, x) -> torch.Tensor:      # pragma: no cover - never called
        raise RuntimeError("OracleScorer needs labels; use score_with_labels")

    @staticmethod
    def score_with_labels(y: np.ndarray) -> np.ndarray:
        return y.astype(np.float32)


SCORERS = {
    "linear": LinearScorer,
    "mlp": MLPScorer,
    "cosine": CosineScorer,
    "recency": RecencyScorer,
    "random": RandomScorer,
    "oracle": OracleScorer,
}


def build_scorer(name: str, cfg, seed: int = 0) -> BaseScorer:
    """`quantum` is handled by the caller (it needs the full Config); everything else
    comes from here."""
    if name in ("linear", "mlp"):
        return SCORERS[name](cfg.encode.feature_dim, seed=seed)
    if name in ("cosine", "recency"):
        return SCORERS[name](cfg.encode.feature_dim)
    return SCORERS[name]()
