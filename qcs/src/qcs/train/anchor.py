"""Stability machinery for the plastic core: a Fisher-weighted anchor, a trust region,
and a drift detector.

The quantum geometry buys plasticity and nothing else — the reference paper is explicit
that VQCs still forget catastrophically without an explicit stability mechanism. This
module is that mechanism, and `configs/ablations/no_anchor.yaml` (λ_Ω = 0) exists so the
claim can be tested rather than asserted.

    Ω(θ, θ*) = ½ Σ_k F_k (θ_k − θ*_k)²

For the quantum arm F is the diagonal of the QUANTUM Fisher information — the metric of
the state manifold — because Euclidean EWC would measure distance in the wrong geometry
on a curved parameter space. For the classical arm the empirical FIM diagonal is used,
which is ordinary EWC.
"""

from __future__ import annotations

import numpy as np
import torch


class FisherAnchor:
    def __init__(self, lambda_omega: float):
        self.lam = lambda_omega
        self.theta_star: torch.Tensor | None = None
        self.fisher: torch.Tensor | None = None
        self.n_checkpoints = 0

    def checkpoint(self, model, fisher_diag: np.ndarray) -> None:
        self.theta_star = model.theta.detach().clone()
        f = torch.as_tensor(np.asarray(fisher_diag, dtype=np.float32)).reshape(
            self.theta_star.shape)
        # normalise so λ_Ω means the same thing regardless of the Fisher's raw scale
        self.fisher = f / (f.mean() + 1e-12)
        self.n_checkpoints += 1

    def penalty(self, model) -> torch.Tensor:
        if self.theta_star is None or self.lam == 0.0:
            return torch.zeros((), dtype=torch.float32)
        d = model.theta - self.theta_star
        return 0.5 * self.lam * (self.fisher * d**2).sum()

    def pull(self, model) -> float:
        """‖θ − θ*‖ — plotted as V26; a sawtooth means checkpoints are firing."""
        if self.theta_star is None:
            return 0.0
        return float((model.theta.detach() - self.theta_star).norm())


class TrustRegion:
    """Clip the per-task parameter movement so one conversational shift cannot rotate the
    circuit far. Reports the clip rate, which is itself diagnostic: >50% means δ is
    strangling learning, ~0% means it is doing nothing."""

    def __init__(self, delta: float):
        self.delta = delta
        self.n_steps = 0
        self.n_clipped = 0

    def start_task(self, model) -> torch.Tensor:
        return model.theta.detach().clone()

    @torch.no_grad()
    def apply(self, model, theta_start: torch.Tensor) -> None:
        self.n_steps += 1
        d = model.theta - theta_start
        m = float(d.abs().max())
        if m > self.delta:
            self.n_clipped += 1
            model.theta.copy_(theta_start + d * (self.delta / m))

    @property
    def clip_rate(self) -> float:
        return self.n_clipped / max(self.n_steps, 1)


class DriftDetector:
    """Population-stability index on the score distribution.

    Fires when what the scorer considers relevant has shifted enough that the anchor
    should be re-based; that is what makes θ* track the conversation instead of pinning
    the model to its first hundred turns.
    """

    def __init__(self, threshold: float, n_bins: int = 10):
        self.threshold = threshold
        self.n_bins = n_bins
        self.reference: np.ndarray | None = None
        self.n_fired = 0

    def psi(self, scores: np.ndarray) -> float:
        hist = np.histogram(scores, bins=self.n_bins, range=(0, 1))[0].astype(float)
        hist = (hist + 1e-6) / (hist.sum() + 1e-6 * self.n_bins)
        if self.reference is None:
            return 0.0
        r = self.reference
        return float(((hist - r) * np.log(hist / r)).sum())

    def update(self, scores: np.ndarray) -> tuple[float, bool]:
        value = self.psi(scores)
        fired = self.reference is not None and value > self.threshold
        if self.reference is None or fired:
            hist = np.histogram(scores, bins=self.n_bins, range=(0, 1))[0].astype(float)
            self.reference = (hist + 1e-6) / (hist.sum() + 1e-6 * self.n_bins)
        self.n_fired += int(fired)
        return value, fired
