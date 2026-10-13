"""Hybrid Quantum Relevance Engine — a frozen similarity prior plus a quantum residual.

    logit(x) = g · s(x)  +  β ⟨Z_q⟩_θ(x)  +  b

`s(x)` is the query·candidate similarity, read straight out of the interaction block of
the 1024-d feature vector (the same quantity the no-learning cosine baseline uses). The
circuit is therefore not asked to rediscover similarity — a linear functional of the
amplitudes, which a unitary + local-Z readout (a quadratic form in x) cannot express — and
instead learns only what similarity cannot say: which candidate is *current*, who *owns*
the fact, and what the entity graph implies.

Split of responsibilities (the dual-manifold story, now at the scorer level):
    stability  — the prior `s` has no trainable direction except one scalar gain; it can
                 not be overwritten by later tasks, and the Fisher anchor protects θ.
    plasticity — θ lives on the compact angle manifold, so Tr(F) cannot collapse.
"""

from __future__ import annotations

import torch
from torch import nn

from ..config import Config
from .circuit import QRE
from .encoding import amplitude_encode
from .readout import readout_expectation


def similarity_block(feature_dim: int) -> slice:
    return slice(feature_dim // 2, 3 * feature_dim // 4)


class HybridQRE(QRE):
    def __init__(self, cfg: Config, seed: int | None = None, prior_gain: float = 6.0):
        super().__init__(cfg, seed=seed)
        self.block = similarity_block(cfg.encode.feature_dim)
        self.gain = nn.Parameter(torch.tensor(float(prior_gain)))

    def forward(self, x, theta=None, return_expectation: bool = False):
        x = torch.as_tensor(x, dtype=torch.float32) if not torch.is_tensor(x) else x.float()
        s = x[:, self.block].sum(dim=1)
        state = amplitude_encode(x, self.n, device=self.theta.device)
        state = self.evolve(state, theta)
        z = readout_expectation(state, self.n, self.readout_qubit, self.observable)
        r = torch.sigmoid(self.gain * s + self.beta * z + self.bias)
        return (r, z) if return_expectation else r


class MultiReadoutQRE(QRE):
    """QRE whose score is a learned weighted sum of ⟨Z_q⟩ over ALL qubits.

    Same circuit, same |θ| (plus n readout weights); only the measurement is richer. The
    weights start as the one-hot of the original readout qubit, so step 0 is identical to
    the baseline QRE.
    """

    def __init__(self, cfg: Config, seed: int | None = None):
        super().__init__(cfg, seed=seed)
        w = torch.zeros(self.n)
        w[self.readout_qubit] = 1.0
        self.readout_w = nn.Parameter(w)

    def forward(self, x, theta=None, return_expectation: bool = False):
        state = amplitude_encode(x, self.n, device=self.theta.device)
        state = self.evolve(state, theta)
        B = state.shape[0]
        p = (state.abs() ** 2).reshape(B, *([2] * self.n))
        zs = []
        for q in range(self.n):
            pq = p.sum(dim=[i + 1 for i in range(self.n) if i != q])
            zs.append(pq[:, 0] - pq[:, 1])
        z = torch.stack(zs, dim=1) @ self.readout_w
        r = torch.sigmoid(self.beta * z + self.bias)
        return (r, z) if return_expectation else r
