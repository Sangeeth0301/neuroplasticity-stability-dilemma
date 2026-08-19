"""S7 · amplitude encoding: 1024 real numbers → the amplitudes of a 10-qubit state.

    |ψ₀⟩ = Σ_{j=0}^{1023} x_j |j⟩,   with ‖x‖₂ = 1

This is lossless precisely because the feature dimension equals 2^N — every real number
becomes exactly one amplitude, and no qubit is left holding vacuum. `Config.validate()`
refuses any configuration where that identity does not hold.

On hardware the state-preparation circuit is O(2^N) deep, which is the real bottleneck
for scaling this design beyond simulation; that limitation is recorded rather than hidden.
"""

from __future__ import annotations

import numpy as np
import torch

from .gates import CDTYPE


def amplitude_encode(x: np.ndarray | torch.Tensor, n_qubits: int,
                     device=None, check: bool = True) -> torch.Tensor:
    """(B, 2**n) real, unit-norm → (B, 2**n) complex statevector."""
    if isinstance(x, np.ndarray):
        x = torch.from_numpy(np.ascontiguousarray(x, dtype=np.float32))
    x = torch.atleast_2d(x).to(device=device, dtype=torch.float32)
    dim = 2**n_qubits
    if x.shape[1] != dim:
        raise ValueError(f"amplitude encoding needs exactly {dim} features, got {x.shape[1]}")
    if check:
        err = (x.norm(dim=1) - 1.0).abs().max().item()
        if err > 1e-4:
            raise ValueError(f"input is not unit-norm (max deviation {err:.2e}) — "
                             "it is not a valid quantum state")
    return x.to(CDTYPE)


def probabilities(state: torch.Tensor) -> torch.Tensor:
    return state.abs() ** 2


def participation_ratio(state: torch.Tensor) -> torch.Tensor:
    """How many basis states carry the amplitude: 1 = one state, 2**n = uniform."""
    p = probabilities(state)
    p = p / p.sum(dim=1, keepdim=True)
    return 1.0 / (p**2).sum(dim=1)


def entanglement_entropy(state: torch.Tensor, cut: int, n_qubits: int) -> torch.Tensor:
    """Von Neumann entropy (bits) across the bipartition [0..cut) | [cut..n)."""
    B = state.shape[0]
    m = state.reshape(B, 2**cut, 2 ** (n_qubits - cut))
    s = torch.linalg.svdvals(m)
    p = (s**2).clamp_min(1e-15)
    p = p / p.sum(dim=1, keepdim=True)
    return -(p * torch.log2(p)).sum(dim=1)


def page_entropy(cut: int, n_qubits: int) -> float:
    """Maximum possible entropy for that cut — the ceiling in figure V18."""
    return float(min(cut, n_qubits - cut))
