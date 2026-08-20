"""Hardware-realistic noise: depolarising channels by trajectory sampling.

The reference paper reports that the plasticity advantage survives depolarising noise at
p = 5e-4. This module lets us check the same thing on the trained scorer, which is the
difference between "works in an exact simulator" and "might work on a device".

Implementation is stochastic unravelling rather than a density matrix: after every
two-qubit gate, with probability p, a random non-identity Pauli is applied to one of the
two qubits. Averaging ⟨Z⟩ over R trajectories estimates the noisy expectation. A density
matrix would be exact but costs 4^N = 10^6 entries at N=10; trajectories cost R × 2^N and
converge at 1/√R, which is enough to see whether ranking survives.
"""

from __future__ import annotations

import numpy as np
import torch

from .encoding import amplitude_encode
from .gates import CDTYPE, apply_2q
from .readout import readout_expectation

_PAULIS = {
    "X": torch.tensor([[0, 1], [1, 0]], dtype=CDTYPE),
    "Y": torch.tensor([[0, -1j], [1j, 0]], dtype=CDTYPE),
    "Z": torch.tensor([[1, 0], [0, -1]], dtype=CDTYPE),
}


def _apply_1q(state: torch.Tensor, U: torch.Tensor, q: int, n: int) -> torch.Tensor:
    B = state.shape[0]
    st = state.reshape(B, *([2] * n))
    perm = [0, q + 1] + [i + 1 for i in range(n) if i != q]
    st = st.permute(perm).reshape(B, 2, -1)
    st = torch.matmul(U, st).reshape(B, 2, *([2] * (n - 1)))
    inv = [0] * (n + 1)
    for pos, src in enumerate(perm):
        inv[src] = pos
    return st.permute(inv).reshape(B, 2**n)


@torch.no_grad()
def noisy_expectation(model, x: np.ndarray, p: float, trajectories: int = 8,
                      seed: int = 0) -> np.ndarray:
    """⟨Z⟩ under per-gate depolarising noise of strength p, averaged over trajectories."""
    if p <= 0:
        _, z = model(x, return_expectation=True)
        return z.cpu().numpy()

    rng = np.random.default_rng(seed)
    U = model.gates()
    acc = None
    for _ in range(trajectories):
        state = amplitude_encode(x, model.n, device=model.theta.device)
        for layer in range(model.depth):
            for gi, (q0, q1) in enumerate(model.pairs):
                state = apply_2q(state, U[layer, gi], q0, q1, model.n)
                if rng.random() < p:
                    q = q0 if rng.random() < 0.5 else q1
                    pauli = _PAULIS[str(rng.choice(["X", "Y", "Z"]))]
                    state = _apply_1q(state, pauli, q, model.n)
        z = readout_expectation(state, model.n, model.readout_qubit,
                                model.observable).cpu().numpy()
        acc = z if acc is None else acc + z
    return acc / trajectories


@torch.no_grad()
def noisy_score(model, x: np.ndarray, p: float, trajectories: int = 8,
                seed: int = 0) -> np.ndarray:
    z = noisy_expectation(model, x, p, trajectories, seed)
    beta = float(model.beta.detach())
    bias = float(model.bias.detach())
    return 1.0 / (1.0 + np.exp(-(beta * z + bias)))
