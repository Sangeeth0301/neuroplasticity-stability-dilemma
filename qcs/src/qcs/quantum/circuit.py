"""The Quantum Relevance Engine (QRE) — the plasticity manifold 𝒰.

This is the ONLY component that learns online. Its parameters are rotation angles on a
compact periodic manifold, which is the entire reason it is here: unitarity bounds the
operator norm at 1, angles wrap instead of growing without bound, and the Fisher
information is bounded away from zero, so learning capacity cannot decay the way an
unbounded Euclidean network's does.

    x (1024, unit norm) ─► |ψ₀⟩ ─► U(θ) = Π_layers Π_pairs V(θ) ─► ⟨Z_q⟩ ─► σ(β⟨Z⟩+b)

The output is a single relevance score per candidate. That narrow interface is what makes
the quantum claim falsifiable: `classical/mlp_scorer.py` implements the identical
signature with a matched parameter count.
"""

from __future__ import annotations

import numpy as np
import torch
from torch import nn

from ..config import Config
from .ansatz import pairs_for
from .encoding import amplitude_encode
from .gates import CDTYPE, apply_2q, pauli_basis, su4
from .readout import readout_expectation


class QRE(nn.Module):
    def __init__(self, cfg: Config, seed: int | None = None):
        super().__init__()
        q = cfg.quantum
        self.n = q.n_qubits
        self.depth = q.depth
        self.observable = q.observable
        self.readout_qubit = q.readout_qubit
        self.pairs = pairs_for(q.ansatz, self.n)
        self.gate_form = q.gate_form
        g = torch.Generator().manual_seed(cfg.seed if seed is None else seed)

        # Identity-block initialisation: θ ≈ 0 ⇒ U ≈ I, so the first gradients are
        # informative instead of arriving from a Haar-random scrambler.
        theta = q.init_scale * torch.randn(self.depth, len(self.pairs), 15, generator=g)
        self.theta = nn.Parameter(theta)
        self.beta = nn.Parameter(torch.tensor(float(q.beta_init)))
        self.bias = nn.Parameter(torch.tensor(float(q.bias_init)))
        self.register_buffer("basis", pauli_basis(), persistent=False)

    # ------------------------------------------------------------------ #
    @property
    def n_circuit_params(self) -> int:
        return int(self.theta.numel())

    def gates(self, theta: torch.Tensor | None = None) -> torch.Tensor:
        """(depth, n_pairs, 4, 4) unitaries for the current angles."""
        return su4(self.theta if theta is None else theta, self.basis, self.gate_form)

    def evolve(self, state: torch.Tensor, theta: torch.Tensor | None = None,
               up_to: int | None = None, keep: bool = False):
        """Run the circuit. `keep=True` returns the state after every layer (for V18/V20b)."""
        U = self.gates(theta)
        states = [state] if keep else None
        for layer in range(self.depth if up_to is None else up_to):
            for gi, (q0, q1) in enumerate(self.pairs):
                state = apply_2q(state, U[layer, gi], q0, q1, self.n)
            if keep:
                states.append(state)
        return (state, states) if keep else state

    def forward(self, x: torch.Tensor | np.ndarray, theta: torch.Tensor | None = None,
                return_expectation: bool = False):
        state = amplitude_encode(x, self.n, device=self.theta.device)
        state = self.evolve(state, theta)
        z = readout_expectation(state, self.n, self.readout_qubit, self.observable)
        r = torch.sigmoid(self.beta * z + self.bias)
        return (r, z) if return_expectation else r

    # ------------------------------------------------------------------ #
    @torch.no_grad()
    def score(self, x: np.ndarray) -> np.ndarray:
        """Inference path used by the assembler."""
        return self.forward(x).cpu().numpy()

    @torch.no_grad()
    def wrap_angles(self) -> None:
        """Fold θ back into (−π, π].

        The compact manifold is the whole point — without wrapping, the logged 'weight
        norm' would drift upward and hide the very property being demonstrated.
        """
        self.theta.copy_(torch.remainder(self.theta + torch.pi, 2 * torch.pi) - torch.pi)

    def theta_norm(self, wrapped: bool = True) -> float:
        t = torch.remainder(self.theta + torch.pi, 2 * torch.pi) - torch.pi if wrapped else self.theta
        return float(t.norm())

    def grad_norm(self) -> float:
        return float(torch.sqrt(sum((p.grad**2).sum() for p in self.parameters()
                                    if p.grad is not None)))
