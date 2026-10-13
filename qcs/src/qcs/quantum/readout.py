"""S9 · measurement — where the quantum state collapses to one classical number.

`local_z`  ⟨Z_q⟩ on a single qubit. THE DEFAULT. It is justified by the light-cone argument and by standard
           barren-plateau theory for deep global-observable circuits. (An early demo reported a measured
           plateau for the global observable; that was a bug, because it had computed a product of
           marginals. Corrected in Phase 4: at N=10, depth up to 16, the global observable shows no plateau.)
`global_z` ⟨Z⊗Z⊗…⊗Z⟩. Kept for the ablation in scripts/pipeline/run_eval.py.
`basis`    the full 2^N distribution, for the dense-feature readout variant.
"""

from __future__ import annotations

import torch


def marginal_z(state: torch.Tensor, n: int, q: int) -> torch.Tensor:
    """⟨Z_q⟩ for a batch of statevectors."""
    B = state.shape[0]
    p = (state.abs() ** 2).reshape(B, *([2] * n))
    axes = [i + 1 for i in range(n) if i != q]
    pq = p.sum(dim=axes)
    return pq[:, 0] - pq[:, 1]


def parity_z(state: torch.Tensor, n: int) -> torch.Tensor:
    """⟨Z⊗Z⊗…⊗Z⟩ — the global observable, i.e. the barren-plateau path."""
    B = state.shape[0]
    p = (state.abs() ** 2).reshape(B, -1)
    idx = torch.arange(2**n, device=state.device)
    parity = torch.tensor([(-1.0) ** bin(int(i)).count("1") for i in idx],
                          device=state.device, dtype=p.dtype)
    return (p * parity).sum(dim=1)


def readout_expectation(state: torch.Tensor, n: int, q: int, observable: str) -> torch.Tensor:
    if observable == "local_z":
        return marginal_z(state, n, q)
    if observable == "global_z":
        return parity_z(state, n)
    raise KeyError(f"unknown observable '{observable}'")


def sample_shots(state: torch.Tensor, n: int, q: int, shots: int,
                 generator: torch.Generator | None = None) -> torch.Tensor:
    """Shot-noise estimate of ⟨Z_q⟩ — the hardware-realistic path (Phase 9).

    Exact simulation gives the infinite-shot limit; this says how many measurements a
    real device would need before the score ranking is stable.
    """
    B = state.shape[0]
    p = (state.abs() ** 2).reshape(B, *([2] * n))
    axes = [i + 1 for i in range(n) if i != q]
    pq = p.sum(dim=axes)
    pq = pq / pq.sum(dim=1, keepdim=True)
    draws = torch.distributions.Binomial(total_count=shots, probs=pq[:, 0]).sample()
    return 2.0 * draws / shots - 1.0
