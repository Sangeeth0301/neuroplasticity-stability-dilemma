"""S5–S6: assemble the 1024-dimensional feature vector and put it on the unit sphere.

    x_raw = [ p_q | p_c | p_q ⊙ p_c | ρ ]      256 + 256 + 256 + 256 = 1024 = 2^10
    x     = x_raw / ‖x_raw‖₂                    ⇒ a valid quantum state vector

PER-BLOCK NORM BALANCING is the important part. The pre-build demo showed that if the
blocks arrive at different natural scales, the largest one absorbs almost all the
probability mass after normalisation — participation ratio collapsed to 61 of 1024
basis states, meaning the circuit was scoring on recency features and ignoring
semantics entirely. So each block is first scaled to a *designed* share of the total
norm (config: block_norm_targets), and the realised shares are logged as figure V13.
"""

from __future__ import annotations

import numpy as np

BLOCKS = ("p_q", "p_c", "interaction", "rho")


def block_slices(feature_dim: int = 1024) -> dict[str, slice]:
    b = feature_dim // 4
    return {
        "p_q": slice(0, b),
        "p_c": slice(b, 2 * b),
        "interaction": slice(2 * b, 3 * b),
        "rho": slice(3 * b, 4 * b),
    }


def assemble(
    p_q: np.ndarray,
    p_c: np.ndarray,
    rho: np.ndarray,
    block_norm_targets: tuple[float, ...] | list[float] = (0.35, 0.35, 0.20, 0.10),
    feature_dim: int = 1024,
    eps: float = 1e-8,
) -> np.ndarray:
    """One triple → one unit-norm feature vector.

    Accepts either single vectors (d,) or batches (n, d); returns matching rank.
    """
    single = p_q.ndim == 1
    p_q = np.atleast_2d(np.asarray(p_q, dtype=np.float32))
    p_c = np.atleast_2d(np.asarray(p_c, dtype=np.float32))
    rho = np.atleast_2d(np.asarray(rho, dtype=np.float32))

    # direction-only semantics: the projection's magnitude carries no meaning here
    p_q = p_q / (np.linalg.norm(p_q, axis=1, keepdims=True) + eps)
    p_c = p_c / (np.linalg.norm(p_c, axis=1, keepdims=True) + eps)
    inter = p_q * p_c

    targets = np.asarray(block_norm_targets, dtype=np.float32)
    assert abs(float(targets.sum()) - 1.0) < 1e-6, "block_norm_targets must sum to 1"
    # norm shares, not energy shares: target_i is the share of ‖x‖ this block gets
    scale = np.sqrt(targets)

    parts = []
    for arr, s in zip((p_q, p_c, inter, rho), scale):
        n = np.linalg.norm(arr, axis=1, keepdims=True)
        parts.append(arr / (n + eps) * s)

    x = np.concatenate(parts, axis=1)
    assert x.shape[1] == feature_dim, f"expected {feature_dim} dims, got {x.shape[1]}"
    x = x / (np.linalg.norm(x, axis=1, keepdims=True) + eps)

    err = np.abs(np.linalg.norm(x, axis=1) - 1.0).max()
    assert err < 1e-5, f"‖x‖₂ deviates from 1 by {err:.2e} — not a valid quantum state"
    return x[0] if single else x


def block_norm_shares(x: np.ndarray, feature_dim: int = 1024) -> dict[str, float]:
    """Realised share of the squared norm per block — the V13 diagnostic."""
    x = np.atleast_2d(x)
    sl = block_slices(feature_dim)
    total = (x**2).sum(axis=1)
    return {k: float(np.mean((x[:, s] ** 2).sum(axis=1) / total)) for k, s in sl.items()}


def participation_ratio(x: np.ndarray) -> float:
    """1 / Σ p_j² over the measurement distribution: how many basis states are actually used.

    1024 = perfectly spread, 1 = a single basis state. Low values mean wasted qubits.
    """
    p = np.atleast_2d(x) ** 2
    p = p / p.sum(axis=1, keepdims=True)
    return float(np.mean(1.0 / (p**2).sum(axis=1)))
