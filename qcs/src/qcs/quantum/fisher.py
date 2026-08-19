"""Fisher information — the instrument that measures plasticity.

`Tr(F)` on a FIXED probe batch is the paper's central quantity: it is the model's
effective learning capacity. In classical networks it collapses as weights inflate
(parameters become functionally dead while still existing); the claim under test is that
in a unitary model it stays bounded away from zero for the whole run.

Two variants, both reported:
  empirical F — curvature with respect to the batch's true labels (update agility)
  model F     — labels sampled from the model's own predictions (intrinsic capacity)

`qfim_diagonal` is a different object: the quantum Fisher information of the state
itself, used by the Phase 6 anchor as the geometrically correct metric for a penalty on
a curved parameter manifold. Euclidean EWC would measure distance in the wrong geometry.
"""

from __future__ import annotations

import numpy as np
import torch


def _bernoulli_logprob_grad(model, x: np.ndarray, y: torch.Tensor) -> torch.Tensor:
    """∂ log p(y|x) / ∂θ for a Bernoulli head, per example, flattened."""
    grads = []
    for i in range(len(y)):
        model.zero_grad(set_to_none=True)
        r = model(x[i:i + 1]).clamp(1e-6, 1 - 1e-6)
        logp = y[i] * torch.log(r) + (1 - y[i]) * torch.log(1 - r)
        logp.backward()
        grads.append(model.theta.grad.detach().flatten().clone())
    return torch.stack(grads)


def fim_trace(model, x: np.ndarray, labels: np.ndarray | None = None,
              max_examples: int = 64, seed: int = 0) -> float:
    """Tr(F) = E[ Σ_k (∂ log p / ∂θ_k)² ] on a fixed probe batch.

    `labels=None` gives the model FIM (labels sampled from the model's own output);
    passing labels gives the empirical FIM used in the paper's main text.
    """
    n = min(len(x), max_examples)
    x = x[:n]
    if labels is None:
        with torch.no_grad():
            p = model(x).clamp(1e-6, 1 - 1e-6)
        g = torch.Generator().manual_seed(seed)
        y = torch.bernoulli(p, generator=g)
    else:
        y = torch.tensor(np.asarray(labels[:n], dtype=np.float32))
    G = _bernoulli_logprob_grad(model, x, y)
    return float((G**2).sum(dim=1).mean())


def fim_spectrum(model, x: np.ndarray, labels: np.ndarray | None = None,
                 max_examples: int = 32, seed: int = 0) -> np.ndarray:
    """Eigenvalues of the empirical FIM (via the Gram matrix — n ≪ |θ|)."""
    n = min(len(x), max_examples)
    if labels is None:
        with torch.no_grad():
            p = model(x[:n]).clamp(1e-6, 1 - 1e-6)
        y = torch.bernoulli(p, generator=torch.Generator().manual_seed(seed))
    else:
        y = torch.tensor(np.asarray(labels[:n], dtype=np.float32))
    G = _bernoulli_logprob_grad(model, x[:n], y)
    gram = (G @ G.T) / n
    return np.sort(np.linalg.eigvalsh(gram.numpy()))[::-1]


def effective_rank(spectrum: np.ndarray, eps: float = 1e-12) -> float:
    """Participation ratio of the spectrum: how many directions actually carry curvature."""
    s = np.clip(spectrum, eps, None)
    p = s / s.sum()
    return float(np.exp(-(p * np.log(p)).sum()))


@torch.no_grad()
def qfim_diagonal(model, x: np.ndarray, eps: float = 1e-3, max_examples: int = 32,
                  n_sample: int | None = 96, seed: int = 0) -> np.ndarray:
    """Diagonal of the quantum Fisher information metric, by finite differences of state
    fidelity:  F_Q,kk ≈ 8(1 − |⟨ψ(θ)|ψ(θ + ε e_k)⟩|) / ε².

    The diagonal is enough for the anchor penalty and is O(|θ|) states instead of O(|θ|²).
    """
    from .encoding import amplitude_encode

    xb = x[:max_examples]
    state0 = model.evolve(amplitude_encode(xb, model.n, device=model.theta.device))
    base = model.theta.detach().clone()
    out = np.zeros(base.numel(), dtype=np.float32)
    flat = base.flatten()
    # Computing all |θ| entries costs |θ| circuit evaluations (1,080 at depth 8) and
    # dominated the task loop. The diagonal is used only to WEIGHT an anchor penalty, so
    # a random subsample with the sample mean imputed elsewhere is accurate enough and
    # ~11× cheaper. Set n_sample=None for the exact diagonal.
    idx = range(flat.numel()) if n_sample is None or n_sample >= flat.numel() else         np.random.default_rng(seed).choice(flat.numel(), n_sample, replace=False)
    for k in idx:
        pert = flat.clone()
        pert[k] += eps
        state1 = model.evolve(amplitude_encode(xb, model.n, device=model.theta.device),
                              theta=pert.reshape(base.shape))
        fid = (state0.conj() * state1).sum(dim=1).abs()
        out[k] = float((8.0 * (1.0 - fid) / eps**2).mean())
    if n_sample is not None and n_sample < flat.numel():
        sampled = out[list(idx)]
        out[out == 0] = float(sampled.mean())
    return out
