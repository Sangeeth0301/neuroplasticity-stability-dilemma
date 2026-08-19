"""General SU(4) gates and the statevector application primitive.

The building block is the FULL two-qubit unitary — 15 real parameters, the whole su(4)
Lie algebra — not a hardware-efficient Ry/Rz/CNOT block:

    V(θ) = exp( −(i/2) · Σ_{(α,β)≠(0,0)} θ_{αβ} (σ_α ⊗ σ_β) ),   σ ∈ {I, X, Y, Z}

Two reasons. It can reach any two-qubit state, so shallow depth suffices; and it is the
exact ansatz whose plasticity behaviour the reference paper establishes, which keeps our
results comparable to theirs.

Everything is torch so that a single autodiff pass differentiates the whole circuit, and
so that all M candidates are scored in one batched contraction rather than a loop.
"""

from __future__ import annotations

import torch

CDTYPE = torch.complex64


def pauli_basis(device=None) -> torch.Tensor:
    """The 15 non-identity tensor products σ_α ⊗ σ_β, shape (15, 4, 4)."""
    I = torch.eye(2, dtype=CDTYPE, device=device)
    X = torch.tensor([[0, 1], [1, 0]], dtype=CDTYPE, device=device)
    Y = torch.tensor([[0, -1j], [1j, 0]], dtype=CDTYPE, device=device)
    Z = torch.tensor([[1, 0], [0, -1]], dtype=CDTYPE, device=device)
    paulis = [I, X, Y, Z]
    return torch.stack([
        torch.kron(paulis[a], paulis[b])
        for a in range(4) for b in range(4) if not (a == 0 and b == 0)
    ])


def su4_exp(theta: torch.Tensor, basis: torch.Tensor | None = None) -> torch.Tensor:
    """Canonical exponential map — the reference paper's form.

        V(θ) = exp(−(i/2) Σ_k θ_k G_k)

    Note: the 15 generators do NOT commute, so the two-term parameter-shift rule is
    *invalid* for this form — shifting one angle does not isolate its derivative.
    Measured discrepancy against autodiff is ~25% (see tests/test_quantum.py). Autodiff
    is exact either way; hardware transfer needs `su4_product` or a compiled
    decomposition.
    """
    basis = pauli_basis(theta.device) if basis is None else basis
    H = torch.einsum("...k,kij->...ij", theta.to(CDTYPE), basis)
    return torch.linalg.matrix_exp(-0.5j * H)


def su4_product(theta: torch.Tensor, basis: torch.Tensor | None = None) -> torch.Tensor:
    """Product of 15 single-Pauli rotations — the shift-rule-valid form.

        V(θ) = Π_k exp(−(i/2) θ_k G_k),   G_k² = I

    Each parameter now has its own involutory generator, so
    ∂⟨O⟩/∂θ_k = ½[⟨O⟩(θ_k+π/2) − ⟨O⟩(θ_k−π/2)] holds exactly. This is also what a real
    device runs, since a general two-qubit unitary is compiled into a gate sequence
    anyway. Spans the same 15-parameter family; the ordering fixes a different chart on
    SU(4), not a smaller one.
    """
    basis = pauli_basis(theta.device) if basis is None else basis
    shape = theta.shape[:-1]
    out = torch.eye(4, dtype=CDTYPE, device=theta.device).expand(*shape, 4, 4).clone()
    for k in range(15):
        ang = theta[..., k].to(CDTYPE).reshape(*shape, 1, 1)
        G = basis[k]
        # exp(-i θ G / 2) = cos(θ/2) I − i sin(θ/2) G, valid because G² = I
        rot = torch.cos(ang / 2) * torch.eye(4, dtype=CDTYPE, device=theta.device) \
            - 1j * torch.sin(ang / 2) * G
        out = rot @ out
    return out


GATE_FORMS = {"exp": su4_exp, "product": su4_product}


def su4(theta: torch.Tensor, basis: torch.Tensor | None = None,
        form: str = "exp") -> torch.Tensor:
    """theta (..., 15) real → (..., 4, 4) unitary. Differentiable in theta."""
    if form not in GATE_FORMS:
        raise KeyError(f"unknown gate form '{form}'; have {sorted(GATE_FORMS)}")
    return GATE_FORMS[form](theta, basis)


def apply_2q(state: torch.Tensor, U: torch.Tensor, q0: int, q1: int, n: int) -> torch.Tensor:
    """Apply a 4×4 gate to qubits (q0, q1) of a batched statevector.

    state: (B, 2**n) complex.  Returns the same shape.

    Implemented as a permute → reshape → matmul → un-permute, i.e. one dense (4×4) @
    (4, B·2**(n-2)) product. That is what makes scoring 256 candidates a single tensor
    op instead of a Python loop over candidates.
    """
    B = state.shape[0]
    st = state.reshape(B, *([2] * n))
    perm = [0, q0 + 1, q1 + 1] + [i + 1 for i in range(n) if i not in (q0, q1)]
    st = st.permute(perm).reshape(B, 4, -1)
    st = torch.matmul(U, st)
    st = st.reshape(B, 2, 2, *([2] * (n - 2)))
    inv = [0] * (n + 1)
    for pos, src in enumerate(perm):
        inv[src] = pos
    return st.permute(inv).reshape(B, 2**n)


def is_unitary(U: torch.Tensor, tol: float = 1e-5) -> bool:
    eye = torch.eye(U.shape[-1], dtype=U.dtype, device=U.device)
    return bool(torch.allclose(U.conj().transpose(-1, -2) @ U, eye, atol=tol))
