"""Phase 4 · the quantum core must be provably correct before it is trained.

The heavy tests here are the cross-check against PennyLane's independent simulator and
the autodiff-vs-parameter-shift agreement. Together they establish that our fast batched
simulator computes the same physics as a reference implementation, and that both gradient
paths agree — which is what licenses using the cheap one for training.
"""

from __future__ import annotations

import numpy as np
import pytest
import torch

from qcs.config import Config
from qcs.quantum.ansatz import brickwall_pairs, light_cone_depth, n_params, pairs_for
from qcs.quantum.circuit import QRE
from qcs.quantum.encoding import (amplitude_encode, entanglement_entropy,
                                  participation_ratio, probabilities)
from qcs.quantum.gates import apply_2q, is_unitary, pauli_basis, su4
from qcs.quantum.gradients import agreement, autodiff_expectation_grad
from qcs.quantum.readout import marginal_z, parity_z, sample_shots


def _cfg(depth=4, n=6, observable="local_z", gate_form="exp"):
    cfg = Config.load()
    cfg.quantum.gate_form = gate_form
    cfg.quantum.n_qubits = n
    cfg.quantum.depth = depth
    cfg.quantum.readout_qubit = n - 1
    cfg.quantum.observable = observable
    cfg.encode.feature_dim = 2**n
    return cfg


def _unit(rng, b, dim):
    x = rng.normal(size=(b, dim)).astype(np.float32)
    return x / np.linalg.norm(x, axis=1, keepdims=True)


# --------------------------------------------------------------------- gates
def test_pauli_basis_has_15_traceless_generators():
    B = pauli_basis()
    assert B.shape == (15, 4, 4)
    for G in B:
        assert abs(complex(torch.trace(G))) < 1e-6
        assert torch.allclose(G @ G, torch.eye(4, dtype=G.dtype), atol=1e-6)


@pytest.mark.parametrize("form", ["exp", "product"])
def test_su4_is_unitary_and_identity_at_zero(form):
    assert torch.allclose(su4(torch.zeros(15), form=form),
                          torch.eye(4, dtype=torch.complex64), atol=1e-6)
    rng = np.random.default_rng(0)
    for _ in range(5):
        U = su4(torch.tensor(rng.uniform(-np.pi, np.pi, 15), dtype=torch.float32), form=form)
        assert is_unitary(U)


def test_both_gate_forms_agree_to_first_order():
    """Near θ=0 the sum and product forms coincide; they diverge as angles grow, which is
    why the choice is a config knob rather than an implementation detail."""
    small = torch.full((15,), 1e-3)
    assert torch.allclose(su4(small, form="exp"), su4(small, form="product"), atol=1e-5)


def test_apply_2q_preserves_norm_and_acts_on_the_right_qubits():
    rng = np.random.default_rng(1)
    n = 5
    x = torch.tensor(_unit(rng, 3, 2**n), dtype=torch.complex64)
    U = su4(torch.tensor(rng.uniform(-np.pi, np.pi, 15), dtype=torch.float32))
    out = apply_2q(x, U, 1, 3, n)
    assert torch.allclose(out.norm(dim=1), torch.ones(3), atol=1e-5)
    # identity gate must be a no-op
    assert torch.allclose(apply_2q(x, torch.eye(4, dtype=torch.complex64), 0, 4, n), x, atol=1e-6)


# ------------------------------------------------------------------ encoding
def test_amplitude_encoding_is_lossless():
    rng = np.random.default_rng(2)
    x = _unit(rng, 4, 1024)
    psi = amplitude_encode(x, 10)
    assert psi.shape == (4, 1024)
    assert np.abs(psi.real.numpy() - x).max() < 1e-7
    assert abs(float(probabilities(psi).sum(dim=1).mean()) - 1.0) < 1e-6


def test_amplitude_encoding_rejects_non_states():
    rng = np.random.default_rng(3)
    with pytest.raises(ValueError, match="not unit-norm"):
        amplitude_encode(rng.normal(size=(2, 1024)) * 5, 10)
    with pytest.raises(ValueError, match="exactly 1024"):
        amplitude_encode(_unit(rng, 2, 900), 10)


def test_participation_ratio_bounds():
    n = 4
    uniform = torch.full((1, 2**n), 1 / np.sqrt(2**n), dtype=torch.complex64)
    basis = torch.zeros(1, 2**n, dtype=torch.complex64)
    basis[0, 0] = 1.0
    assert abs(float(participation_ratio(uniform)) - 2**n) < 1e-4
    assert abs(float(participation_ratio(basis)) - 1.0) < 1e-4


# -------------------------------------------------------------------- ansatz
def test_param_count_formula():
    assert len(brickwall_pairs(10)) == 9
    assert n_params("brickwall", 10, 8) == 8 * 9 * 15 == 1080
    cfg = _cfg(depth=3, n=10)
    assert QRE(cfg).n_circuit_params == n_params("brickwall", 10, 3)


def test_light_cone_depth_matches_the_config_floor():
    """The floor Config.validate() enforces must be the real light-cone depth."""
    d = light_cone_depth(10, 9, "brickwall")
    assert d == 5
    assert d <= 10 // 2 + 1


def test_unknown_ansatz_is_rejected():
    with pytest.raises(KeyError):
        pairs_for("spaghetti", 6)


# ------------------------------------------------------------------- circuit
def test_forward_shapes_and_range():
    rng = np.random.default_rng(4)
    m = QRE(_cfg())
    x = _unit(rng, 7, 64)
    r = m(x)
    assert r.shape == (7,)
    assert float(r.min()) >= 0.0 and float(r.max()) <= 1.0


def test_circuit_preserves_norm_at_every_layer():
    rng = np.random.default_rng(5)
    m = QRE(_cfg(depth=6))
    psi = amplitude_encode(_unit(rng, 3, 64), m.n)
    _, states = m.evolve(psi, keep=True)
    for s in states:
        assert torch.allclose(s.norm(dim=1), torch.ones(3), atol=1e-4)


def test_identity_init_starts_near_the_input_state():
    rng = np.random.default_rng(6)
    cfg = _cfg(depth=4)
    cfg.quantum.init_scale = 0.0
    m = QRE(cfg)
    psi = amplitude_encode(_unit(rng, 2, 64), m.n)
    assert torch.allclose(m.evolve(psi), psi, atol=1e-5)


def test_entanglement_grows_with_depth():
    """Flat entropy would mean the pairing is wrong and qubits are disconnected.

    The input must be a PRODUCT state: a random unit vector in 2^n dimensions is already
    near-maximally entangled (Page value), so it cannot show entangling power.
    """
    rng = np.random.default_rng(7)
    cfg = _cfg(depth=6)
    cfg.quantum.init_scale = 1.0
    m = QRE(cfg)
    single = rng.normal(size=(m.n, 2)).astype(np.float32)
    single /= np.linalg.norm(single, axis=1, keepdims=True)
    prod = single[0]
    for q in range(1, m.n):
        prod = np.kron(prod, single[q])
    psi = amplitude_encode(np.stack([prod] * 4), m.n)
    _, states = m.evolve(psi, keep=True)
    e0 = float(entanglement_entropy(states[0], 3, m.n).mean())
    e_end = float(entanglement_entropy(states[-1], 3, m.n).mean())
    assert e_end > e0 + 0.2


def test_angle_wrapping_bounds_the_norm():
    m = QRE(_cfg())
    with torch.no_grad():
        m.theta.add_(17.0)
    before = m.theta_norm(wrapped=False)
    m.wrap_angles()
    assert m.theta_norm(wrapped=False) < before
    assert float(m.theta.abs().max()) <= np.pi + 1e-6


# ------------------------------------------------------------------ readout
def test_local_and_global_observables_are_in_range():
    rng = np.random.default_rng(8)
    m = QRE(_cfg(depth=3))
    psi = m.evolve(amplitude_encode(_unit(rng, 5, 64), m.n))
    for z in (marginal_z(psi, m.n, m.n - 1), parity_z(psi, m.n)):
        assert float(z.min()) >= -1.0 - 1e-5 and float(z.max()) <= 1.0 + 1e-5


def test_shot_sampling_converges_to_the_exact_expectation():
    rng = np.random.default_rng(9)
    m = QRE(_cfg(depth=3))
    psi = m.evolve(amplitude_encode(_unit(rng, 4, 64), m.n))
    exact = marginal_z(psi, m.n, m.n - 1)
    torch.manual_seed(0)
    est = sample_shots(psi, m.n, m.n - 1, shots=20000)
    assert float((est - exact).abs().max()) < 0.05


# ----------------------------------------------------------------- gradients
def test_autodiff_and_parameter_shift_agree():
    """THE gate on Phase 4: if this fails, nothing downstream is trustworthy.

    Uses the PRODUCT gate form, where each parameter has its own involutory generator and
    the two-term shift rule is exact.
    """
    rng = np.random.default_rng(10)
    m = QRE(_cfg(depth=4, gate_form="product"))
    with torch.no_grad():
        m.theta.copy_(torch.tensor(rng.uniform(-np.pi, np.pi, tuple(m.theta.shape)),
                                   dtype=torch.float32))
    res = agreement(m, _unit(rng, 8, 64))
    assert res["max_abs_diff"] < 1e-5, res


def test_shift_rule_is_invalid_for_the_exponential_form():
    """Regression test for a mathematical fact, not a bug.

    exp(−i/2 Σ θ_k G_k) has non-commuting generators, so shifting one angle does not
    isolate its derivative. Measured discrepancy is ~1e-2 — 3+ orders of magnitude above
    the product form's. If this test ever starts passing at 1e-5, someone has silently
    changed the gate parameterisation.
    """
    rng = np.random.default_rng(10)
    m = QRE(_cfg(depth=4, gate_form="exp"))
    with torch.no_grad():
        m.theta.copy_(torch.tensor(rng.uniform(-np.pi, np.pi, tuple(m.theta.shape)),
                                   dtype=torch.float32))
    assert agreement(m, _unit(rng, 8, 64))["max_abs_diff"] > 1e-4


def test_gradients_reach_every_layer_above_the_light_cone():
    rng = np.random.default_rng(11)
    m = QRE(_cfg(depth=6))
    g = autodiff_expectation_grad(m, _unit(rng, 8, 64))
    per_layer = np.abs(g).sum(axis=(1, 2))
    assert (per_layer > 0).all(), f"dead layers: {np.flatnonzero(per_layer == 0)}"


def test_gradients_vanish_below_the_light_cone():
    """A parameter that physically cannot reach the readout must have EXACTLY zero
    gradient — this is the effect that sets the minimum depth."""
    rng = np.random.default_rng(12)
    m = QRE(_cfg(depth=1, n=10))       # one layer: information cannot cross the register
    with torch.no_grad():              # random angles, so smallness is physics not init
        m.theta.copy_(torch.tensor(rng.uniform(-np.pi, np.pi, tuple(m.theta.shape)),
                                   dtype=torch.float32))
    g = np.abs(autodiff_expectation_grad(m, _unit(rng, 4, 1024)))
    # In one brickwall layer, gate 0 acts on qubits (0,1) — it cannot influence q9.
    # Gate 4 acts on (8,9) — it acts on the readout qubit directly.
    far, near = g[0, 0].max(), g[0, 4].max()
    assert far < near / 1e3, f"far={far:.2e} near={near:.2e}"
    assert far < 1e-6


# --------------------------------------------------------------- cross-check
def test_matches_pennylane_reference_simulator():
    """Independent verification: PennyLane must reproduce our statevector and ⟨Z⟩."""
    pennylane = pytest.importorskip("pennylane")
    rng = np.random.default_rng(13)
    n, depth = 4, 2
    cfg = _cfg(depth=depth, n=n)
    m = QRE(cfg)
    with torch.no_grad():
        m.theta.copy_(torch.tensor(rng.uniform(-np.pi, np.pi, tuple(m.theta.shape)),
                                   dtype=torch.float32))
    x = _unit(rng, 1, 2**n)
    ours_state = m.evolve(amplitude_encode(x, n))[0].detach().numpy()
    ours_z = float(marginal_z(m.evolve(amplitude_encode(x, n)), n, n - 1))

    U = m.gates().detach().numpy()
    dev = pennylane.device("default.qubit", wires=n)

    @pennylane.qnode(dev)
    def circuit():
        pennylane.StatePrep(x[0], wires=range(n))
        for layer in range(depth):
            for gi, (q0, q1) in enumerate(m.pairs):
                pennylane.QubitUnitary(U[layer, gi], wires=[q0, q1])
        return pennylane.state()

    ref = np.asarray(circuit())
    assert np.abs(np.abs(np.vdot(ref, ours_state)) - 1.0) < 1e-5, "statevectors differ"

    @pennylane.qnode(dev)
    def z_circuit():
        pennylane.StatePrep(x[0], wires=range(n))
        for layer in range(depth):
            for gi, (q0, q1) in enumerate(m.pairs):
                pennylane.QubitUnitary(U[layer, gi], wires=[q0, q1])
        return pennylane.expval(pennylane.PauliZ(n - 1))

    assert abs(float(z_circuit()) - ours_z) < 1e-5
