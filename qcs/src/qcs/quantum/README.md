# `quantum/` — the plasticity side (the 10-qubit circuit)

A 10-qubit variational circuit, simulated exactly on a normal computer. The state is a vector of
1,024 complex numbers; each gate is a matrix multiplied into it. Everything is written in PyTorch,
so one automatic-differentiation pass differentiates the whole circuit and all candidates of a probe
are scored in one batched operation.

```
unit vector x (1024) ─► amplitude_encode ─► 8 layers × 9 two-qubit gates ─► read ⟨Z⟩ of each qubit ─► score r
```

| File | What it does | Logic worth knowing |
|---|---|---|
| `gates.py` | The two-qubit gate and how it is applied. | `pauli_basis()` returns the 15 non-identity products of Pauli matrices. `su4_exp` builds `V(θ) = exp(−(i/2) Σ θₖ Gₖ)` with a matrix exponential, a general 4×4 unitary with 15 angles. `su4_product` is the alternative, a product of 15 single rotations, which makes the hardware parameter-shift rule exact. `apply_2q(state, U, q0, q1, n)` reshapes the state to expose the two qubits, multiplies by the 4×4 gate and reshapes back: one dense product for the whole batch. |
| `ansatz.py` | Which qubit pairs each layer touches. | `brickwall_pairs(n)` = the even pairs (0,1), (2,3) … then the odd pairs (1,2), (3,4) …: **9 gates per layer** for 10 qubits. `n_params = depth × gates × 15` (= 1,080). `light_cone_depth` computes how many layers a gate on qubits (0,1) needs before it can influence the readout qubit; below that depth some angles would have exactly zero gradient. |
| `encoding.py` | `amplitude_encode`: 1,024 numbers → a 10-qubit state. | Lossless because `1024 == 2**10`. It refuses input that is not unit length (it would not be a valid state). Also contains diagnostics: `participation_ratio` (how many basis states carry the amplitude) and `entanglement_entropy`. |
| `readout.py` | Turns the state into numbers. | `marginal_z(state, n, q)` = P(qubit q reads 0) − P(reads 1), computed exactly from `|amplitude|²`. A **local** readout is the default, chosen from the light-cone argument and standard barren-plateau theory for deep global observables. (An early demo reported a measured plateau for the global observable; that was a bug, it had computed a product of marginals, and was corrected in Phase 4: at 10 qubits the global observable shows no plateau.) `global_z` is kept for the ablation. `sample_shots` simulates finite measurements. |
| `circuit.py` | `QRE`: the original model (Quantum Relevance Engine). | `theta` has shape `(depth, 9, 15)` and starts near zero, so the circuit starts near the identity. `forward`: encode → run all layers → `⟨Z⟩` of the readout qubit (qubit 9) → `sigmoid(β·z + b)`. `wrap_angles()` folds angles into (−π, π] after each task, which is the "compact manifold". |
| `hybrid.py` | The two variants built in this project. | **`MultiReadoutQRE` (the final model):** the same circuit, but the score is `sigmoid(β · Σ_q w_q ⟨Z_q⟩ + b)` over **all 10 qubits** with 10 learned weights `w_q`. The weights start as a one-hot on qubit 9, so step 0 is identical to `QRE`. **`HybridQRE` (a failed idea, kept as a record):** adds a fixed similarity prior to the logit; it ranked worse than plain cosine similarity. |
| `fisher.py` | The plasticity instruments. | `fim_trace`: the Fisher information trace on a fixed probe batch (the "learning capacity" the base paper tracks), from per-example gradients of a Bernoulli log-likelihood. `qfim_diagonal`: the diagonal of the **quantum** Fisher information by finite differences of state overlap, used to weight the anchor penalty; only 96 randomly chosen angles are measured and the mean is used for the rest (about 11× cheaper). Both look only at `model.theta`. |
| `gradients.py` | Two independent gradient routes and the test that they agree. | `autodiff_grad` (used for training) against `parameter_shift_grad` (the only route that works on hardware). `gradient_variance` is the barren-plateau instrument. |
| `noise.py` | Depolarising noise by trajectory sampling. | After each gate, with probability p, a random Pauli is applied to one of its qubits; results are averaged over trajectories. It uses `model.readout_qubit`, so it applies to the original single-readout `QRE` only. |

## Parameters of the final model

1,080 circuit angles + 10 readout weights + β + b = 1,092 trainable numbers.

## Tests

`tests/test_quantum.py` (23 tests): unitarity and identity at zero angles; norm preserved at every layer; amplitude
encoding is lossless and rejects non-states; parameter count formula; light-cone depth; entanglement grows with depth;
angle wrapping bounds the parameter norm; autodiff and parameter-shift agree (product form) and **disagree** for the
exponential form; gradients vanish below the light cone; the simulator matches PennyLane at fidelity 1.000.
