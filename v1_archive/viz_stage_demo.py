"""
viz_stage_demo.py — runnable demonstration of what every quantum-side stage of the
Dual-Manifold ContextSync pipeline actually LOOKS like.

Pure numpy/scipy statevector simulation (no pennylane needed) so it runs anywhere.
Produces three PNGs in ./figs/ that mirror the per-stage visual artefacts the real
pipeline is required to emit:

    fig_s5_s7_encoding.png   S5 feature assembly -> S6 L2 norm -> S7 amplitude encoding
    fig_s8_s9_circuit.png    S8 brickwall SU(4) evolution -> S9 <Z> readout
    fig_p5_trainability.png  Phase 5 barren-plateau / gradient-variance study

Run:  python viz_stage_demo.py
"""

import numpy as np
from scipy.linalg import expm
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from pathlib import Path

RNG = np.random.default_rng(7)
N = 10                      # qubits
D = 2 ** N                  # Hilbert dimension = 1024

# palette that reads on both light and dark grounds
C_STAB = "#2E93B5"          # classical / stability
C_PLAS = "#C7548E"          # quantum / plasticity
C_MUTE = "#8A949B"
C_WARN = "#D9A441"
plt.rcParams.update({
    "figure.facecolor": "none", "axes.facecolor": "none", "savefig.facecolor": "none",
    "text.color": C_MUTE, "axes.labelcolor": C_MUTE, "axes.edgecolor": C_MUTE,
    "xtick.color": C_MUTE, "ytick.color": C_MUTE, "grid.color": C_MUTE,
    "font.size": 8, "axes.titlesize": 9, "axes.titleweight": "bold",
    "axes.spines.top": False, "axes.spines.right": False,
})

# ----------------------------------------------------------------------------
# Pauli basis + general SU(4) gate  V(theta) = exp(-i/2 * sum theta_ab s_a x s_b)
# ----------------------------------------------------------------------------
I2 = np.eye(2, dtype=complex)
X = np.array([[0, 1], [1, 0]], dtype=complex)
Y = np.array([[0, -1j], [1j, 0]], dtype=complex)
Z = np.array([[1, 0], [0, -1]], dtype=complex)
PAULIS = [I2, X, Y, Z]
BASIS = [np.kron(PAULIS[a], PAULIS[b])
         for a in range(4) for b in range(4) if not (a == 0 and b == 0)]  # 15 generators


def su4(theta):
    """theta: (15,) real -> 4x4 unitary."""
    H = sum(t * G for t, G in zip(theta, BASIS))
    return expm(-0.5j * H)


def apply_2q(state, U, q0, q1):
    """Apply 4x4 gate U to qubits q0,q1 of a state of shape (2,)*N."""
    st = np.moveaxis(state, (q0, q1), (0, 1))
    shp = st.shape
    st = st.reshape(4, -1)
    st = U @ st
    st = st.reshape(shp)
    return np.moveaxis(st, (0, 1), (q0, q1))


def brickwall_pairs(n):
    even = [(i, i + 1) for i in range(0, n - 1, 2)]
    odd = [(i, i + 1) for i in range(1, n - 1, 2)]
    return even + odd


PAIRS = brickwall_pairs(N)
GATES_PER_LAYER = len(PAIRS)                 # 9 for N=10
PARAMS_PER_LAYER = GATES_PER_LAYER * 15      # 135


def run_circuit(psi_flat, thetas):
    """thetas: (L, GATES_PER_LAYER, 15). Returns flat statevector."""
    st = psi_flat.reshape((2,) * N)
    for layer in thetas:
        for (q0, q1), th in zip(PAIRS, layer):
            st = apply_2q(st, su4(th), q0, q1)
    return st.reshape(-1)


def expect_z(psi_flat, q):
    p = np.abs(psi_flat.reshape((2,) * N)) ** 2
    axes = tuple(i for i in range(N) if i != q)
    pq = p.sum(axis=axes)
    return float(pq[0] - pq[1])


def entanglement_entropy(psi_flat, cut):
    """von Neumann entropy across bipartition [0..cut-1] | [cut..N-1]."""
    m = psi_flat.reshape(2 ** cut, 2 ** (N - cut))
    s = np.linalg.svd(m, compute_uv=False)
    p = np.clip(s ** 2, 1e-15, None)
    p = p / p.sum()
    return float(-(p * np.log2(p)).sum())


# ----------------------------------------------------------------------------
# S5: build a realistic 1024-d feature vector with the four structural blocks
# ----------------------------------------------------------------------------
def make_feature_vector(relevant=True):
    p_q = RNG.normal(0, 1, 256)                         # query projection
    if relevant:                                        # candidate shares direction
        p_c = 0.75 * p_q + 0.65 * RNG.normal(0, 1, 256)
    else:
        p_c = RNG.normal(0, 1, 256)
    p_q /= np.linalg.norm(p_q)
    p_c /= np.linalg.norm(p_c)
    inter = p_q * p_c * 16.0                            # interaction block
    rho = np.zeros(256)
    rho[:64] = RNG.normal(0, .35, 64)                   # user salience
    rho[64:80] = np.tanh(RNG.normal(.4 if relevant else -.2, .3, 16))   # recency
    rho[80:112] = RNG.choice([0., .8], 32) * (1 if relevant else .3)    # graph feats
    rho[112:128] = RNG.choice([0., 1.], 16) * .5        # speaker match
    rho[128:136] = RNG.random(8) * .4                   # prior selection
    x_raw = np.concatenate([p_q, p_c, inter, rho])
    return x_raw, x_raw / np.linalg.norm(x_raw)


x_raw, x = make_feature_vector(relevant=True)
assert x.shape == (1024,) and abs(np.linalg.norm(x) - 1) < 1e-9

Path("figs").mkdir(exist_ok=True)

# ============================ FIGURE 1 =======================================
fig, ax = plt.subplots(2, 2, figsize=(11, 6.2))

a = ax[0, 0]
blocks = [(0, 256, "p_q  query", C_STAB), (256, 512, "p_c  candidate", "#5FB3A1"),
          (512, 768, "p_q ⊙ p_c  interaction", C_PLAS), (768, 1024, "ρ  relational", C_WARN)]
for s, e, lbl, c in blocks:
    a.plot(np.arange(s, e), x_raw[s:e], lw=.6, color=c, label=lbl)
    a.axvline(e, color=C_MUTE, lw=.5, ls=":")
a.set_title("S5 · assembled feature vector x_raw ∈ ℝ¹⁰²⁴ (4 blocks)")
a.set_xlabel("feature index"); a.set_ylabel("value")
a.legend(fontsize=6, frameon=False, ncol=2)

a = ax[0, 1]
a.plot(x, lw=.5, color=C_PLAS)
a.set_title(f"S6/S7 · amplitudes after L2 norm  ‖x‖₂={np.linalg.norm(x):.6f}")
a.set_xlabel("basis state index j  (|0000000000⟩ … |1111111111⟩)")
a.set_ylabel("amplitude xⱼ")
a.text(.02, .93, "this vector IS |ψ₀⟩ — 1024 reals → 1024 amplitudes, lossless",
       transform=a.transAxes, fontsize=6.5, color=C_MUTE)

a = ax[1, 0]
im = a.imshow(x.reshape(32, 32), cmap="PuOr", aspect="auto",
              vmin=-np.abs(x).max(), vmax=np.abs(x).max())
a.set_title("S7 · register map: amplitude as a 32×32 grid")
a.set_xlabel("low qubits q₅…q₉  (column)"); a.set_ylabel("high qubits q₀…q₄  (row)")
plt.colorbar(im, ax=a, fraction=.046)

a = ax[1, 1]
probs = x ** 2
top = np.argsort(probs)[::-1][:40]
a.bar(range(40), probs[top], color=C_PLAS, width=.8)
a.set_title("S7 · measurement probabilities |xⱼ|² — top 40 basis states")
a.set_xlabel("rank"); a.set_ylabel("P(j)")
a.text(.35, .8, f"total prob = {probs.sum():.6f}\nmax P(j) = {probs.max():.4f}\n"
                f"participation ratio = {1/np.sum(probs**2):.0f} / 1024",
       transform=a.transAxes, fontsize=6.5, color=C_MUTE)

fig.suptitle("STAGES S5 → S7   ·   how one (query, candidate, user) triple becomes a 10-qubit state",
             fontsize=10, color=C_MUTE, y=1.0)
fig.tight_layout()
fig.savefig("figs/fig_s5_s7_encoding.png", dpi=130, bbox_inches="tight", transparent=True)
plt.close(fig)
print("wrote figs/fig_s5_s7_encoding.png")

# ============================ FIGURE 2 =======================================
L = 8
thetas = RNG.normal(0, 0.6, (L, GATES_PER_LAYER, 15))

psi = x.astype(complex)
states = [psi.copy()]
for layer in thetas:
    st = states[-1].reshape((2,) * N)
    for (q0, q1), th in zip(PAIRS, layer):
        st = apply_2q(st, su4(th), q0, q1)
    states.append(st.reshape(-1))

fig, ax = plt.subplots(2, 2, figsize=(11, 6.2))

a = ax[0, 0]
a.plot(np.abs(states[0]) ** 2, lw=.4, color=C_STAB, label="|ψ₀⟩ input")
a.plot(np.abs(states[-1]) ** 2, lw=.4, color=C_PLAS, label=f"|ψ_L⟩ after L={L}")
a.set_yscale("log"); a.set_ylim(1e-9, 1)
a.set_title("S8 · probability distribution before / after the circuit")
a.set_xlabel("basis state j"); a.set_ylabel("P(j)  (log)")
a.legend(fontsize=6.5, frameon=False)
a.text(.02, .06, f"norm preserved: ‖ψ₀‖={np.linalg.norm(states[0]):.6f}  "
                 f"‖ψ_L‖={np.linalg.norm(states[-1]):.6f}",
       transform=a.transAxes, fontsize=6.5, color=C_MUTE)

a = ax[0, 1]
w = .38
z0 = [expect_z(states[0], q) for q in range(N)]
zL = [expect_z(states[-1], q) for q in range(N)]
a.bar(np.arange(N) - w/2, z0, w, color=C_STAB, label="input")
a.bar(np.arange(N) + w/2, zL, w, color=C_PLAS, label=f"after L={L}")
a.axhline(0, color=C_MUTE, lw=.5)
a.set_title("S9 · per-qubit ⟨Z⟩ — readout qubit is q₉")
a.set_xlabel("qubit"); a.set_ylabel("⟨Z⟩"); a.set_xticks(range(N))
a.legend(fontsize=6.5, frameon=False)

a = ax[1, 0]
for depth in [0, 1, 2, 4, 8]:
    ent = [entanglement_entropy(states[depth], c) for c in range(1, N)]
    a.plot(range(1, N), ent, marker="o", ms=3, lw=1,
           color=plt.cm.plasma(depth / 9), label=f"L={depth}")
a.plot(range(1, N), [min(c, N - c) for c in range(1, N)], ls="--", lw=.8,
       color=C_MUTE, label="max (Page)")
a.set_title("S8 · entanglement entropy vs bipartition cut, by depth")
a.set_xlabel("cut position"); a.set_ylabel("S(ρ_A)  [bits]")
a.legend(fontsize=6, frameon=False, ncol=2)

a = ax[1, 1]
pos, neg = [], []
for _ in range(120):
    _, xp = make_feature_vector(relevant=True)
    _, xn = make_feature_vector(relevant=False)
    pos.append(expect_z(run_circuit(xp.astype(complex), thetas), N - 1))
    neg.append(expect_z(run_circuit(xn.astype(complex), thetas), N - 1))
bins = np.linspace(-1, 1, 30)
a.hist(neg, bins=bins, color=C_STAB, alpha=.75, label="irrelevant candidates")
a.hist(pos, bins=bins, color=C_PLAS, alpha=.75, label="relevant candidates")
a.set_title("S9 · ⟨Z₉⟩ readout separation (UNTRAINED θ — this is the baseline)")
a.set_xlabel("⟨Z₉⟩ → r = σ(β⟨Z⟩+b)"); a.set_ylabel("count")
a.legend(fontsize=6.5, frameon=False)
sep = (np.mean(pos) - np.mean(neg)) / np.sqrt(.5 * (np.var(pos) + np.var(neg)) + 1e-12)
a.text(.02, .78, f"Cohen's d = {sep:.2f}\n(training must move this)",
       transform=a.transAxes, fontsize=6.5, color=C_MUTE)

fig.suptitle("STAGES S8 → S9   ·   brickwall SU(4) evolution and the collapse to one scalar",
             fontsize=10, color=C_MUTE, y=1.0)
fig.tight_layout()
fig.savefig("figs/fig_s8_s9_circuit.png", dpi=130, bbox_inches="tight", transparent=True)
plt.close(fig)
print("wrote figs/fig_s8_s9_circuit.png")

# ============================ FIGURE 3 =======================================
# Phase 5 trainability study: Var[dE/dtheta_k] vs depth over random initialisations
depths = [1, 2, 4, 6, 9, 12, 16]
n_samp = 40
shift = np.pi / 2
var_local, var_global = [], []

_, x0 = make_feature_vector(relevant=True)
psi0 = x0.astype(complex)


def energy(psi, thetas, observable="local"):
    out = run_circuit(psi, thetas)
    if observable == "local":
        return expect_z(out, N - 1)
    # CORRECTION (2026-08-20): this originally computed the product of MARGINALS,
    # prod_q <Z_q>, which decays exponentially in N by construction and is NOT the global
    # observable. The true parity expectation is
    #     <Z (x) Z (x) ... (x) Z> = sum_j (-1)^popcount(j) |psi_j|^2
    # With the correct observable, the 10-qubit circuit shows NO barren plateau at depths
    # <= 16 (see qcs/ figure V17). The local readout is still the right default, but for
    # the light-cone reason, not because the global one was measured to collapse.
    p = np.abs(out) ** 2
    parity = np.array([(-1.0) ** bin(j).count("1") for j in range(len(p))])
    return float((p * parity).sum())


for Ld in depths:
    gl, gg = [], []
    for _ in range(n_samp):
        th = RNG.uniform(-np.pi, np.pi, (Ld, GATES_PER_LAYER, 15))
        thp, thm = th.copy(), th.copy()
        thp[0, 0, 0] += shift
        thm[0, 0, 0] -= shift
        gl.append(.5 * (energy(psi0, thp, "local") - energy(psi0, thm, "local")))
        gg.append(.5 * (energy(psi0, thp, "global") - energy(psi0, thm, "global")))
    var_local.append(np.var(gl))
    var_global.append(np.var(gg))
    print(f"  depth {Ld:2d}: Var[local]={np.var(gl):.3e}  Var[global]={np.var(gg):.3e}")

fig, ax = plt.subplots(1, 2, figsize=(11, 3.4))

a = ax[0]
a.semilogy(depths, var_local, marker="o", color=C_PLAS, label="local observable ⟨Z₉⟩")
a.semilogy(depths, var_global, marker="s", color=C_STAB, label="global observable ⟨Z⊗…⊗Z⟩")
a.set_title("Phase 5 · gradient variance vs circuit depth (barren-plateau guard)")
a.set_xlabel("depth L"); a.set_ylabel("Var[∂⟨O⟩/∂θ₀]")
a.legend(fontsize=6.5, frameon=False)
a.text(.03, .08, "flat/slow decay ⇒ trainable.  exponential decay ⇒ cap depth here.",
       transform=a.transAxes, fontsize=6.5, color=C_MUTE)

a = ax[1]
a.plot(depths, [d * PARAMS_PER_LAYER for d in depths], marker="o", color=C_WARN)
a.set_title("Phase 5 · parameter count vs depth  (135 params / layer, N=10)")
a.set_xlabel("depth L"); a.set_ylabel("|θ|")
for d in depths:
    a.annotate(f"{d*PARAMS_PER_LAYER}", (d, d * PARAMS_PER_LAYER), fontsize=6,
               color=C_MUTE, xytext=(0, 5), textcoords="offset points", ha="center")

fig.tight_layout()
fig.savefig("figs/fig_p5_trainability.png", dpi=130, bbox_inches="tight", transparent=True)
plt.close(fig)
print("wrote figs/fig_p5_trainability.png")
print("\nAll figures written to ./figs/")
