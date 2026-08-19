"""Phase 4 · verify the quantum core on REAL data and choose the depth by evidence.

    python scripts/study_trainability.py [--threads 4] [--samples 30]

Emits V14-V20. Nothing in Phase 5+ is allowed to start until the gradient-agreement
check (V19, product form) passes and the depth is chosen from V17.
"""

from __future__ import annotations

import argparse
import json

import numpy as np
import torch

from qcs.config import Config
from qcs.logging_ import RunLogger
from qcs.quantum.ansatz import light_cone_depth, n_params, pairs_for
from qcs.quantum.circuit import QRE
from qcs.quantum.encoding import amplitude_encode, entanglement_entropy, page_entropy
from qcs.quantum.gradients import agreement, gradient_variance
from qcs.seeding import seed_everything
from qcs.train.dataset import ScoringDataset
from qcs.viz import circuit_viz


def flat_with_stale(ds, negatives_per_probe: int, rng):
    """Same selection as ScoringDataset.flat(), keeping the stale flag aligned.

    Kept here rather than in the dataset class because only the figures need the flag;
    the training path must never see it (it is ground truth, not a feature).
    """
    Xs, ys, ss = [], [], []
    for ex in ds.examples:
        pos = np.flatnonzero(ex.y > 0)
        neg = np.flatnonzero(ex.y == 0)
        if len(neg) > negatives_per_probe:
            stale_idx = neg[ex.is_stale[neg] > 0][: negatives_per_probe // 2]
            rest = rng.choice(neg, negatives_per_probe - len(stale_idx), replace=False)
            neg = np.concatenate([stale_idx, rest])
        idx = np.concatenate([pos, neg])
        Xs.append(ex.X[idx]); ys.append(ex.y[idx]); ss.append(ex.is_stale[idx])
    return np.concatenate(Xs), np.concatenate(ys), np.concatenate(ss)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--threads", type=int, default=4)
    ap.add_argument("--probes", type=int, default=40)
    ap.add_argument("--samples", type=int, default=30)
    ap.add_argument("--tag", default="phase4_quantum")
    args = ap.parse_args()

    cfg = Config.load()
    cfg.validate()
    rng = seed_everything(cfg.seed)
    log = RunLogger.create(cfg, args.tag)
    prov = f"{log.run_dir.name} · seed {cfg.seed}"

    print("building scorer dataset from real probes…")
    ds = ScoringDataset(cfg).build(n_threads=args.threads, max_probes_per_thread=args.probes)
    print("  ", json.dumps(ds.stats(), indent=2))
    ds.save()

    X, y, stale = flat_with_stale(ds, negatives_per_probe=16, rng=rng)
    print(f"  flattened: X {X.shape}, positives {int(y.sum())}")

    n, q = cfg.quantum.n_qubits, cfg.quantum
    lc = light_cone_depth(n, q.readout_qubit, q.ansatz)
    ppl = len(pairs_for(q.ansatz, n)) * 15
    print(f"light-cone depth for N={n}, readout q{q.readout_qubit}: {lc} layers "
          f"({ppl} params/layer)")

    # ---- V14-V16: the data as qubits ---------------------------------------
    figs = [circuit_viz.v14_v16_encoding(X, y, n, log.figs, prov)]

    # ---- V17: gradient variance vs depth, on real feature vectors -----------
    depths = [1, 2, 4, 6, 8, 12, 16]
    xb = X[:16]

    def factory(depth: int, observable: str):
        c = Config.load()
        c.quantum.depth = depth
        c.quantum.observable = observable
        return QRE(c)

    print(f"gradient-variance study ({args.samples} random inits per depth)…")
    var_local = gradient_variance(factory, xb, depths, args.samples, rng, observable="local_z")
    var_global = gradient_variance(factory, xb, depths, args.samples, rng, observable="global_z")
    for d in depths:
        print(f"   L={d:2d}  local {var_local[d]:.3e}   global {var_global[d]:.3e}")
    figs.append(circuit_viz.v17_grad_variance(var_local, var_global, lc, ppl, log.figs, prov))

    # ---- V18: entanglement -------------------------------------------------
    model = QRE(cfg)
    with torch.no_grad():   # Haar-ish angles: measures the ansatz, not the init
        model.theta.copy_(torch.tensor(rng.uniform(-np.pi, np.pi, tuple(model.theta.shape)),
                                       dtype=torch.float32))
    _, states = model.evolve(amplitude_encode(X[:8], n), keep=True)
    ent = {layer: np.array([float(entanglement_entropy(s, c, n).mean())
                            for c in range(1, n)])
           for layer, s in enumerate(states) if layer in (0, 1, 2, 4, cfg.quantum.depth)}
    page = [page_entropy(c, n) for c in range(1, n)]
    figs.append(circuit_viz.v18_entanglement(ent, page, n, log.figs, prov))
    print(f"mid-cut entropy: input {ent[0][n // 2 - 1]:.2f} → "
          f"depth {cfg.quantum.depth} {ent[cfg.quantum.depth][n // 2 - 1]:.2f} bits "
          f"(Page {page[n // 2 - 1]:.0f})")

    # ---- V19: gradient agreement, both gate forms ---------------------------
    res = {}
    for form in ("product", "exp"):
        c = Config.load()
        c.quantum.gate_form = form
        m = QRE(c)
        with torch.no_grad():
            m.theta.copy_(torch.tensor(rng.uniform(-np.pi, np.pi, tuple(m.theta.shape)),
                                       dtype=torch.float32))
        res[form] = agreement(m, xb)
        print(f"gradient agreement [{form}]: max|Δ| = {res[form]['max_abs_diff']:.2e}")
    figs.append(circuit_viz.v19_gradient_agreement(res["exp"]["pairs"], res["product"]["pairs"],
                                                   log.figs, prov))
    assert res["product"]["max_abs_diff"] < 1e-5, "PHASE 4 GATE FAILED: shift rule disagrees"

    # ---- V20: readout separation at initialisation --------------------------
    model0 = QRE(cfg)
    with torch.no_grad():
        _, z = model0(X, return_expectation=True)
    z = z.numpy()
    figs.append(circuit_viz.v20_readout_separation(
        z[y > 0], z[(y == 0) & (stale == 0)], z[(y == 0) & (stale > 0)],
        "(UNTRAINED — the baseline training must move)", log.figs, prov))
    d0 = float((z[y > 0].mean() - z[y == 0].mean()) /
               np.sqrt(0.5 * (z[y > 0].var() + z[y == 0].var()) + 1e-12))
    print(f"untrained readout separation: Cohen's d = {d0:.3f}")

    log.note("quantum_report", {
        "light_cone_depth": lc, "params_per_layer": ppl,
        "n_params_at_configured_depth": n_params(q.ansatz, n, q.depth),
        "grad_var_local": {str(k): v for k, v in var_local.items()},
        "grad_var_global": {str(k): v for k, v in var_global.items()},
        "agreement": {k: v["max_abs_diff"] for k, v in res.items()},
        "untrained_cohens_d": d0, "dataset": ds.stats(),
    })
    for f in figs:
        print("  ", f)
    print("run dir:", log.run_dir)


if __name__ == "__main__":
    main()
