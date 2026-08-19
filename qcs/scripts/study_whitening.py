"""Phase 2 study · how many whitening components, and how much shrinkage?

Whitening is a trade-off that only exists because of the quantum encoding:
  - too little  → anisotropic embeddings put all amplitude mass on a few basis states
                  (wasted qubits, low participation ratio)
  - too much    → low-variance noise directions get amplified to the same scale as
                  signal directions, and kNN retrieval degrades

This script measures both sides on held-out threads and prints the frontier, so the
config value is chosen by evidence rather than by taste. Emits V06b.
"""

from __future__ import annotations

import argparse

import numpy as np

from qcs.config import Config
from qcs.encode.assemble import participation_ratio
from qcs.encode.cache import EmbeddingCache
from qcs.encode.encoder import FrozenEncoder
from qcs.encode.whiten import Whitener
from qcs.ingest.loader import group_by_thread, load_facts, load_sus, load_turns
from qcs.logging_ import RunLogger
from qcs.seeding import seed_everything
from qcs.viz import style


def recall_at_k(R: np.ndarray, pairs: list[tuple[int, int]], k: int = 20) -> float:
    Rn = R / (np.linalg.norm(R, axis=1, keepdims=True) + 1e-8)
    hits = 0
    for a, p in pairs:
        sims = Rn @ Rn[a]
        sims[a] = -np.inf
        hits += int(p in set(np.argpartition(-sims, k)[:k].tolist()))
    return hits / max(len(pairs), 1)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--tag", default="phase2_whitening_study")
    args = ap.parse_args()

    cfg = Config.load()
    seed_everything(cfg.seed)
    log = RunLogger.create(cfg, args.tag)

    turns = load_turns("train")
    req = {(t.thread_id, t.turn_idx): t.required_facts for t in turns}
    sus = load_sus("train", cfg.data.su_target_tokens)
    facts = load_facts()
    by_thread = group_by_thread(sus)
    tids = sorted(by_thread)
    hold = tids[: max(2, len(tids) // 6)]
    hold_sus = [s for t in hold for s in by_thread[t]]
    idx = {s.su_id: i for i, s in enumerate(hold_sus)}

    pairs: list[tuple[int, int]] = []
    for t in hold:
        by_fact = {s.fact_id: s for s in by_thread[t] if s.kind in ("fact", "update") and s.fact_id}
        for s in by_thread[t]:
            if s.kind != "probe":
                continue
            for fid in req.get((s.thread_id, s.turn_idx), []):
                if fid in by_fact:
                    pairs.append((idx[s.su_id], idx[by_fact[fid].su_id]))
    print(f"{len(hold_sus)} held-out units, {len(pairs)} probe→fact pairs")
    _ = facts

    enc = FrozenEncoder(cfg.encode.model_name, cfg.encode.embed_dim)
    E = EmbeddingCache(enc).embed([s.text for s in hold_sus])

    rows = []
    base = recall_at_k(E, pairs)
    print(f"raw recall@20 = {base:.3f}   PR(raw) = {participation_ratio(_pad(E)):.0f}")

    for k in (32, 64, 128, 192, 256, 384):
        for eps in (1e-6, 1e-3, 1e-2, 1e-1):
            if k > E.shape[1]:
                continue
            w = Whitener(k, eps=eps).fit(E)
            W = w.transform(E)
            rows.append({
                "k": k, "eps": eps,
                "recall": recall_at_k(W, pairs),
                "pr": participation_ratio(_pad(W)),
                "cond": w.condition_number(E)[1],
            })
            print(f"  k={k:3d} eps={eps:<6} recall={rows[-1]['recall']:.3f} "
                  f"PR={rows[-1]['pr']:6.0f}  cond={rows[-1]['cond']:.2f}")

    _plot(rows, base, log.figs, f"{log.run_dir.name} · seed {cfg.seed}")
    log.note("whitening_study", {"raw_recall": base, "rows": rows})
    best = max(rows, key=lambda r: r["recall"])
    print(f"\nbest recall: k={best['k']} eps={best['eps']} → {best['recall']:.3f} "
          f"(raw {base:.3f}), PR={best['pr']:.0f}")
    print("run dir:", log.run_dir)


def _pad(X: np.ndarray, dim: int = 1024) -> np.ndarray:
    """Pad/truncate to a power of two so participation ratio is comparable across ranks."""
    out = np.zeros((min(len(X), 400), dim), dtype=np.float32)
    out[:, : min(X.shape[1], dim)] = X[:400, : min(X.shape[1], dim)]
    return out


def _plot(rows, base, out_dir, prov):
    import matplotlib.pyplot as plt

    style.apply_style()
    fig, ax = plt.subplots(1, 3, figsize=(12, 3.2))
    eps_vals = sorted({r["eps"] for r in rows})
    for i, eps in enumerate(eps_vals):
        sub = [r for r in rows if r["eps"] == eps]
        ks = [r["k"] for r in sub]
        ax[0].plot(ks, [r["recall"] for r in sub], marker="o", ms=3,
                   color=style.CYCLE[i % len(style.CYCLE)], label=f"eps={eps:g}")
        ax[1].plot(ks, [r["pr"] for r in sub], marker="o", ms=3,
                   color=style.CYCLE[i % len(style.CYCLE)], label=f"eps={eps:g}")
    ax[0].axhline(base, ls="--", color=style.C_MUTE, label="raw (no whitening)")
    style.panel(ax[0], "V06b · recall@20 vs whitening rank", "components kept", "recall")
    ax[0].legend()
    style.panel(ax[1], "amplitude spread (participation ratio)", "components kept", "PR / 1024")
    ax[1].legend()
    ax[2].scatter([r["pr"] for r in rows], [r["recall"] for r in rows],
                  c=[r["k"] for r in rows], cmap="viridis", s=22)
    ax[2].axhline(base, ls="--", color=style.C_MUTE)
    style.panel(ax[2], "the actual trade-off", "participation ratio", "recall@20")
    fig.tight_layout()
    style.save(fig, out_dir, "V06b", "whitening_frontier", prov)


if __name__ == "__main__":
    main()
