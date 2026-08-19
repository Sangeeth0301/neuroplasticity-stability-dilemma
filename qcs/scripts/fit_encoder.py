"""Phase 2 · fit the frozen encoding stack and emit V05-V08, V13.

    python scripts/fit_encoder.py [--epochs 30] [--tag phase2]

Steps
  1. segment the training corpus into semantic units
  2. embed with the frozen encoder (cached by sha256(text))
  3. fit PCA whitening on a HELD-OUT slice of threads (never on training probes)
  4. train the projection with InfoNCE: positives = required facts, negatives = hard
     in-thread distractors mentioning the same entity
  5. freeze everything, measure retention, assemble real 1024-d triples, plot
"""

from __future__ import annotations

import argparse
import json

import numpy as np

from qcs import paths
from qcs.config import Config
from qcs.encode.pipeline import EncodingPipeline
from qcs.encode.assemble import block_norm_shares, participation_ratio
from qcs.encode.projection import Projection
from qcs.encode.relational import build_rho
from qcs.ingest.loader import group_by_thread, load_facts, load_sus
from qcs.logging_ import RunLogger
from qcs.seeding import seed_everything
from qcs.viz import embedding_viz


def build_training_pairs(sus_by_thread, facts, rng, n_neg: int):
    """(probe, required-fact, hard negatives) index triples, training threads only."""
    anchors, positives, negatives = [], [], []
    for tid, sus in sus_by_thread.items():
        by_fact = {s.fact_id: s for s in sus if s.kind in ("fact", "update") and s.fact_id}
        probes = [s for s in sus if s.kind == "probe"]
        for p in probes:
            turn_facts = [f for f in facts.values() if f.thread_id == tid]
            required = [s.su_id for fid, s in by_fact.items()
                        if any(fid == rf for rf in _required_of(p, sus))]
            if not required:
                continue
            pos_su = by_fact[_required_of(p, sus)[0]]
            entity = facts[pos_su.fact_id].entity
            hard = [s for s in sus
                    if s.turn_idx < p.turn_idx and s.su_id != pos_su.su_id
                    and entity.split()[0] in s.text]
            other = [s for s in sus if s.turn_idx < p.turn_idx and s.su_id != pos_su.su_id]
            pool = hard if len(hard) >= n_neg else hard + list(rng.choice(other, n_neg, replace=True))
            picks = list(rng.choice(pool, n_neg, replace=len(pool) < n_neg))
            anchors.append(p)
            positives.append(pos_su)
            negatives.append(picks)
            _ = turn_facts
    return anchors, positives, negatives


def _required_of(probe_su, sus):
    """Recover a probe's required fact ids from the turn it was segmented from."""
    return probe_su._required  # attached below


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", nargs="*", default=[])
    ap.add_argument("--epochs", type=int, default=30)
    ap.add_argument("--tag", default="phase2_encoder")
    args = ap.parse_args()

    cfg = Config.load(*args.config)
    cfg.validate()
    rng = seed_everything(cfg.seed)
    log = RunLogger.create(cfg, args.tag)
    prov = f"{log.run_dir.name} · seed {cfg.seed}"

    print("segmenting…")
    from qcs.ingest.loader import load_turns

    turns = load_turns("train")
    turn_req = {(t.thread_id, t.turn_idx): t.required_facts for t in turns}
    sus = load_sus("train", cfg.data.su_target_tokens)
    for s in sus:
        s._required = turn_req.get((s.thread_id, s.turn_idx), [])
    facts = load_facts()
    print(f"  {len(turns)} turns → {len(sus)} semantic units")

    pipe = EncodingPipeline(cfg)
    print(f"encoder backend: {pipe.encoder.info}")

    # --- whitening fitted on held-out threads only ---------------------------
    by_thread = group_by_thread(sus)
    tids = sorted(by_thread)
    holdout = set(tids[: max(2, len(tids) // 6)])
    fit_texts = [s.text for t in holdout for s in by_thread[t]]
    print(f"fitting whitening on {len(holdout)} held-out threads ({len(fit_texts)} units)…")
    E_fit = pipe.fit_whitener(fit_texts)

    train_sus = [s for t in tids if t not in holdout for s in by_thread[t]]
    E_all = pipe.embed_texts([s.text for s in train_sus])
    W_all = pipe.whitener.transform(E_all)
    idx = {s.su_id: i for i, s in enumerate(train_sus)}

    # --- projection ----------------------------------------------------------
    train_by_thread = {t: by_thread[t] for t in tids if t not in holdout}
    anchors, positives, negatives = build_training_pairs(
        train_by_thread, facts, rng, cfg.train.n_negatives
    )
    print(f"training projection on {len(anchors)} (probe, fact, {cfg.train.n_negatives}-neg) pairs…")
    A = np.stack([W_all[idx[s.su_id]] for s in anchors])
    P = np.stack([W_all[idx[s.su_id]] for s in positives])
    N = np.stack([[W_all[idx[s.su_id]] for s in row] for row in negatives])

    proj = Projection(W_all.shape[1], cfg.encode.proj_dim, seed=cfg.seed)
    hist = proj.fit_infonce(A, P, N, epochs=args.epochs, seed=cfg.seed)
    pipe.projection = proj
    print(f"  epoch 0: loss {hist[0]['loss']:.4f} acc {hist[0]['rank1_acc']:.3f}")
    print(f"  epoch {len(hist)-1}: loss {hist[-1]['loss']:.4f} acc {hist[-1]['rank1_acc']:.3f}")

    # --- retention (V07) on HELD-OUT threads ---------------------------------
    # Measuring retention on the projection's own training pairs would report ~1.0 and
    # mean nothing. The whitening/holdout threads were never seen by InfoNCE, so they
    # are the only honest place to ask whether the 384→256 projection kept the signal.
    hold_sus = [s for t in holdout for s in by_thread[t]]
    E_hold = pipe.embed_texts([s.text for s in hold_sus])
    W_hold = pipe.whitener.transform(E_hold)
    hidx = {s.su_id: i for i, s in enumerate(hold_sus)}
    h_anchors, h_positives, _ = build_training_pairs(
        {t: by_thread[t] for t in holdout}, facts, rng, cfg.train.n_negatives
    )
    print(f"evaluating retention on {len(h_anchors)} held-out probe→fact pairs…")

    ks = [1, 3, 5, 10, 20]
    reps = {"raw": E_hold, "whitened": W_hold, "projected": proj(W_hold)}
    curves: dict[str, list[float]] = {}
    a_ids = [s.su_id for s in h_anchors]
    p_ids = [s.su_id for s in h_positives]
    for name, R in reps.items():
        Rn = R / (np.linalg.norm(R, axis=1, keepdims=True) + 1e-8)
        curve = []
        for k in ks:
            hits = 0
            for aid, pid in zip(a_ids[:400], p_ids[:400]):
                q = Rn[hidx[aid]]
                sims = Rn @ q
                sims[hidx[aid]] = -np.inf
                top = np.argpartition(-sims, k)[:k]
                hits += int(hidx[pid] in set(top.tolist()))
            curve.append(hits / min(400, len(a_ids)))
        curves[name] = curve
        print(f"  recall@{ks[-1]} [{name}]: {curve[-1]:.3f}")

    # --- separation (V08), also on held-out threads ---------------------------
    _, _, h_negatives = build_training_pairs(
        {t: by_thread[t] for t in holdout}, facts, rng, cfg.train.n_negatives
    )
    Pr = reps["projected"]
    Pn = Pr / (np.linalg.norm(Pr, axis=1, keepdims=True) + 1e-8)
    pos_sim = np.array([Pn[hidx[a]] @ Pn[hidx[p]] for a, p in zip(a_ids, p_ids)])
    hard_sim = np.array([Pn[hidx[a.su_id]] @ Pn[hidx[n.su_id]]
                         for a, row in zip(h_anchors, h_negatives) for n in row[:2]])
    rnd = rng.choice(len(Pn), size=len(pos_sim) * 2)
    rand_sim = np.array([Pn[hidx[a]] @ Pn[j] for a, j in zip(a_ids * 2, rnd)])

    # --- assembled features (V13) -------------------------------------------
    users = sorted({s.user_id for s in hold_sus})
    owner_of = {fid: f.owner_user for fid, f in facts.items()}
    X = []
    for a, p in zip(h_anchors[:600], h_positives[:600]):
        rho = build_rho(
            salience=np.zeros(64), turn_gap=a.turn_idx - p.turn_idx,
            seconds_gap=a.ts - p.ts, query_user=a.user_id, cand_user=p.user_id,
            owner_user=owner_of.get(p.fact_id or ""), users=users,
        )
        X.append(pipe.features(Pr[hidx[a.su_id]], Pr[hidx[p.su_id]], rho))
    X = np.stack(X)
    shares = block_norm_shares(X, cfg.encode.feature_dim)
    pr = participation_ratio(X)
    print(f"  block shares: { {k: round(v, 3) for k, v in shares.items()} }")
    print(f"  participation ratio: {pr:.0f} / {cfg.encode.feature_dim}")

    saved = pipe.save()
    figs = [
        embedding_viz.v05_embedding_geometry(
            E_hold, [s.kind for s in hold_sus], [s.user_id for s in hold_sus],
            [s.turn_idx for s in hold_sus], log.figs, prov),
        embedding_viz.v06_whitening_spectrum(pipe.whitener, E_fit, log.figs, prov),
        embedding_viz.v07_projection_retention(curves, ks, log.figs, prov),
        embedding_viz.v08_cosine_separation(pos_sim, hard_sim, rand_sim, log.figs, prov),
        embedding_viz.v13_feature_blocks(X, shares, cfg.encode.block_norm_targets, pr,
                                         log.figs, prov),
    ]

    log.note("encoder_report", {
        "encoder": pipe.encoder.info.__dict__, "artifacts": saved,
        "n_sus": len(sus), "n_pairs": len(anchors),
        "recall_curves": curves, "block_shares": shares,
        "participation_ratio": pr, "projection_history": hist[-1],
        "cohens_d_hard": float((pos_sim.mean() - hard_sim.mean()) /
                               np.sqrt(0.5 * (pos_sim.var() + hard_sim.var()))),
    })
    print("\nartifacts:", json.dumps(saved, indent=2))
    for f in figs:
        print("  ", f)
    print("run dir:", log.run_dir)


if __name__ == "__main__":
    main()
