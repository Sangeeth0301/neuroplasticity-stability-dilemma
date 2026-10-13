"""Phase 3 · build the memory manifold ℳ and emit V09-V12.

    python scripts/pipeline/build_memory.py [--threads 8]

Builds: append-only SQLite store · FAISS prefilter · consolidation summaries ·
entity/supersession DAG · per-user salience. Then measures the one number that caps the
entire system: does the prefilter actually show the scorer the fact it needs?
"""

from __future__ import annotations

import argparse
import json

import numpy as np

from qcs import paths
from qcs.config import Config
from qcs.encode.pipeline import EncodingPipeline
from qcs.ingest.loader import group_by_thread, load_facts, load_sus, load_turns
from qcs.logging_ import RunLogger
from qcs.memory.ann_index import ANNIndex
from qcs.memory.consolidate import consolidate_thread, verify_sources_intact
from qcs.memory.graph import EntityGraph
from qcs.memory.salience import SalienceTracker
from qcs.memory.store import MemoryStore
from qcs.seeding import seed_everything
from qcs.viz import memory_viz


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--threads", type=int, default=8)
    ap.add_argument("--tag", default="phase3_memory")
    args = ap.parse_args()

    cfg = Config.load()
    cfg.validate()
    seed_everything(cfg.seed)
    log = RunLogger.create(cfg, args.tag)
    prov = f"{log.run_dir.name} · seed {cfg.seed}"

    db_path = paths.DATA_PROCESSED / "qcs.db"
    if db_path.exists():
        db_path.unlink()          # rebuild from scratch; ℳ is append-only WITHIN a run
    store = MemoryStore(db_path)

    turns = load_turns("train")
    req = {(t.thread_id, t.turn_idx): t.required_facts for t in turns}
    sus_all = load_sus("train", cfg.data.su_target_tokens)
    facts = load_facts()
    by_thread = group_by_thread(sus_all)
    tids = sorted(by_thread)[: args.threads]
    print(f"building ℳ for {len(tids)} threads")

    pipe = EncodingPipeline.load(cfg)
    store.write_facts(facts)

    growth = {"semantic_units": [], "summaries": [], "edges": []}
    growth_turns: list[int] = []
    index = ANNIndex(cfg.encode.proj_dim)
    proj_by_id: dict[str, np.ndarray] = {}
    graphs: dict[str, EntityGraph] = {}
    salience = SalienceTracker(cfg.memory.salience_alpha)

    for tid in tids:
        sus = by_thread[tid]
        P = pipe.project([s.text for s in sus])
        for s, v in zip(sus, P):
            proj_by_id[s.su_id] = v
        store.write_sus(sus)
        index.add([s.su_id for s in sus], P)

        g = EntityGraph().build(sus, {k: v for k, v in facts.items() if v.thread_id == tid})
        store.write_edges(g.edges())
        graphs[tid] = g

        before = {s.su_id: s.text for s in store.sus(thread_id=tid)}
        consolidate_thread(store, tid, cfg.memory.consolidate_every)
        intact = verify_sources_intact(store, tid, before)
        assert intact, f"consolidation modified source units in {tid}"

        salience.replay_thread(sus, proj_by_id)

        c = store.counts()
        growth_turns.append(len(growth_turns) * cfg.data.turns_per_thread)
        for k in growth:
            growth[k].append(c[k])

    salience.persist(store)
    index.build()
    counts = store.counts()
    print("store:", json.dumps(counts))

    # ---- V10: the ceiling on everything -------------------------------------
    Ms = [8, 16, 32, 64, 128, 256, 512]
    hits = {m: 0 for m in Ms}
    hits_band: dict[int, dict[int, int]] = {}
    totals_band: dict[int, int] = {}
    total = 0
    su_index = {sid: i for i, sid in enumerate(index.ids)}
    su_rows = [store.su(sid) for sid in index.ids]
    turn_of = np.array([s.turn_idx for s in su_rows])
    thread_of = np.array([s.thread_id for s in su_rows])
    fact_of = np.array([s.fact_id or "" for s in su_rows])

    for tid in tids:
        for s in by_thread[tid]:
            if s.kind != "probe":
                continue
            required = req.get((s.thread_id, s.turn_idx), [])
            if not required:
                continue
            gold_rows = np.flatnonzero((thread_of == tid) & np.isin(fact_of, required))
            if len(gold_rows) == 0:
                continue
            band = min(cfg.data.probe_distances,
                       key=lambda b: abs(b - (s.turn_idx - min(turn_of[gold_rows]))))
            mask = (thread_of == tid) & (turn_of < s.turn_idx)   # causality
            total += 1
            totals_band[band] = totals_band.get(band, 0) + 1
            for m in Ms:
                ids, _ = index.search(proj_by_id[s.su_id], m, mask=mask)
                got = any(su_index[i] in set(gold_rows.tolist()) for i in ids)
                hits[m] += int(got)
                hits_band.setdefault(band, {}).setdefault(m, 0)
                hits_band[band][m] += int(got)

    recall = [hits[m] / max(total, 1) for m in Ms]
    recall_band = {b: [hits_band[b][m] / totals_band[b] for m in Ms] for b in sorted(hits_band)}
    print(f"ANN recall over {total} probes:")
    for m, r in zip(Ms, recall):
        print(f"   M={m:4d}  recall={r:.3f}")

    # ---- salience separation -------------------------------------------------
    users = sorted({s.user_id for s in by_thread[tids[0]]})
    S = np.stack([salience.get(tids[0], u) for u in users])
    Sn = S / (np.linalg.norm(S, axis=1, keepdims=True) + 1e-8)
    sep = salience.separation()
    print(f"salience separation (mean 1−cos between users): {sep:.3f}")

    gstats = {"n_entities": sum(g.stats()["n_entities"] for g in graphs.values()),
              "n_supersedes": sum(g.stats()["n_supersedes"] for g in graphs.values())}
    edge_kinds: dict[str, int] = {}
    for _, _, kind, _ in store.edges():
        edge_kinds[kind] = edge_kinds.get(kind, 0) + 1
    stale = sum(len(g.stale_su_ids()) for g in graphs.values())

    # ---- V10b: headroom — what cosine CANNOT do -----------------------------
    from qcs.eval.headroom import cosine_headroom

    head = cosine_headroom(
        [s for tid in tids for s in by_thread[tid]], proj_by_id, req, facts)
    print(f"cosine rank-1 {head['rank1']:.3f}  recall@5 {head['recall5']:.3f}  "
          f"headroom {head['headroom']:.3f}")
    print(f"  by regime: { {k: round(v, 3) for k, v in head['rank1_by_regime'].items()} }")
    print(f"  supersession probes: {head['supersession_cases']}, "
          f"stale outranks current: {head['stale_outranks_current']:.3f}")

    figs = [
        memory_viz.v09_memory_growth(growth, growth_turns, log.figs, prov),
        memory_viz.v10b_headroom(head, log.figs, prov),
        memory_viz.v10_ann_recall(Ms, recall, recall_band, 0.95, log.figs, prov),
        memory_viz.v11_consolidation(
            [s for tid in tids for s in store.summaries(tid)],
            True, log.figs, prov),
        memory_viz.v12_entity_graph(gstats, edge_kinds, stale, sep, Sn @ Sn.T, users,
                                    log.figs, prov),
    ]

    at_M = recall[Ms.index(cfg.memory.candidate_M)]
    log.note("memory_report", {
        "counts": counts, "recall": dict(zip(map(str, Ms), recall)),
        "recall_at_configured_M": at_M, "graph": gstats, "edges": edge_kinds,
        "stale_units": stale, "salience_separation": sep, "n_probes": total, "headroom": head,
    })
    for f in figs:
        print("  ", f)
    print(f"\nEXIT CRITERION recall@M={cfg.memory.candidate_M}: {at_M:.3f} "
          f"({'PASS' if at_M >= 0.95 else 'FAIL'})")
    print("run dir:", log.run_dir)
    store.close()


if __name__ == "__main__":
    main()
