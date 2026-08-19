"""Turning the corpus into scorer training data.

For every probe: take the top-M causal candidates from the prefilter, assemble the
1024-dimensional feature vector for each, and label the ones that are the probe's
required facts. This is the object both arms consume — identical inputs, identical
labels, so an A/B differs in exactly one component.

Causality is enforced twice (candidate mask and an assertion) because a single leaked
future candidate would silently inflate every result in the project.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import numpy as np

from .. import paths
from ..config import Config
from ..encode.pipeline import EncodingPipeline
from ..encode.relational import build_rho
from ..ingest.loader import group_by_thread, load_facts, load_sus, load_turns
from ..memory.ann_index import ANNIndex
from ..memory.graph import EntityGraph
from ..memory.salience import SalienceTracker


@dataclass
class ProbeExample:
    thread_id: str
    probe_turn: int
    su_ids: list[str]
    X: np.ndarray            # (M, 1024) unit-norm features
    y: np.ndarray            # (M,) 1 for required facts
    is_stale: np.ndarray     # (M,) 1 if the candidate states a superseded fact
    tokens: np.ndarray       # (M,) token cost, for the Phase 7 knapsack
    kinds: list[str]
    task_id: int = -1


class ScoringDataset:
    """Builds and caches probe examples across threads."""

    def __init__(self, cfg: Config, split: str = "train"):
        self.cfg = cfg
        self.split = split
        self.examples: list[ProbeExample] = []

    # ------------------------------------------------------------------ #
    def build(self, n_threads: int | None = None, max_probes_per_thread: int | None = None,
              max_candidates: int | None = None, verbose: bool = True) -> "ScoringDataset":
        cfg = self.cfg
        # Training uses a smaller candidate set than the assembler: at M=256 the feature
        # tensor for 720 probes is 755 MB, and prefilter recall is already 1.000 at M=8,
        # so the extra candidates are pure storage cost during training.
        M = max_candidates or cfg.memory.candidate_M
        pipe = EncodingPipeline.load(cfg)
        turns = load_turns(self.split)
        req = {(t.thread_id, t.turn_idx): t.required_facts for t in turns}
        sus = load_sus(self.split, cfg.data.su_target_tokens)
        facts = load_facts()
        by_thread = group_by_thread(sus)
        tids = sorted(by_thread)[: n_threads or len(by_thread)]

        for ti, tid in enumerate(tids):
            S = by_thread[tid]
            P = pipe.project([s.text for s in S])
            pos = {s.su_id: i for i, s in enumerate(S)}
            thread_facts = {k: v for k, v in facts.items() if v.thread_id == tid}
            graph = EntityGraph().build(S, thread_facts)
            stale_ids = graph.stale_su_ids()
            users = sorted({s.user_id for s in S})
            owner_of = {fid: f.owner_user for fid, f in thread_facts.items()}

            index = ANNIndex(cfg.encode.proj_dim)
            index.add([s.su_id for s in S], P)
            turn_of = np.array([s.turn_idx for s in S])

            salience = SalienceTracker(cfg.memory.salience_alpha)
            proj_by_id = {s.su_id: P[i] for i, s in enumerate(S)}

            probes = [s for s in S if s.kind == "probe"]
            if max_probes_per_thread:
                probes = probes[:max_probes_per_thread]
            probe_turns = {p.turn_idx for p in probes}
            cursor = 0

            for p in probes:
                # advance salience only over turns strictly before this probe
                while cursor < len(S) and S[cursor].turn_idx < p.turn_idx:
                    salience.update(S[cursor], proj_by_id[S[cursor].su_id])
                    cursor += 1

                gold = set(req.get((tid, p.turn_idx), []))
                if not gold:
                    continue
                mask = turn_of < p.turn_idx
                if not mask.any():
                    continue
                ids, _ = index.search(P[pos[p.su_id]], M, mask=mask)
                cands = [S[pos[i]] for i in ids]
                assert all(c.turn_idx < p.turn_idx for c in cands), "future candidate leaked"

                sal = salience.get(tid, p.user_id)
                rows_q, rows_c, rows_rho = [], [], []
                for c in cands:
                    rows_q.append(P[pos[p.su_id]])
                    rows_c.append(P[pos[c.su_id]])
                    rows_rho.append(build_rho(
                        salience=sal,
                        turn_gap=p.turn_idx - c.turn_idx,
                        seconds_gap=p.ts - c.ts,
                        query_user=p.user_id, cand_user=c.user_id,
                        owner_user=owner_of.get(c.fact_id or ""),
                        users=users,
                        graph=graph.features(c.su_id, p.su_id),
                    ))
                X = pipe.features(np.stack(rows_q), np.stack(rows_c), np.stack(rows_rho))
                y = np.array([1.0 if (c.fact_id in gold) else 0.0 for c in cands], np.float32)
                self.examples.append(ProbeExample(
                    thread_id=tid, probe_turn=p.turn_idx, su_ids=[c.su_id for c in cands],
                    X=X.astype(np.float16), y=y,
                    is_stale=np.array([c.su_id in stale_ids for c in cands], np.float32),
                    tokens=np.array([len(c.text.split()) for c in cands], np.int32),
                    kinds=[c.kind for c in cands],
                ))
            _ = probe_turns
            if verbose:
                print(f"  [{ti + 1}/{len(tids)}] {tid}: {len(probes)} probes")
        return self

    # ------------------------------------------------------------------ #
    def flat(self, positives_per_probe: int | None = None, negatives_per_probe: int = 8,
             rng: np.random.Generator | None = None) -> tuple[np.ndarray, np.ndarray]:
        """Flatten to (X, y) with a bounded negative sample per probe.

        Using every candidate would make the label distribution ~1:255 and the scorer
        would learn to answer "no" to everything.
        """
        rng = rng or np.random.default_rng(0)
        Xs, ys = [], []
        for ex in self.examples:
            pos = np.flatnonzero(ex.y > 0)
            neg = np.flatnonzero(ex.y == 0)
            if positives_per_probe:
                pos = pos[:positives_per_probe]
            if len(neg) > negatives_per_probe:
                # keep the hardest negatives (stale facts) plus a random sample
                stale = neg[ex.is_stale[neg] > 0][: negatives_per_probe // 2]
                rest = rng.choice(neg, negatives_per_probe - len(stale), replace=False)
                neg = np.concatenate([stale, rest])
            idx = np.concatenate([pos, neg])
            Xs.append(ex.X[idx])
            ys.append(ex.y[idx])
        return np.concatenate(Xs), np.concatenate(ys)

    def stats(self) -> dict:
        n_pos = sum(int(e.y.sum()) for e in self.examples)
        n_cand = sum(len(e.y) for e in self.examples)
        return {
            "n_probes": len(self.examples),
            "n_candidates": n_cand,
            "n_positives": n_pos,
            "positive_rate": n_pos / max(n_cand, 1),
            "gold_present_rate": float(np.mean([e.y.max() > 0 for e in self.examples])),
            "stale_rate": float(np.mean([e.is_stale.mean() for e in self.examples])),
        }

    # ------------------------------------------------------------------ #
    def save(self, path: Path | None = None) -> Path:
        """Object arrays, not a stacked tensor: early probes legitimately have fewer than
        M causal candidates, so the per-probe rows are ragged."""
        path = Path(path or paths.DATA_PROCESSED / f"scoring_{self.split}.npz")
        def ragged(attr):
            arr = np.empty(len(self.examples), dtype=object)
            for i, e in enumerate(self.examples):
                arr[i] = getattr(e, attr)
            return arr
        np.savez_compressed(
            path,
            X=ragged("X"), y=ragged("y"), stale=ragged("is_stale"),
            tokens=ragged("tokens"), su_ids=ragged("su_ids"), kinds=ragged("kinds"),
            thread=np.array([e.thread_id for e in self.examples]),
            turn=np.array([e.probe_turn for e in self.examples]),
        )
        return path

    @classmethod
    def load(cls, cfg: Config, split: str = "train", path: Path | None = None
             ) -> "ScoringDataset":
        path = Path(path or paths.DATA_PROCESSED / f"scoring_{split}.npz")
        blob = np.load(path, allow_pickle=True)
        obj = cls(cfg, split)
        for i in range(len(blob["y"])):
            obj.examples.append(ProbeExample(
                thread_id=str(blob["thread"][i]), probe_turn=int(blob["turn"][i]),
                su_ids=list(blob["su_ids"][i]),
                X=blob["X"][i].astype(np.float32),   # stored fp16, used fp32
                y=blob["y"][i],
                is_stale=blob["stale"][i], tokens=blob["tokens"][i],
                kinds=list(blob["kinds"][i]),
            ))
        return obj
