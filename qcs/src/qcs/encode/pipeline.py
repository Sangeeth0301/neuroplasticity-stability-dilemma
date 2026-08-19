"""The full classical encoding pipeline as one object.

    text → frozen encoder → whitening → projection → (with ρ) → 1024-d unit vector

Everything here is frozen after Phase 2. This object is what the quantum layer, the
classical baseline, and the memory store all share, so that an A/B between arms differs
in exactly one component.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import numpy as np

from .. import paths
from ..config import Config
from ..types import SemanticUnit
from .assemble import assemble
from .cache import EmbeddingCache
from .encoder import FrozenEncoder
from .projection import Projection
from .relational import GraphFeatures, build_rho
from .whiten import Whitener


@dataclass
class EncodedCorpus:
    """Projected vectors for a set of semantic units, in a stable order."""

    su_ids: list[str]
    P: np.ndarray                      # (n, proj_dim)
    index: dict[str, int]

    def get(self, su_id: str) -> np.ndarray:
        return self.P[self.index[su_id]]


class EncodingPipeline:
    def __init__(self, cfg: Config):
        self.cfg = cfg
        self.encoder = FrozenEncoder(cfg.encode.model_name, cfg.encode.embed_dim)
        self.cache = EmbeddingCache(self.encoder)
        self.whitener = Whitener(cfg.encode.whiten_components, eps=cfg.encode.whiten_eps)
        self.projection: Projection | None = None

    # ---------------- fitting (Phase 2 only) ---------------- #
    def embed_texts(self, texts: list[str]) -> np.ndarray:
        return self.cache.embed(texts, batch_size=self.cfg.encode.batch_size)

    def fit_whitener(self, texts: list[str]) -> np.ndarray:
        E = self.embed_texts(texts)
        self.whitener.fit(E)
        return E

    def project(self, texts: list[str]) -> np.ndarray:
        assert self.projection is not None, "projection not fitted/loaded"
        return self.projection(self.whitener.transform(self.embed_texts(texts)))

    def encode_sus(self, sus: list[SemanticUnit]) -> EncodedCorpus:
        P = self.project([s.text for s in sus])
        ids = [s.su_id for s in sus]
        return EncodedCorpus(ids, P, {sid: i for i, sid in enumerate(ids)})

    # ---------------- inference ---------------- #
    def features(
        self,
        p_q: np.ndarray,
        p_c: np.ndarray,
        rho: np.ndarray,
    ) -> np.ndarray:
        return assemble(
            p_q, p_c, rho,
            block_norm_targets=self.cfg.encode.block_norm_targets,
            feature_dim=self.cfg.encode.feature_dim,
        )

    def triple_features(
        self,
        query: SemanticUnit,
        cands: list[SemanticUnit],
        enc: EncodedCorpus,
        salience: np.ndarray,
        users: list[str],
        owner_of: dict[str, str] | None = None,
        graph: dict[str, GraphFeatures] | None = None,
    ) -> np.ndarray:
        """Batch: one row per candidate, each a unit-norm 1024-vector."""
        p_q = enc.get(query.su_id)
        rows_q, rows_c, rows_rho = [], [], []
        for c in cands:
            rows_q.append(p_q)
            rows_c.append(enc.get(c.su_id))
            rows_rho.append(
                build_rho(
                    salience=salience,
                    turn_gap=query.turn_idx - c.turn_idx,
                    seconds_gap=query.ts - c.ts,
                    query_user=query.user_id,
                    cand_user=c.user_id,
                    owner_user=(owner_of or {}).get(c.fact_id or ""),
                    users=users,
                    graph=(graph or {}).get(c.su_id),
                )
            )
        return self.features(np.stack(rows_q), np.stack(rows_c), np.stack(rows_rho))

    # ---------------- persistence ---------------- #
    def save(self, out_dir: Path | None = None) -> dict[str, str]:
        out_dir = Path(out_dir or paths.DATA_PROCESSED)
        out_dir.mkdir(parents=True, exist_ok=True)
        self.whitener.save(out_dir / "whiten.npz")
        assert self.projection is not None
        self.projection.save(out_dir / "projection.npz")
        self.cache.save()
        return {"whiten": str(out_dir / "whiten.npz"),
                "projection": str(out_dir / "projection.npz"),
                "cache": str(self.cache.path)}

    @classmethod
    def load(cls, cfg: Config, in_dir: Path | None = None) -> "EncodingPipeline":
        in_dir = Path(in_dir or paths.DATA_PROCESSED)
        obj = cls(cfg)
        obj.whitener = Whitener.load(in_dir / "whiten.npz")
        obj.projection = Projection.load(in_dir / "projection.npz")
        return obj
