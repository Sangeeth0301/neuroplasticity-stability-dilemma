"""The frozen sentence encoder.

Two hard rules:
  1. The weights are never updated. Any gradient reaching this module is a bug — thread
     knowledge belongs in the memory manifold ℳ, not in encoder parameters.
  2. The model revision is recorded with every run, because a silently upgraded encoder
     invalidates every cached embedding and every comparison against earlier results.

A deterministic hashing fallback keeps the pipeline runnable with no network access.
It is NOT a semantic encoder — it exists so that structure/plumbing work never blocks
on a model download, and every run records which backend was used.
"""

from __future__ import annotations

import hashlib
from dataclasses import dataclass

import numpy as np


@dataclass
class EncoderInfo:
    backend: str
    model_name: str
    dim: int
    revision: str


class FrozenEncoder:
    def __init__(self, model_name: str, dim: int, allow_fallback: bool = True):
        self.model_name = model_name
        self.dim = dim
        self._model = None
        self.backend = "hash"
        self.revision = "n/a"
        try:
            from sentence_transformers import SentenceTransformer

            self._model = SentenceTransformer(model_name)
            self._model.eval()
            for p in self._model.parameters():
                p.requires_grad_(False)          # rule 1, enforced not merely intended
            self.dim = int(self._model.get_sentence_embedding_dimension())
            self.backend = "sentence-transformers"
            self.revision = getattr(self._model, "model_card_vars", {}).get("revision", "local")
        except Exception as exc:  # noqa: BLE001
            if not allow_fallback:
                raise
            self._fallback_reason = str(exc)[:200]

    @property
    def info(self) -> EncoderInfo:
        return EncoderInfo(self.backend, self.model_name, self.dim, self.revision)

    def encode(self, texts: list[str], batch_size: int = 128) -> np.ndarray:
        if self._model is not None:
            import torch

            with torch.no_grad():
                vecs = self._model.encode(
                    texts, batch_size=batch_size, convert_to_numpy=True,
                    show_progress_bar=False, normalize_embeddings=False,
                )
            return np.asarray(vecs, dtype=np.float32)
        return np.stack([self._hash_embed(t) for t in texts]).astype(np.float32)

    def _hash_embed(self, text: str) -> np.ndarray:
        """Deterministic bag-of-words random projection. Reproducible across processes."""
        v = np.zeros(self.dim, dtype=np.float32)
        for tok in text.lower().split():
            h = int.from_bytes(hashlib.sha256(tok.encode()).digest()[:8], "big")
            rng = np.random.default_rng(h % (2**32))
            v += rng.normal(0, 1, self.dim).astype(np.float32)
        n = np.linalg.norm(v)
        return v / n if n > 0 else v
