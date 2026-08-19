"""Disk-backed embedding cache keyed by sha256(text).

Encoding the corpus is the single most repeated expensive operation in the project
(every phase re-reads it). Caching by content hash — not by index — means the cache
survives regeneration, reordering, and partial reruns.
"""

from __future__ import annotations

import hashlib
from pathlib import Path

import numpy as np

from .. import paths
from .encoder import FrozenEncoder


def text_key(text: str) -> str:
    return hashlib.sha256(text.encode("utf-8")).hexdigest()[:16]


class EmbeddingCache:
    def __init__(self, encoder: FrozenEncoder, path: Path | None = None):
        self.encoder = encoder
        self.path = Path(path or paths.DATA_INTERIM / f"emb_cache_{encoder.backend}.npz")
        self.keys: dict[str, int] = {}
        self.vecs: np.ndarray = np.zeros((0, encoder.dim), dtype=np.float32)
        if self.path.exists():
            blob = np.load(self.path, allow_pickle=True)
            self.vecs = blob["vecs"]
            self.keys = {k: i for i, k in enumerate(blob["keys"].tolist())}

    def embed(self, texts: list[str], batch_size: int = 128) -> np.ndarray:
        missing = [t for t in texts if text_key(t) not in self.keys]
        if missing:
            uniq = list(dict.fromkeys(missing))
            new = self.encoder.encode(uniq, batch_size=batch_size)
            start = len(self.keys)
            self.vecs = np.concatenate([self.vecs, new], axis=0) if len(self.vecs) else new
            for j, t in enumerate(uniq):
                self.keys[text_key(t)] = start + j
        return np.stack([self.vecs[self.keys[text_key(t)]] for t in texts])

    def save(self) -> Path:
        self.path.parent.mkdir(parents=True, exist_ok=True)
        keys = np.array(sorted(self.keys, key=self.keys.get))
        np.savez_compressed(self.path, keys=keys, vecs=self.vecs)
        return self.path
