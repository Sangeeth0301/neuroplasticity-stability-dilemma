"""Reservoir replay — uniform over the whole history, not over a recent window.

A FIFO buffer would hold only recent tasks and would therefore protect against nothing:
the point of replay is to keep the scorer accountable to material it saw hundreds of
tasks ago. Reservoir sampling gives every example seen so far the same probability of
being in the buffer, using O(capacity) memory and no knowledge of the stream length.
"""

from __future__ import annotations

import numpy as np


class ReservoirBuffer:
    def __init__(self, capacity: int, feature_dim: int, seed: int = 0):
        self.capacity = capacity
        self.X = np.zeros((capacity, feature_dim), dtype=np.float32)
        self.y = np.zeros(capacity, dtype=np.float32)
        self.task = np.full(capacity, -1, dtype=np.int32)
        self.n_seen = 0
        self.size = 0
        self.rng = np.random.default_rng(seed)

    def add(self, X: np.ndarray, y: np.ndarray, task_id: int) -> None:
        for i in range(len(X)):
            self.n_seen += 1
            if self.size < self.capacity:
                slot = self.size
                self.size += 1
            else:
                j = int(self.rng.integers(0, self.n_seen))
                if j >= self.capacity:
                    continue
                slot = j
            self.X[slot] = X[i]
            self.y[slot] = y[i]
            self.task[slot] = task_id

    def sample(self, n: int) -> tuple[np.ndarray, np.ndarray]:
        if self.size == 0:
            return np.zeros((0, self.X.shape[1]), np.float32), np.zeros(0, np.float32)
        idx = self.rng.integers(0, self.size, size=min(n, self.size))
        return self.X[idx], self.y[idx]

    def task_coverage(self) -> int:
        return int(len(np.unique(self.task[: self.size])))

    def oldest_task(self) -> int:
        return int(self.task[: self.size].min()) if self.size else -1
