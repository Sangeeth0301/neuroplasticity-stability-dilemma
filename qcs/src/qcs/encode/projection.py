"""The learned projection P: whitened embedding → proj_dim, trained once then frozen.

This is the last classical component that learns anything about *semantics*. It is
trained offline, on the training split only, with an InfoNCE objective whose positives
are (probe, required-fact) pairs and whose negatives are hard in-thread distractors —
turns that mention the same entity but carry no usable value.

After Phase 2 it is frozen forever. Everything that adapts during the continual run
lives on the quantum manifold; keeping P frozen is what makes that claim true.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np


class Projection:
    """A plain linear map. Kept deliberately simple: it must be frozen, inspectable,
    and cheap enough to apply to every candidate at scoring time."""

    def __init__(self, in_dim: int, out_dim: int, seed: int = 0):
        rng = np.random.default_rng(seed)
        self.P = (rng.normal(0, 1 / np.sqrt(in_dim), (out_dim, in_dim))).astype(np.float32)
        self.in_dim, self.out_dim = in_dim, out_dim
        self.history: list[dict] = []

    def __call__(self, X: np.ndarray) -> np.ndarray:
        return np.asarray(X, dtype=np.float32) @ self.P.T

    # ------------------------------------------------------------------ #
    def fit_infonce(
        self,
        anchors: np.ndarray,
        positives: np.ndarray,
        negatives: np.ndarray,
        epochs: int = 30,
        lr: float = 3e-3,
        temperature: float = 0.07,
        batch_size: int = 256,
        seed: int = 0,
    ) -> list[dict]:
        """InfoNCE with explicit hard negatives.

        anchors   (n, d)      probe embeddings (whitened)
        positives (n, d)      the required fact's embedding
        negatives (n, k, d)   hard in-thread distractors
        """
        import torch

        torch.manual_seed(seed)
        dev = "cpu"
        P = torch.nn.Parameter(torch.tensor(self.P, device=dev))
        opt = torch.optim.Adam([P], lr=lr)
        A = torch.tensor(anchors, dtype=torch.float32, device=dev)
        Pos = torch.tensor(positives, dtype=torch.float32, device=dev)
        Neg = torch.tensor(negatives, dtype=torch.float32, device=dev)
        n = len(A)

        for ep in range(epochs):
            perm = torch.randperm(n)
            tot, nb, acc = 0.0, 0, 0.0
            for i in range(0, n, batch_size):
                idx = perm[i:i + batch_size]
                a = torch.nn.functional.normalize(A[idx] @ P.T, dim=-1)
                p = torch.nn.functional.normalize(Pos[idx] @ P.T, dim=-1)
                g = torch.nn.functional.normalize(Neg[idx] @ P.T, dim=-1)   # (b, k, o)
                pos_sim = (a * p).sum(-1, keepdim=True)                      # (b, 1)
                neg_sim = torch.einsum("bo,bko->bk", a, g)                   # (b, k)
                logits = torch.cat([pos_sim, neg_sim], dim=1) / temperature
                target = torch.zeros(len(idx), dtype=torch.long, device=dev)
                loss = torch.nn.functional.cross_entropy(logits, target)
                opt.zero_grad()
                loss.backward()
                opt.step()
                tot += float(loss)
                acc += float((logits.argmax(1) == 0).float().mean())
                nb += 1
            self.history.append({"epoch": ep, "loss": tot / nb, "rank1_acc": acc / nb})

        self.P = P.detach().cpu().numpy().astype(np.float32)
        return self.history

    # ------------------------------------------------------------------ #
    def save(self, path: str | Path) -> None:
        Path(path).parent.mkdir(parents=True, exist_ok=True)
        np.savez(path, P=self.P, history=np.array(self.history, dtype=object),
                 in_dim=self.in_dim, out_dim=self.out_dim)

    @classmethod
    def load(cls, path: str | Path) -> "Projection":
        blob = np.load(path, allow_pickle=True)
        obj = cls(int(blob["in_dim"]), int(blob["out_dim"]))
        obj.P = blob["P"]
        obj.history = list(blob["history"])
        return obj
