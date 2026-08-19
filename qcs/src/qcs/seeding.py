"""Determinism. Randomness is a controlled variable in this project, not an excuse."""

from __future__ import annotations

import hashlib
import os
import random

import numpy as np


def seed_everything(seed: int) -> np.random.Generator:
    """Seed every RNG we might touch and return the project's canonical Generator.

    Modules should thread this Generator through explicitly rather than calling
    np.random.* directly — global state makes partial reruns irreproducible.
    """
    os.environ["PYTHONHASHSEED"] = str(seed)
    random.seed(seed)
    np.random.seed(seed)
    try:
        import torch

        torch.manual_seed(seed)
        torch.use_deterministic_algorithms(False)  # CPU kernels here are already deterministic
    except ImportError:
        pass
    return np.random.default_rng(seed)


def child_rng(seed: int, *tags: object) -> np.random.Generator:
    """A reproducible sub-stream keyed by (seed, tags).

    Lets thread 17 be regenerated identically without replaying threads 0..16.
    """
    # hashlib, not hash(): CPython's str hash is salted per process, so hash()
    # would silently produce different streams on different runs.
    key = "|".join([str(seed), *map(str, tags)]).encode()
    mix = int.from_bytes(hashlib.sha256(key).digest()[:8], "big")
    return np.random.default_rng(mix % (2**32))
