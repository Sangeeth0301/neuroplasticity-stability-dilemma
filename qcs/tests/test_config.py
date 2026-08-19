"""Phase 0 · configuration and determinism."""

from __future__ import annotations

import numpy as np
import pytest

from qcs.config import Config
from qcs.seeding import child_rng, seed_everything


def test_base_config_loads_and_validates():
    cfg = Config.load()
    cfg.validate()
    assert cfg.encode.feature_dim == 2 ** cfg.quantum.n_qubits


def test_unknown_key_rejected():
    with pytest.raises(KeyError):
        Config.from_dict({"nonsense": 1})


def test_hash_is_content_addressed():
    a, b = Config.load(), Config.load()
    assert a.hash() == b.hash()
    b.quantum.depth += 1
    assert a.hash() != b.hash()


def test_light_cone_floor_enforced():
    cfg = Config.load()
    cfg.quantum.depth = 2  # below n_qubits // 2
    with pytest.raises(AssertionError, match="light-cone"):
        cfg.validate()


def test_feature_dim_must_match_hilbert_dim():
    cfg = Config.load()
    cfg.encode.feature_dim = 900
    with pytest.raises(AssertionError, match="amplitude encoding"):
        cfg.validate()


def test_child_rng_is_process_stable():
    """Uses hashlib, not hash(): CPython salts str hashes per process."""
    a = child_rng(7, "train", 3).random(5)
    b = child_rng(7, "train", 3).random(5)
    c = child_rng(7, "train", 4).random(5)
    assert np.allclose(a, b)
    assert not np.allclose(a, c)


def test_seed_everything_returns_reproducible_generator():
    x = seed_everything(11).normal(size=8)
    y = seed_everything(11).normal(size=8)
    assert np.allclose(x, y)
