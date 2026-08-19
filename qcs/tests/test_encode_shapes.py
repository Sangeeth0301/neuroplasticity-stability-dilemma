"""Phase 2 · the classical→Hilbert interface.

The assertions here are the ones that keep amplitude encoding valid. If any fails, the
"quantum state" the circuit receives is not a quantum state.
"""

from __future__ import annotations

import numpy as np
import pytest

from qcs.config import Config
from qcs.encode.assemble import (assemble, block_norm_shares, block_slices,
                                 participation_ratio)
from qcs.encode.relational import build_rho, recency_features, usage_features
from qcs.encode.whiten import Whitener
from qcs.ingest.segment import normalize, segment_turn
from qcs.types import Turn


def _turn(text: str) -> Turn:
    return Turn(thread_id="t", turn_idx=3, user_id="Ada", kind="filler", text=text, ts=1.0)


def test_segment_roundtrip():
    """Concatenating a turn's semantic units reproduces the normalised turn exactly."""
    text = "The Helios dashboard is due on March 14. Bo owns it.  Let's move on."
    sus = segment_turn(_turn(text), target_tokens=8)
    assert len(sus) > 1
    assert " ".join(s.text for s in sus) == normalize(text)


def test_segment_is_deterministic():
    t = _turn("One. Two. Three. Four. Five.")
    assert [s.text for s in segment_turn(t, 6)] == [s.text for s in segment_turn(t, 6)]


def test_assemble_is_unit_norm_and_1024d():
    rng = np.random.default_rng(0)
    x = assemble(rng.normal(size=256), rng.normal(size=256), rng.normal(size=256) * 0.1)
    assert x.shape == (1024,)
    assert abs(np.linalg.norm(x) - 1.0) < 1e-6


def test_assemble_batches_match_single():
    rng = np.random.default_rng(1)
    q, c, r = rng.normal(size=(4, 256)), rng.normal(size=(4, 256)), rng.normal(size=(4, 256))
    X = assemble(q, c, r)
    assert X.shape == (4, 1024)
    for i in range(4):
        assert np.allclose(X[i], assemble(q[i], c[i], r[i]), atol=1e-6)


def test_block_norm_targets_are_actually_hit():
    """The fix for the block-dominance failure: shares must land on their targets
    regardless of the incoming scales."""
    rng = np.random.default_rng(2)
    targets = [0.35, 0.35, 0.20, 0.10]
    X = assemble(
        rng.normal(size=(64, 256)) * 0.001,     # tiny
        rng.normal(size=(64, 256)) * 1000.0,    # huge
        rng.normal(size=(64, 256)),
        block_norm_targets=targets,
    )
    shares = block_norm_shares(X)
    for name, target in zip(shares, targets):
        assert abs(shares[name] - target) < 1e-4, f"{name}: {shares[name]:.4f} vs {target}"


def test_unbalanced_blocks_would_collapse_participation():
    """Demonstrates why balancing exists: without it, one block eats the register."""
    rng = np.random.default_rng(3)
    q, c, r = rng.normal(size=(32, 256)), rng.normal(size=(32, 256)), rng.normal(size=(32, 256))
    balanced = assemble(q, c, r)
    skewed = assemble(q, c, r, block_norm_targets=[0.01, 0.01, 0.01, 0.97])
    assert participation_ratio(balanced) > 2 * participation_ratio(skewed)


def test_rho_features_are_bounded():
    """Every ρ feature must be in [-1,1] BEFORE normalisation, or its magnitude is lost."""
    rho = build_rho(
        salience=np.random.default_rng(4).normal(size=64) * 50,
        turn_gap=100_000, seconds_gap=1e9, query_user="Ada", cand_user="Bo",
        owner_user="Bo", users=["Ada", "Bo", "Chen"], times_selected=999, times_seen=9999,
    )
    assert rho.shape == (256,)
    assert np.all(np.abs(rho) <= 1.0 + 1e-6)


def test_recency_is_monotone_and_saturating():
    v = [recency_features(g, g * 137.0)[0] for g in (0, 10, 50, 200, 1000)]
    assert all(a < b for a, b in zip(v, v[1:]))
    assert v[-1] < 1.0


def test_usage_features_saturate():
    assert usage_features(10_000, 10_000)[0] <= 1.0


def test_block_slices_partition_the_vector():
    sl = block_slices(1024)
    covered = np.zeros(1024, dtype=int)
    for s in sl.values():
        covered[s] += 1
    assert (covered == 1).all()


def test_whitener_refuses_to_refit():
    rng = np.random.default_rng(5)
    w = Whitener(8).fit(rng.normal(size=(64, 16)))
    with pytest.raises(RuntimeError, match="already fitted"):
        w.fit(rng.normal(size=(64, 16)))


def test_whitener_flattens_the_spectrum():
    rng = np.random.default_rng(6)
    X = rng.normal(size=(512, 32)) @ np.diag(np.linspace(10, 0.1, 32))
    w = Whitener(32, eps=1e-6).fit(X)
    before, after = w.condition_number(X)
    assert after < before / 10


def test_config_feature_dim_matches_encoder_blocks():
    cfg = Config.load()
    assert cfg.encode.proj_dim * 4 == cfg.encode.feature_dim
