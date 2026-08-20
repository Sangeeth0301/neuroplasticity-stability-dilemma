"""Phases 5-7 · scorers, continual machinery, and the assembler."""

from __future__ import annotations

import numpy as np
import pytest
import torch

from qcs.assemble.budget import Budget, count_tokens
from qcs.assemble.knapsack import greedy_knapsack, topk_baseline, value_prior
from qcs.assemble.pipeline import ExtractiveReader
from qcs.classical.scorers import build_scorer
from qcs.config import Config
from qcs.eval.metrics import auc, evaluate_probes, mrr, recall_at_k, stale_above_current
from qcs.train.anchor import DriftDetector, FisherAnchor, TrustRegion
from qcs.train.loop import ContinualResult
from qcs.train.replay import ReservoirBuffer


class _Ex:
    def __init__(self, y, stale=None, X=None):
        self.y = np.asarray(y, np.float32)
        self.is_stale = np.zeros_like(self.y) if stale is None else np.asarray(stale, np.float32)
        self.X = np.zeros((len(self.y), 8), np.float32) if X is None else X
        self.su_ids = [f"s{i}" for i in range(len(self.y))]


# ------------------------------------------------------------------ scorers
@pytest.mark.parametrize("name", ["linear", "mlp", "cosine", "recency", "random"])
def test_every_scorer_returns_probabilities(name):
    cfg = Config.load()
    m = build_scorer(name, cfg)
    rng = np.random.default_rng(0)
    X = rng.normal(size=(16, cfg.encode.feature_dim)).astype(np.float32)
    X /= np.linalg.norm(X, axis=1, keepdims=True)
    r = m.score(X)
    assert r.shape == (16,)
    assert r.min() >= 0.0 and r.max() <= 1.0


def test_linear_twin_is_parameter_matched_to_the_circuit():
    """1,025 vs the QRE's 1,080 — the closest classical model of equal size."""
    from qcs.quantum.circuit import QRE

    cfg = Config.load()
    assert abs(build_scorer("linear", cfg).n_circuit_params - QRE(cfg).n_circuit_params) < 100


def test_classical_arms_do_not_wrap_angles():
    """wrap_angles must be a no-op classically — that asymmetry is the experiment."""
    cfg = Config.load()
    m = build_scorer("linear", cfg)
    with torch.no_grad():
        m.theta.add_(17.0)
    before = m.theta_norm()
    m.wrap_angles()
    assert m.theta_norm() == before


# ------------------------------------------------------------------ metrics
def test_ranking_metrics_on_hand_computed_cases():
    s = np.array([0.9, 0.8, 0.7, 0.6])
    y = np.array([0, 1, 0, 1])
    assert recall_at_k(s, y, 2) == 0.5
    assert mrr(s, y) == pytest.approx(0.5)
    # positives {0.8, 0.6} vs negatives {0.9, 0.7}: only (0.8 > 0.7) of the four pairs
    # is correctly ordered, so AUC = 1/4
    assert auc(s, y) == pytest.approx(0.25)


def test_auc_handles_ties():
    """All-equal scores must give exactly 0.5, not an arbitrary value."""
    assert auc(np.ones(6), np.array([1, 1, 0, 0, 0, 1])) == pytest.approx(0.5)


def test_stale_above_current_detects_the_failure():
    y = np.array([0, 1, 0])
    stale = np.array([1, 0, 0])
    assert stale_above_current(np.array([0.9, 0.5, 0.1]), y, stale) == 1.0
    assert stale_above_current(np.array([0.1, 0.9, 0.5]), y, stale) == 0.0


def test_evaluate_probes_averages_over_probes():
    out = evaluate_probes(lambda ex: np.array([0.9, 0.1]), [_Ex([1, 0]), _Ex([0, 1])])
    assert out["rank1"] == 0.5
    assert out["n_probes"] == 2


# ------------------------------------------------------------------ replay
def test_reservoir_is_uniform_over_history():
    buf = ReservoirBuffer(200, 4, seed=0)
    for t in range(50):
        buf.add(np.zeros((20, 4), np.float32), np.zeros(20, np.float32), t)
    assert buf.size == 200
    assert buf.oldest_task() < 10, "reservoir dropped early history — it became a FIFO"
    assert buf.task_coverage() > 20


def test_replay_sample_shapes():
    buf = ReservoirBuffer(10, 4, seed=0)
    buf.add(np.ones((5, 4), np.float32), np.ones(5, np.float32), 0)
    X, y = buf.sample(3)
    assert X.shape == (3, 4) and y.shape == (3,)


# ------------------------------------------------------------------ anchor
def test_anchor_penalty_is_zero_at_the_checkpoint_and_grows_away_from_it():
    cfg = Config.load()
    m = build_scorer("linear", cfg)
    a = FisherAnchor(1.0)
    a.checkpoint(m, np.ones(m.theta.numel()))
    assert float(a.penalty(m)) == pytest.approx(0.0)
    with torch.no_grad():
        m.theta.add_(0.5)
    assert float(a.penalty(m)) > 0
    assert a.pull(m) > 0


def test_anchor_disabled_when_lambda_is_zero():
    cfg = Config.load()
    m = build_scorer("linear", cfg)
    a = FisherAnchor(0.0)
    a.checkpoint(m, np.ones(m.theta.numel()))
    with torch.no_grad():
        m.theta.add_(5.0)
    assert float(a.penalty(m)) == 0.0


def test_trust_region_clips_and_reports():
    cfg = Config.load()
    m = build_scorer("linear", cfg)
    tr = TrustRegion(0.1)
    start = tr.start_task(m)
    with torch.no_grad():
        m.theta.add_(1.0)
    tr.apply(m, start)
    assert float((m.theta - start).abs().max()) <= 0.1 + 1e-6
    assert tr.clip_rate == 1.0


def test_drift_detector_fires_on_a_shift_and_not_on_noise():
    d = DriftDetector(0.25)
    rng = np.random.default_rng(0)
    d.update(rng.uniform(0, 0.2, 500))
    _, quiet = d.update(rng.uniform(0, 0.2, 500))
    _, shifted = d.update(rng.uniform(0.8, 1.0, 500))
    assert not quiet and shifted


# ------------------------------------------------------------------ CL math
def test_bwt_fwt_on_a_known_matrix():
    A = np.array([[0.8, 0.0], [0.5, 0.9]])
    r = ContinualResult("x", A)
    assert r.bwt() == pytest.approx(0.5 - 0.8)
    assert r.plasticity_index(window=1) == pytest.approx(0.9 / 0.8)


# ------------------------------------------------------------------ assembler
def test_budget_reserves_must_leave_room():
    with pytest.raises(AssertionError):
        Budget(100, 50, 40, 30).validate()


def test_knapsack_never_exceeds_the_budget():
    rng = np.random.default_rng(0)
    for _ in range(200):
        n = int(rng.integers(2, 40))
        s = rng.random(n)
        tok = rng.integers(1, 60, n)
        B = int(rng.integers(20, 300))
        sel = greedy_knapsack(s, tok, B)
        assert sel.tokens_used <= B


def test_knapsack_prefers_gain_per_token_over_raw_score():
    scores = np.array([0.9, 0.5])
    tokens = np.array([100, 5])
    sel = greedy_knapsack(scores, tokens, budget=100)
    assert sel.idx[0] == 1, "cheap-but-decent must be picked before expensive-but-great"


def test_redundancy_penalty_suppresses_near_duplicates():
    v = np.array([[1.0, 0.0], [0.999, 0.01], [0.0, 1.0]], np.float32)
    scores = np.array([0.9, 0.89, 0.6])
    tokens = np.array([10, 10, 10])
    picked = greedy_knapsack(scores, tokens, 20, vectors=v, redundancy_lambda=0.9).idx
    assert set(picked) == {0, 2}, "the near-duplicate should lose to the diverse item"


def test_topk_baseline_can_waste_the_budget_on_duplicates():
    v = np.array([[1.0, 0.0], [0.999, 0.01], [0.0, 1.0]], np.float32)
    scores = np.array([0.9, 0.89, 0.6])
    tokens = np.array([10, 10, 10])
    assert set(topk_baseline(scores, tokens, 20).idx) == {0, 1}


def test_value_prior_ranks_facts_above_filler():
    v = value_prior(["fact", "filler", "update"], {"fact": 1.0, "filler": 0.3})
    assert v[0] == 1.0 and v[2] == 1.0 and v[1] == 0.3


def test_reader_ignores_facts_marked_outdated():
    """The stale-fact rule, tested at the reader: an OUTDATED line must never be the
    source of an answer."""
    prompt = "[Ada · turn 3] OUTDATED: due March 14\n[Ada · turn 40] CURRENT: due June 3"

    class F:
        def __init__(self, v):
            self.value = v

    assert ExtractiveReader.answer(prompt, [F("June 3")]) == "June 3"
    assert ExtractiveReader.answer(prompt, [F("March 14")]) is None


def test_token_counting_is_positive_and_monotone():
    assert count_tokens("one two three") >= 3
    assert count_tokens("a b c d e f") > count_tokens("a b")
