"""Phase 1 · the corpus must be a valid measuring instrument.

These tests generate a small corpus in a temp dir rather than trusting whatever
happens to be in data/raw — a passing test on stale data is worse than no test.
"""

from __future__ import annotations

from collections import Counter

import pytest

from qcs.config import Config
from qcs.datagen.generate import write_corpus
from qcs.datagen.validate import CorpusError, oracle_vs_random, validate_corpus
from qcs.logging_ import read_jsonl
from qcs.seeding import child_rng
from qcs.datagen.thread_planner import plan_thread


@pytest.fixture(scope="module")
def small_corpus(tmp_path_factory):
    cfg = Config.load()
    cfg.data.n_threads_train = 3
    cfg.data.n_threads_test = 1
    cfg.data.turns_per_thread = 300
    cfg.data.n_facts_per_thread = 30
    cfg.data.probe_distances = [10, 50, 200]
    out = tmp_path_factory.mktemp("raw")
    counts = write_corpus(cfg, out_dir=out)
    return cfg, out, counts


def test_corpus_validates(small_corpus):
    cfg, out, _ = small_corpus
    report = validate_corpus(raw_dir=out, probe_distances=cfg.data.probe_distances)
    assert report["probes"] > 0
    assert report["train_threads"] == 3


def test_instrument_is_calibrated(small_corpus):
    """Oracle must be perfect and random must be weak, or nothing downstream means anything."""
    _, out, _ = small_corpus
    calib = oracle_vs_random(raw_dir=out, seed=0)
    assert calib["oracle_recall"] == 1.0
    assert calib["random_recall"] < 0.3


def test_every_probe_is_answerable(small_corpus):
    """The V1 invariant, checked directly rather than through the validator."""
    _, out, _ = small_corpus
    turns = read_jsonl(out / "threads_train.jsonl")
    by_thread: dict[str, list[dict]] = {}
    for t in turns:
        by_thread.setdefault(t["thread_id"], []).append(t)
    for ts in by_thread.values():
        ts.sort(key=lambda x: x["turn_idx"])
        stated: set[str] = set()
        for t in ts:
            if t["kind"] in ("fact", "update"):
                stated.add(t["fact_id"])
            elif t["kind"] == "probe":
                assert set(t["required_facts"]) <= stated


def test_no_probe_targets_a_superseded_fact(small_corpus):
    _, out, _ = small_corpus
    facts = {f["fact_id"]: f for f in read_jsonl(out / "fact_bank.jsonl")}
    for split in ("train", "test"):
        for t in read_jsonl(out / f"threads_{split}.jsonl"):
            for fid in t.get("required_facts", []):
                assert facts[fid]["superseded_by"] is None


def test_adversarial_regime_actually_contains_supersessions(small_corpus):
    _, out, _ = small_corpus
    facts = read_jsonl(out / "fact_bank.jsonl")
    adv = [f for f in facts if f["regime"] == "adversarial"]
    assert any(f["superseded_by"] for f in adv), "no supersessions ⇒ stale-fact test is vacuous"


def test_regeneration_is_deterministic(tmp_path):
    cfg = Config.load()
    cfg.data.n_threads_train = 2
    cfg.data.n_threads_test = 1
    cfg.data.turns_per_thread = 200
    cfg.data.n_facts_per_thread = 20
    a, b = tmp_path / "a", tmp_path / "b"
    write_corpus(cfg, out_dir=a)
    write_corpus(cfg, out_dir=b)
    assert (a / "threads_train.jsonl").read_bytes() == (b / "threads_train.jsonl").read_bytes()


def test_probe_bands_are_balanced():
    """Distances are assigned round-robin, so no band may dominate."""
    plan = plan_thread(
        thread_id="t", regime="multi_owner", n_turns=600, n_users=5, n_facts=60,
        probe_distances=[10, 50, 200, 500], supersede_ratio=0.2, distractor_ratio=0.25,
        rng=child_rng(1, "band-test"),
    )
    dists = [t.probe_distance for t in plan.turns if t.kind == "probe"]
    bands = Counter(min([10, 50, 200, 500], key=lambda b: abs(b - d)) for d in dists)
    assert len(bands) == 4
    assert max(bands.values()) / min(bands.values()) < 2.0


def test_validator_catches_a_broken_corpus(tmp_path):
    """A validator that never fails is not a validator."""
    from qcs.logging_ import write_jsonl

    write_jsonl(tmp_path / "fact_bank.jsonl", [
        {"fact_id": "t:f000", "superseded_by": None, "regime": "x", "attribute": "deadline",
         "owner_user": "Ada"},
    ])
    # probe at turn 0 requires a fact that is never stated
    write_jsonl(tmp_path / "threads_train.jsonl", [
        {"thread_id": "t", "turn_idx": 0, "user_id": "Ada", "kind": "probe", "text": "?",
         "ts": 0.0, "fact_id": None, "required_facts": ["t:f000"], "answer": "x",
         "probe_distance": 5},
    ])
    write_jsonl(tmp_path / "threads_test.jsonl", [])
    with pytest.raises(CorpusError):
        validate_corpus(raw_dir=tmp_path)
