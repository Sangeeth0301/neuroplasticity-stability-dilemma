"""Phase 3 · the stability manifold must be genuinely append-only.

The whole stability argument of this architecture reduces to these tests. If a semantic
unit can be mutated or deleted, thread knowledge can be lost, and the claim that
forgetting is "only a retrieval miss" is false.
"""

from __future__ import annotations

import sqlite3

import numpy as np
import pytest

from qcs.encode.relational import GraphFeatures
from qcs.memory.ann_index import ANNIndex
from qcs.memory.consolidate import consolidate_thread, verify_sources_intact
from qcs.memory.graph import EntityGraph
from qcs.memory.salience import SalienceTracker, compress
from qcs.memory.store import MemoryStore
from qcs.types import Fact, SemanticUnit


def _su(i: int, thread="t", kind="filler", fact_id=None, user="Ada", text=None):
    return SemanticUnit(
        su_id=f"{thread}:{i:05d}:0", thread_id=thread, turn_idx=i, su_idx=0, user_id=user,
        kind=kind, text=text or f"turn {i} text", ts=float(i) * 137.0, fact_id=fact_id,
    )


@pytest.fixture()
def store(tmp_path):
    s = MemoryStore(tmp_path / "t.db")
    yield s
    s.close()


def test_update_is_rejected_by_the_database(store):
    store.write_sus([_su(0)])
    with pytest.raises(sqlite3.IntegrityError, match="append-only"):
        store.con.execute("UPDATE semantic_units SET text = 'tampered' WHERE su_id = ?",
                          ("t:00000:0",))


def test_delete_is_rejected_by_the_database(store):
    store.write_sus([_su(0)])
    with pytest.raises(sqlite3.IntegrityError, match="append-only"):
        store.con.execute("DELETE FROM semantic_units WHERE su_id = ?", ("t:00000:0",))


def test_rewriting_the_same_unit_does_not_change_it(store):
    store.write_sus([_su(0, text="original")])
    store.write_sus([_su(0, text="replacement attempt")])
    assert store.su("t:00000:0").text == "original"
    assert store.counts()["semantic_units"] == 1


def test_consolidation_adds_and_never_modifies(store):
    sus = [_su(i, kind="fact" if i % 10 == 0 else "filler",
               fact_id=f"t:f{i:03d}" if i % 10 == 0 else None) for i in range(30)]
    store.write_sus(sus)
    store.write_facts([Fact(fact_id=f"t:f{i:03d}", thread_id="t", owner_user="Ada",
                            entity="Helios dashboard", attribute="deadline", value="March 14",
                            regime="single_owner", introduced_turn=i)
                       for i in range(0, 30, 10)])
    before = {s.su_id: s.text for s in store.sus(thread_id="t")}
    ids = consolidate_thread(store, "t", every=10)
    assert len(ids) == 3
    assert verify_sources_intact(store, "t", before)
    assert store.counts()["semantic_units"] == 30       # nothing consumed


def test_consolidation_is_idempotent(store):
    store.write_sus([_su(i) for i in range(20)])
    first = consolidate_thread(store, "t", every=10)
    second = consolidate_thread(store, "t", every=10)
    assert len(first) == 2 and second == []


def test_ann_search_respects_causality():
    """A candidate from the future must be unreachable, or the benchmark leaks."""
    rng = np.random.default_rng(0)
    V = rng.normal(size=(50, 16)).astype(np.float32)
    idx = ANNIndex(16)
    idx.add([f"s{i}" for i in range(50)], V)
    mask = np.zeros(50, dtype=bool)
    mask[:10] = True
    ids, _ = idx.search(V[40], k=5, mask=mask)
    assert all(int(i[1:]) < 10 for i in ids)


def test_ann_returns_exact_top1_without_mask():
    rng = np.random.default_rng(1)
    V = rng.normal(size=(200, 32)).astype(np.float32)
    idx = ANNIndex(32)
    idx.add([f"s{i}" for i in range(200)], V)
    idx.build()
    ids, _ = idx.search(V[7], k=1)
    assert ids[0] == "s7"


def test_graph_marks_superseded_facts():
    facts = {
        "f0": Fact("f0", "t", "Ada", "Helios dashboard", "deadline", "March 14",
                   "adversarial", 1, None, "f1"),
        "f1": Fact("f1", "t", "Ada", "Helios dashboard", "deadline", "June 3",
                   "adversarial", 40, "f0", None),
    }
    sus = [_su(1, kind="fact", fact_id="f0", text="The Helios dashboard is due on March 14."),
           _su(40, kind="update", fact_id="f1", text="Update — the Helios dashboard is June 3."),
           _su(60, kind="probe", text="When is the Helios dashboard due?")]
    g = EntityGraph().build(sus, facts)
    stale = g.stale_su_ids()
    assert sus[0].su_id in stale and sus[1].su_id not in stale
    kinds = {k for _, _, k, _ in g.edges()}
    assert {"supersedes", "contradicts", "shares_entity"} <= kinds


def test_graph_features_are_bounded_and_informative():
    facts = {"f0": Fact("f0", "t", "Ada", "Onyx audit", "budget", "$5,000",
                        "multi_owner", 1)}
    sus = [_su(1, kind="fact", fact_id="f0", text="The Onyx audit budget is $5,000."),
           _su(9, kind="probe", text="What is the Onyx audit budget?"),
           _su(5, text="unrelated chatter")]
    g = EntityGraph().build(sus, facts)
    f_rel = g.features(sus[0].su_id, sus[1].su_id)
    f_unrel = g.features(sus[2].su_id, sus[1].su_id)
    assert f_rel.shares_entity_with_query == 1.0
    assert f_unrel.shares_entity_with_query == 0.0
    for f in (f_rel, f_unrel):
        assert np.all(np.abs(f.vector()) <= 1.0)


def test_salience_separates_users():
    rng = np.random.default_rng(2)
    tr = SalienceTracker(alpha=0.3)
    a_dir, b_dir = rng.normal(size=256), rng.normal(size=256)
    for _ in range(20):
        tr.update(_su(1, user="Ada"), a_dir + 0.1 * rng.normal(size=256))
        tr.update(_su(2, user="Bo"), b_dir + 0.1 * rng.normal(size=256))
    assert tr.separation() > 0.05
    assert compress(np.ones(256), 64).shape == (64,)


def test_graph_features_default_to_unknown_not_wrong():
    """Missing graph info must read as 'unknown' (zeros), never as a positive claim."""
    assert np.all(GraphFeatures().vector() == 0.0)
