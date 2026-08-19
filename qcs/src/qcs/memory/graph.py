"""The entity / supersession DAG.

This is how the architecture represents disagreement instead of resolving it by
overwrite. "Ada said the deadline is March 14" and "Ada later said it is June 3" are two
nodes joined by a `supersedes` edge — both survive, and the assembler decides what to do
with them (Phase 7 drops the stale one or labels it explicitly).

It also fills the 32 graph dimensions of ρ that are zero until this phase exists.
"""

from __future__ import annotations

import re
from collections import defaultdict

from ..encode.relational import GraphFeatures
from ..types import Fact, SemanticUnit


def entity_tokens(entity: str) -> list[str]:
    return [t for t in re.split(r"\W+", entity.lower()) if len(t) > 2]


class EntityGraph:
    """Built per thread from the append-only units plus the declared facts."""

    def __init__(self) -> None:
        self.su_entities: dict[str, set[str]] = defaultdict(set)
        self.entity_sus: dict[str, set[str]] = defaultdict(set)
        self.su_fact: dict[str, str] = {}
        self.fact_su: dict[str, str] = {}
        self.supersedes: dict[str, str] = {}      # newer fact -> older fact
        self.superseded_by: dict[str, str] = {}   # older fact -> newer fact
        self.degree: dict[str, int] = defaultdict(int)

    def build(self, sus: list[SemanticUnit], facts: dict[str, Fact]) -> "EntityGraph":
        entities = {f.entity for f in facts.values()}
        for su in sus:
            low = su.text.lower()
            for e in entities:
                toks = entity_tokens(e)
                if toks and all(t in low for t in toks):
                    self.su_entities[su.su_id].add(e)
                    self.entity_sus[e].add(su.su_id)
            if su.fact_id:
                self.su_fact[su.su_id] = su.fact_id
                self.fact_su.setdefault(su.fact_id, su.su_id)
        for f in facts.values():
            if f.supersedes:
                self.supersedes[f.fact_id] = f.supersedes
                self.superseded_by[f.supersedes] = f.fact_id
        for su_id, es in self.su_entities.items():
            self.degree[su_id] = sum(len(self.entity_sus[e]) for e in es)
        return self

    # ------------------------------------------------------------------ #
    def edges(self) -> list[tuple[str, str, str, float]]:
        """Materialised edges for the store: supersedes + shares_entity."""
        out: list[tuple[str, str, str, float]] = []
        for new_f, old_f in self.supersedes.items():
            if new_f in self.fact_su and old_f in self.fact_su:
                out.append((self.fact_su[new_f], self.fact_su[old_f], "supersedes", 1.0))
                out.append((self.fact_su[old_f], self.fact_su[new_f], "contradicts", 1.0))
        for e, sus in self.entity_sus.items():
            ordered = sorted(sus)
            for a, b in zip(ordered, ordered[1:]):   # a chain, not a clique: O(n) not O(n²)
                out.append((a, b, "shares_entity", 1.0))
        return out

    def features(self, cand_su_id: str, query_su_id: str) -> GraphFeatures:
        """The six bounded graph features that ρ carries for one (query, candidate) pair."""
        fid = self.su_fact.get(cand_su_id)
        shared = self.su_entities.get(cand_su_id, set()) & self.su_entities.get(query_su_id, set())
        hops = 0.0 if shared else 1.0     # 1/(1+hops): 1.0 = direct entity overlap
        return GraphFeatures(
            is_superseded=float(fid is not None and fid in self.superseded_by),
            supersedes_something=float(fid is not None and fid in self.supersedes),
            contradicted=float(fid is not None and (fid in self.superseded_by
                                                    or fid in self.supersedes)),
            shares_entity_with_query=float(bool(shared)),
            hops_to_query=1.0 / (1.0 + hops),
            degree=float(min(1.0, self.degree.get(cand_su_id, 0) / 8.0)),
        )

    def stale_su_ids(self) -> set[str]:
        """Units stating a fact that a later turn overwrote — Phase 7 must not serve these
        unlabelled."""
        return {self.fact_su[f] for f in self.superseded_by if f in self.fact_su}

    def stats(self) -> dict:
        return {
            "n_entities": len(self.entity_sus),
            "n_units_with_entity": len(self.su_entities),
            "n_supersedes": len(self.supersedes),
            "mean_entity_mentions": (
                sum(len(v) for v in self.entity_sus.values()) / max(len(self.entity_sus), 1)),
        }
