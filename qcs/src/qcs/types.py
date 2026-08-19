"""The typed records that flow through the whole pipeline.

Everything is JSON-serialisable by construction so that any stage boundary can be
dumped to disk, inspected, and reloaded without a custom codec.
"""

from __future__ import annotations

from dataclasses import asdict, dataclass, field
from typing import Any, Literal

TurnKind = Literal["fact", "update", "filler", "distractor", "probe"]


@dataclass
class Fact:
    """An atomic, ownable piece of thread knowledge."""

    fact_id: str
    thread_id: str
    owner_user: str
    entity: str
    attribute: str
    value: str
    regime: str
    introduced_turn: int = -1
    supersedes: str | None = None       # fact_id this one replaces
    superseded_by: str | None = None    # filled in when a later fact replaces this

    def statement(self) -> str:
        return f"{self.entity} {self.attribute} {self.value}"

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


@dataclass
class Turn:
    """One utterance in a thread."""

    thread_id: str
    turn_idx: int
    user_id: str
    kind: TurnKind
    text: str
    ts: float
    fact_id: str | None = None            # for kind in {fact, update}
    required_facts: list[str] = field(default_factory=list)  # for kind == probe
    answer: str | None = None             # ground-truth answer for probes
    probe_distance: int | None = None     # turns since the required fact appeared

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


@dataclass
class SemanticUnit:
    """The atomic unit of memory: 1-3 sentences with full provenance."""

    su_id: str
    thread_id: str
    turn_idx: int
    su_idx: int
    user_id: str
    kind: TurnKind
    text: str
    ts: float
    fact_id: str | None = None

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


@dataclass
class Triple:
    """One training example for the relevance scorer."""

    thread_id: str
    probe_turn: int
    query_su_id: str
    cand_su_id: str
    user_id: str
    label: int                # 1 if the candidate is a required fact for this probe
    task_id: int = -1

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


@dataclass
class ThreadPlan:
    """The structured plan a thread is realised from. Ground truth lives here."""

    thread_id: str
    regime: str
    users: list[str]
    facts: list[Fact]
    turns: list[Turn]

    def to_dict(self) -> dict[str, Any]:
        return {
            "thread_id": self.thread_id,
            "regime": self.regime,
            "users": self.users,
            "facts": [f.to_dict() for f in self.facts],
            "turns": [t.to_dict() for t in self.turns],
        }
