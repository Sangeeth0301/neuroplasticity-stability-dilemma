"""Consolidation: add an index level, never replace one.

Every K turns a summary node is written that points at its source units. The sources
remain untouched and retrievable — consolidation adds a coarser view of the thread, it
does not compress the thread. That is the difference between this and rolling
summarisation (baseline B2), which discards what it summarises and is exactly how
production chat systems lose old facts.
"""

from __future__ import annotations

from ..types import Fact, SemanticUnit
from .store import MemoryStore


def summarise_window(sus: list[SemanticUnit], facts: dict[str, Fact]) -> str:
    """Deterministic extractive summary: the decisions in the window, with owners.

    An LLM summariser drops in here later; keeping it extractive for now means the
    summary can never introduce a fact that was not stated, which keeps ground truth
    intact.
    """
    lines: list[str] = []
    for su in sus:
        if su.kind in ("fact", "update") and su.fact_id and su.fact_id in facts:
            f = facts[su.fact_id]
            verb = "updated" if su.kind == "update" else "stated"
            lines.append(f"{f.owner_user} {verb} that the {f.entity} {f.attribute} is {f.value}")
    if not lines:
        speakers = sorted({s.user_id for s in sus})
        return f"General discussion between {', '.join(speakers)}; no decisions recorded."
    return "; ".join(lines) + "."


def consolidate_thread(
    store: MemoryStore, thread_id: str, every: int, up_to_turn: int | None = None
) -> list[str]:
    """Write one summary per window of `every` turns. Returns the new summary ids."""
    sus = store.sus(thread_id=thread_id)
    if not sus:
        return []
    facts = store.facts(thread_id)
    last = up_to_turn if up_to_turn is not None else max(s.turn_idx for s in sus)
    existing = {(s["from_turn"], s["to_turn"]) for s in store.summaries(thread_id)}

    new_ids: list[str] = []
    for start in range(0, last + 1, every):
        end = start + every - 1
        if (start, end) in existing or end > last:
            continue
        window = [s for s in sus if start <= s.turn_idx <= end]
        if not window:
            continue
        sid = f"{thread_id}:sum{start:05d}"
        store.write_summary(sid, thread_id, start, end,
                            summarise_window(window, facts), [s.su_id for s in window])
        new_ids.append(sid)
    return new_ids


def verify_sources_intact(store: MemoryStore, thread_id: str,
                          before: dict[str, str]) -> bool:
    """Consolidation must not have touched a single source unit.

    `before` is {su_id: text} captured prior to consolidating. Called by the Phase 3
    test and by figure V11.
    """
    after = {s.su_id: s.text for s in store.sus(thread_id=thread_id)}
    return all(after.get(k) == v for k, v in before.items()) and len(after) >= len(before)
