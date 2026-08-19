"""Thread planning: decide WHAT happens on every turn before any prose exists.

The plan is the ground truth. Surface text is generated from the plan afterwards and
is never allowed to add or remove information, so "which memories were required to
answer this probe" is known exactly rather than inferred.

Turn kinds
    fact       a participant states a new fact
    update     a participant supersedes an earlier fact (adversarial regime)
    probe      a participant asks a question answerable only from stated facts
    distractor filler that mentions a real entity with a wrong/irrelevant value
    filler     ordinary conversation
"""

from __future__ import annotations

import numpy as np

from ..types import Fact, ThreadPlan, Turn
from . import fact_bank as fb
from .surface_realizer import realize_distractor, realize_filler

TS_STEP = 137.0  # seconds between turns, deterministic


def plan_thread(
    thread_id: str,
    regime: str,
    n_turns: int,
    n_users: int,
    n_facts: int,
    probe_distances: list[int],
    supersede_ratio: float,
    distractor_ratio: float,
    rng: np.random.Generator,
) -> ThreadPlan:
    users = fb.USERS[:n_users]
    facts = fb.build_fact_bank(thread_id, users, n_facts, regime, rng)

    max_d = max(probe_distances)
    # Assign the probe DISTANCE first (round-robin over the bands, so every band is
    # equally represented), then place the fact anywhere that still leaves room for
    # its probe. Placing facts first and picking distances afterwards crushes every
    # fact into the opening turns — caught by figure V01 on the first run.
    bands = list(rng.permutation(np.array(probe_distances)))
    fact_band = {f.fact_id: int(bands[i % len(bands)]) for i, f in enumerate(facts)}
    schedule: dict[int, Turn] = {}
    extra_facts: list[Fact] = []
    placed: list[Fact] = []

    for fact in facts:
        # Each fact is placed uniformly in the window that still leaves room for its
        # own probe. A long-distance probe therefore forces an early fact — that is
        # inherent to the task, not a scheduling artefact.
        latest = n_turns - fact_band[fact.fact_id] - 2
        slot = _free_slot(int(rng.integers(1, max(2, latest))), latest, schedule)
        if slot is None:
            continue
        fact.introduced_turn = slot
        placed.append(fact)
        schedule[slot] = Turn(
            thread_id=thread_id, turn_idx=slot, user_id=fact.owner_user, kind="fact",
            text="", ts=0.0, fact_id=fact.fact_id,
        )
    facts = placed

    # --- supersession chains (adversarial regime) ------------------------------
    if regime == "adversarial":
        n_sup = int(round(supersede_ratio * n_facts))
        for fact in rng.permutation(np.array(facts, dtype=object))[:n_sup]:
            gap = int(rng.integers(15, 120))
            band = fact_band[fact.fact_id]
            u_slot = _free_slot(fact.introduced_turn + gap, n_turns - band - 2, schedule)
            if u_slot is None:
                continue
            new_value = fb.ATTRIBUTES[fact.attribute]["values"](rng)
            if new_value == fact.value:
                continue
            new_fact = Fact(
                fact_id=fact.fact_id + "u", thread_id=thread_id, owner_user=fact.owner_user,
                entity=fact.entity, attribute=fact.attribute, value=new_value,
                regime=regime, introduced_turn=u_slot, supersedes=fact.fact_id,
            )
            fact.superseded_by = new_fact.fact_id
            # the replacement inherits a band that still fits after its own turn
            feasible = [d for d in probe_distances if u_slot + d < n_turns - 1]
            fact_band[new_fact.fact_id] = int(rng.choice(feasible)) if feasible else 10
            extra_facts.append(new_fact)
            schedule[u_slot] = Turn(
                thread_id=thread_id, turn_idx=u_slot, user_id=fact.owner_user, kind="update",
                text="", ts=0.0, fact_id=new_fact.fact_id,
            )

    all_facts = facts + extra_facts

    # --- probes ---------------------------------------------------------------
    for fact in all_facts:
        if fact.superseded_by is not None:
            continue  # never probe a fact that a later turn overwrites
        d = fact_band[fact.fact_id]
        target = fact.introduced_turn + d
        if target >= n_turns:
            feasible = [x for x in probe_distances if fact.introduced_turn + x < n_turns]
            if not feasible:
                continue
            target = fact.introduced_turn + int(min(feasible))
        slot = _free_slot(target, n_turns, schedule)
        if slot is None:
            continue
        asker = _pick_asker(regime, fact, users, rng)
        schedule[slot] = Turn(
            thread_id=thread_id, turn_idx=slot, user_id=asker, kind="probe", text="", ts=0.0,
            required_facts=[fact.fact_id], answer=fact.value,
            probe_distance=slot - fact.introduced_turn,
        )

    # --- filler and distractors ----------------------------------------------
    for idx in range(n_turns):
        if idx in schedule:
            continue
        is_distractor = rng.random() < distractor_ratio
        kind = "distractor" if is_distractor else "filler"
        schedule[idx] = Turn(
            thread_id=thread_id, turn_idx=idx, user_id=str(rng.choice(users)),
            kind=kind, text="", ts=0.0,
        )

    turns = [schedule[i] for i in range(n_turns)]
    fact_by_id = {f.fact_id: f for f in all_facts}
    for turn in turns:
        turn.ts = turn.turn_idx * TS_STEP
        if turn.kind in ("fact", "update"):
            f = fact_by_id[turn.fact_id]
            if turn.kind == "fact":
                turn.text = fb.say_text(f, rng)
            else:
                old = fact_by_id[f.supersedes].value
                turn.text = fb.update_text(f, old, rng)
        elif turn.kind == "probe":
            turn.text = fb.ask_text(fact_by_id[turn.required_facts[0]], rng)
        elif turn.kind == "distractor":
            turn.text = realize_distractor(all_facts, rng)
        else:
            turn.text = realize_filler(rng)

    return ThreadPlan(thread_id=thread_id, regime=regime, users=users,
                      facts=all_facts, turns=turns)


# --------------------------------------------------------------------------- #
def _free_slot(target: int, n_turns: int, schedule: dict[int, Turn]) -> int | None:
    """Nearest free turn index at or after `target`, then searching backwards."""
    if target >= n_turns:
        return None
    for offset in range(0, min(40, n_turns - target)):
        if target + offset not in schedule:
            return target + offset
    for offset in range(1, 40):
        if target - offset > 0 and target - offset not in schedule:
            return target - offset
    return None


def _pick_asker(regime: str, fact: Fact, users: list[str], rng: np.random.Generator) -> str:
    if regime == "single_owner":
        return str(rng.choice(users))
    # cross-user questioning is the point of the multi-owner regime
    others = [u for u in users if u != fact.owner_user]
    return str(rng.choice(others)) if others else fact.owner_user
