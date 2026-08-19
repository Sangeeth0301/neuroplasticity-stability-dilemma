"""The fact bank: atomic, ownable, probe-able pieces of thread knowledge.

Facts are (entity, attribute, value) triples with an owner. Everything downstream —
probe questions, ground-truth answers, distractors, supersession chains — is derived
from this structure, which is what makes retrieval ground truth exact rather than
approximate.
"""

from __future__ import annotations

import numpy as np

from ..types import Fact

USERS = ["Ada", "Bo", "Chen", "Dara", "Emil", "Fay", "Gus", "Hana"]

_ADJ = [
    "Helios", "Kestrel", "Onyx", "Vireo", "Meridian", "Quill", "Lumen", "Basalt",
    "Cobalt", "Juniper", "Nimbus", "Orchid", "Pallas", "Reed", "Sable", "Tessera",
]
_NOUN = [
    "dashboard", "migration", "retro", "rollout", "audit", "pipeline", "handbook",
    "prototype", "workshop", "review", "integration", "benchmark", "onboarding",
]

# Each attribute carries: a value generator, statement templates, and probe templates.
# {e} entity, {v} value, {u} owner.
ATTRIBUTES: dict[str, dict] = {
    "deadline": {
        "values": lambda r: f"{r.choice(['March', 'April', 'June', 'September', 'November'])} "
                            f"{int(r.integers(3, 28))}",
        "say": [
            "The {e} is due on {v}.",
            "Let's lock the {e} deadline to {v}.",
            "I've put {v} on the calendar as the {e} deadline.",
        ],
        "ask": [
            "When is the {e} due again?",
            "Remind me — what deadline did we set for the {e}?",
            "What's the {e} deadline?",
        ],
    },
    "owner": {
        "values": lambda r: str(r.choice(USERS)),
        "say": [
            "{v} is taking point on the {e}.",
            "We agreed {v} owns the {e}.",
            "The {e} is {v}'s to drive.",
        ],
        "ask": [
            "Who owns the {e}?",
            "Who's driving the {e} these days?",
            "Who did we put in charge of the {e}?",
        ],
    },
    "budget": {
        "values": lambda r: f"${int(r.integers(3, 90)) * 500:,}",
        "say": [
            "The {e} budget is {v}.",
            "We have {v} allocated for the {e}.",
            "Finance approved {v} for the {e}.",
        ],
        "ask": [
            "How much is budgeted for the {e}?",
            "What's the {e} budget?",
            "Remind me what we allocated for the {e}?",
        ],
    },
    "status": {
        "values": lambda r: str(r.choice([
            "blocked on review", "in staging", "shipped", "paused until Q3",
            "waiting on legal", "half migrated", "in a second round of testing",
        ])),
        "say": [
            "The {e} is {v} right now.",
            "Status update: the {e} is {v}.",
            "As of today the {e} is {v}.",
        ],
        "ask": [
            "Where did the {e} land?",
            "What's the current status of the {e}?",
            "Is the {e} still moving?",
        ],
    },
    "location": {
        "values": lambda r: str(r.choice([
            "Room 4B", "the Lisbon office", "the annex", "the third-floor lab",
            "the Porto hub", "the old design studio",
        ])),
        "say": [
            "The {e} session runs in {v}.",
            "We're holding the {e} in {v}.",
            "{v} is booked for the {e}.",
        ],
        "ask": [
            "Where is the {e} happening?",
            "Which room did we book for the {e}?",
            "Where are we doing the {e}?",
        ],
    },
    "tool": {
        "values": lambda r: str(r.choice([
            "Postgres 16", "the Rust client", "Terraform", "the legacy Java worker",
            "DuckDB", "an internal Go service", "Kafka",
        ])),
        "say": [
            "The {e} runs on {v}.",
            "We standardised the {e} on {v}.",
            "{v} is what the {e} is built against.",
        ],
        "ask": [
            "What is the {e} built on?",
            "Which stack did we pick for the {e}?",
            "What does the {e} run on?",
        ],
    },
    "preference": {
        "values": lambda r: str(r.choice([
            "no meetings before 10", "written updates over calls", "async review only",
            "one reviewer per PR", "demos on Thursdays", "no Friday deploys",
        ])),
        "say": [
            "For the {e}, I want {v}.",
            "My rule for the {e}: {v}.",
            "Please keep {v} for the {e}.",
        ],
        "ask": [
            "What did {u} want for the {e}?",
            "What was {u}'s preference on the {e}?",
            "How does {u} want the {e} handled?",
        ],
    },
}

ATTR_NAMES = list(ATTRIBUTES)

UPDATE_TEMPLATES = [
    "Correction on the {e}: it's {v} now, not {old}.",
    "Update — the {e} changed to {v} (was {old}).",
    "Scratch that, the {e} is {v} these days.",
]


def make_entities(rng: np.random.Generator, n: int) -> list[str]:
    """n distinct two-word entity names."""
    seen: set[str] = set()
    out: list[str] = []
    while len(out) < n:
        name = f"{rng.choice(_ADJ)} {rng.choice(_NOUN)}"
        if name not in seen:
            seen.add(name)
            out.append(name)
    return out


def build_fact_bank(
    thread_id: str,
    users: list[str],
    n_facts: int,
    regime: str,
    rng: np.random.Generator,
) -> list[Fact]:
    """Build the fact bank for one thread.

    `single_owner`  — every fact belongs to one participant.
    `multi_owner`   — facts spread across participants (tests persona attribution).
    `adversarial`   — as multi_owner, plus supersession chains added by the planner.
    """
    # Several facts per entity, each with a DIFFERENT attribute. One fact per entity
    # would make the entity name a unique key and reduce retrieval to string matching —
    # measured in Phase 3, where cosine top-8 recall hit 1.000 and left nothing for any
    # scorer to improve. Sharing entities forces the retriever to resolve the attribute
    # too, and makes distractors genuinely confusable.
    facts_per_entity = 3
    entities = make_entities(rng, max(1, n_facts // facts_per_entity))
    used_attrs: dict[str, set[str]] = {e: set() for e in entities}
    primary = str(rng.choice(users))
    facts: list[Fact] = []
    for i in range(n_facts):
        entity = entities[i % len(entities)]
        free = [a for a in ATTR_NAMES if a not in used_attrs[entity]]
        if not free:
            continue
        attribute = str(rng.choice(free))
        used_attrs[entity].add(attribute)
        value = ATTRIBUTES[attribute]["values"](rng)
        owner = primary if regime == "single_owner" else str(rng.choice(users))
        if attribute == "owner":
            # keep the value inside the participant set so probes stay answerable
            value = str(rng.choice(users))
        facts.append(
            Fact(
                fact_id=f"{thread_id}:f{i:03d}",
                thread_id=thread_id,
                owner_user=owner,
                entity=entity,
                attribute=attribute,
                value=value,
                regime=regime,
            )
        )
    return facts


def say_text(fact: Fact, rng: np.random.Generator) -> str:
    tmpl = str(rng.choice(ATTRIBUTES[fact.attribute]["say"]))
    return tmpl.format(e=fact.entity, v=fact.value, u=fact.owner_user)


def ask_text(fact: Fact, rng: np.random.Generator) -> str:
    tmpl = str(rng.choice(ATTRIBUTES[fact.attribute]["ask"]))
    return tmpl.format(e=fact.entity, v=fact.value, u=fact.owner_user)


def update_text(new_fact: Fact, old_value: str, rng: np.random.Generator) -> str:
    tmpl = str(rng.choice(UPDATE_TEMPLATES))
    return tmpl.format(e=new_fact.entity, v=new_fact.value, old=old_value)
