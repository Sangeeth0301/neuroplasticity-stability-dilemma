"""Surface text for non-fact turns.

Deterministic templates rather than an LLM: the corpus must be byte-reproducible from
a seed, and an LLM writing prose freely is exactly how ground truth leaks. (An LLM
realizer is a drop-in replacement for these functions once the harness is validated —
it must still be given the plan and forbidden from inventing facts.)

Distractors matter more than filler. They mention a REAL entity with a wrong or
irrelevant value, so they are lexically close to the right answer and cannot be
filtered by keyword matching. They are what makes retrieval non-trivial.
"""

from __future__ import annotations

import numpy as np

from ..types import Fact

FILLER = [
    "Morning all — coffee machine is broken again.",
    "I'll be offline for an hour after lunch.",
    "Nice work on the demo yesterday, that landed well.",
    "Does anyone have the link to the old notes?",
    "Quick reminder that the office is closed Monday.",
    "I rewrote the intro paragraph, much shorter now.",
    "Let's keep this thread for decisions only, please.",
    "Ha, that's exactly what happened last time.",
    "Adding one more item to the agenda for later.",
    "I think we're overcomplicating this.",
    "Can someone double-check my numbers before I send it?",
    "Agreed, let's move on.",
    "The train was late so I missed the start.",
    "I'll write it up properly this afternoon.",
    "Sounds good to me.",
    "Sorry, catching up on the backlog here.",
    "That screenshot is unreadable on mobile.",
    "We tried that two quarters ago and it stalled.",
    "Good catch, thanks.",
    "I'd rather decide this synchronously.",
]

DISTRACTOR = [
    "Someone asked about the {e} earlier — I don't have the details in front of me.",
    "The {e} came up in another thread, but nothing was decided.",
    "I keep confusing the {e} with the other one from last quarter.",
    "Is the {e} the one we shelved? I genuinely can't remember.",
    "There's an old doc about the {e} somewhere, probably out of date.",
    "We should revisit the {e} at some point, not urgent.",
    "The {e} was mentioned in passing but no one owned the follow-up.",
    "Fun fact, the {e} nearly got renamed twice.",
]


def realize_filler(rng: np.random.Generator) -> str:
    return str(rng.choice(FILLER))


def realize_distractor(facts: list[Fact], rng: np.random.Generator) -> str:
    """Lexically-close noise: real entity, no usable value."""
    fact = facts[int(rng.integers(0, len(facts)))]
    return str(rng.choice(DISTRACTOR)).format(e=fact.entity)
