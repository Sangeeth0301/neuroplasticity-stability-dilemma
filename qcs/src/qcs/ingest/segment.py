"""Turn → semantic units.

A semantic unit (SU) is the atomic unit of memory: 1-3 sentences, ~20-60 tokens,
carrying full provenance. Segmentation is deterministic and lossless — concatenating
a turn's SUs reproduces the normalised turn text exactly, which
`tests/test_encode_shapes.py::test_segment_roundtrip` asserts. A lossy segmenter would
silently drop facts and the loss would surface much later as an unexplainable
retrieval failure.
"""

from __future__ import annotations

import re

from ..types import SemanticUnit, Turn

_SENT_SPLIT = re.compile(r"(?<=[.!?])\s+")


def normalize(text: str) -> str:
    """Collapse whitespace. The only text mutation allowed anywhere in the pipeline."""
    return re.sub(r"\s+", " ", text).strip()


def split_sentences(text: str) -> list[str]:
    return [s for s in _SENT_SPLIT.split(normalize(text)) if s]


def segment_turn(turn: Turn, target_tokens: int = 40) -> list[SemanticUnit]:
    """Merge sentences greedily up to `target_tokens` words per unit."""
    sents = split_sentences(turn.text)
    if not sents:
        return []

    groups: list[list[str]] = [[]]
    count = 0
    for s in sents:
        n = len(s.split())
        if groups[-1] and count + n > target_tokens:
            groups.append([])
            count = 0
        groups[-1].append(s)
        count += n

    return [
        SemanticUnit(
            su_id=f"{turn.thread_id}:{turn.turn_idx:05d}:{i}",
            thread_id=turn.thread_id,
            turn_idx=turn.turn_idx,
            su_idx=i,
            user_id=turn.user_id,
            kind=turn.kind,
            text=" ".join(g),
            ts=turn.ts,
            fact_id=turn.fact_id,
        )
        for i, g in enumerate(groups)
        if g
    ]


def segment_turns(turns: list[Turn], target_tokens: int = 40) -> list[SemanticUnit]:
    out: list[SemanticUnit] = []
    for t in turns:
        out.extend(segment_turn(t, target_tokens))
    return out
