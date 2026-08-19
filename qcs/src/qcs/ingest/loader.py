"""Corpus loading: JSONL → typed records → semantic units."""

from __future__ import annotations

from pathlib import Path

from .. import paths
from ..logging_ import read_jsonl
from ..types import Fact, SemanticUnit, Turn
from .segment import segment_turns


def load_turns(split: str, raw_dir: Path | None = None) -> list[Turn]:
    raw_dir = Path(raw_dir or paths.DATA_RAW)
    rows = read_jsonl(raw_dir / f"threads_{split}.jsonl")
    return [Turn(**r) for r in rows]


def load_facts(raw_dir: Path | None = None) -> dict[str, Fact]:
    raw_dir = Path(raw_dir or paths.DATA_RAW)
    return {r["fact_id"]: Fact(**r) for r in read_jsonl(raw_dir / "fact_bank.jsonl")}


def load_sus(split: str, target_tokens: int = 40, raw_dir: Path | None = None) -> list[SemanticUnit]:
    return segment_turns(load_turns(split, raw_dir), target_tokens)


def group_by_thread(items) -> dict[str, list]:
    out: dict[str, list] = {}
    for it in items:
        out.setdefault(it.thread_id, []).append(it)
    for v in out.values():
        v.sort(key=lambda x: (x.turn_idx, getattr(x, "su_idx", 0)))
    return out
