"""Canonical filesystem locations. Every other module asks this one for paths."""

from __future__ import annotations

from pathlib import Path

# src/qcs/paths.py -> src/qcs -> src -> repo root
ROOT = Path(__file__).resolve().parents[2]

CONFIGS = ROOT / "configs"
DATA = ROOT / "data"
DATA_RAW = DATA / "raw"
DATA_INTERIM = DATA / "interim"
DATA_PROCESSED = DATA / "processed"
RUNS = ROOT / "runs"
FIGS = ROOT / "figs"
DOCS = ROOT / "docs"


def ensure_dirs() -> None:
    """Create the directory skeleton. Safe to call repeatedly."""
    for p in (DATA_RAW, DATA_INTERIM, DATA_PROCESSED, RUNS, FIGS, DOCS):
        p.mkdir(parents=True, exist_ok=True)
