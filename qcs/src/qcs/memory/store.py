"""MemoryStore — the append-only stability manifold ℳ.

Everything the thread has ever said lives here verbatim, with provenance. Nothing is
learned, so nothing can be unlearned: the failure mode degrades from "the fact is gone"
to "the fact was not retrieved", which is measurable and fixable.
"""

from __future__ import annotations

import sqlite3
import time
from pathlib import Path

import numpy as np

from .. import paths
from ..types import Fact, SemanticUnit

SCHEMA = Path(__file__).with_name("schema.sql")


def n_tokens(text: str) -> int:
    """Word-count proxy. Phase 7 replaces this with the target model's tokenizer."""
    return len(text.split())


class MemoryStore:
    def __init__(self, db_path: str | Path | None = None):
        self.path = Path(db_path or paths.DATA_PROCESSED / "qcs.db")
        self.path.parent.mkdir(parents=True, exist_ok=True)
        self.con = sqlite3.connect(self.path)
        self.con.row_factory = sqlite3.Row
        self.con.executescript(SCHEMA.read_text(encoding="utf-8"))
        self.con.commit()

    # ------------------------------------------------------------------ writes
    def write_sus(self, sus: list[SemanticUnit]) -> int:
        now = time.time()
        rows = [(s.su_id, s.thread_id, s.turn_idx, s.su_idx, s.user_id, s.kind, s.text,
                 s.ts, s.fact_id, n_tokens(s.text), now) for s in sus]
        self.con.executemany(
            "INSERT OR IGNORE INTO semantic_units VALUES (?,?,?,?,?,?,?,?,?,?,?)", rows)
        self.con.commit()
        return len(rows)

    def write_facts(self, facts: dict[str, Fact] | list[Fact]) -> int:
        items = list(facts.values()) if isinstance(facts, dict) else facts
        rows = [(f.fact_id, f.thread_id, f.owner_user, f.entity, f.attribute, f.value,
                 f.regime, f.introduced_turn, f.supersedes, f.superseded_by) for f in items]
        self.con.executemany("INSERT OR REPLACE INTO facts VALUES (?,?,?,?,?,?,?,?,?,?)", rows)
        self.con.commit()
        return len(rows)

    def write_summary(self, summary_id: str, thread_id: str, from_turn: int, to_turn: int,
                      text: str, source_ids: list[str]) -> None:
        self.con.execute(
            "INSERT OR REPLACE INTO summaries VALUES (?,?,?,?,?,?,?,?)",
            (summary_id, thread_id, from_turn, to_turn, text, n_tokens(text),
             len(source_ids), time.time()))
        self.con.executemany(
            "INSERT OR REPLACE INTO edges (src, dst, kind, weight) VALUES (?,?,'summarises',1.0)",
            [(summary_id, sid) for sid in source_ids])
        self.con.commit()

    def write_edges(self, edges: list[tuple[str, str, str, float]]) -> int:
        self.con.executemany("INSERT OR REPLACE INTO edges VALUES (?,?,?,?)", edges)
        self.con.commit()
        return len(edges)

    def log_selection(self, run_id: str, thread_id: str, probe_turn: int,
                      rows: list[tuple[str, float, int, int]]) -> None:
        self.con.executemany(
            "INSERT INTO selection_log VALUES (?,?,?,?,?,?,?)",
            [(run_id, thread_id, probe_turn, su_id, score, sel, rank)
             for su_id, score, sel, rank in rows])
        self.con.commit()

    # ------------------------------------------------------------------ reads
    def sus(self, thread_id: str | None = None, before_turn: int | None = None
            ) -> list[SemanticUnit]:
        q = "SELECT * FROM semantic_units WHERE 1=1"
        args: list = []
        if thread_id is not None:
            q += " AND thread_id = ?"
            args.append(thread_id)
        if before_turn is not None:
            q += " AND turn_idx < ?"
            args.append(before_turn)
        q += " ORDER BY thread_id, turn_idx, su_idx"
        return [self._su(r) for r in self.con.execute(q, args)]

    def su(self, su_id: str) -> SemanticUnit | None:
        r = self.con.execute("SELECT * FROM semantic_units WHERE su_id = ?", (su_id,)).fetchone()
        return self._su(r) if r else None

    def facts(self, thread_id: str | None = None) -> dict[str, Fact]:
        q = "SELECT * FROM facts" + (" WHERE thread_id = ?" if thread_id else "")
        rows = self.con.execute(q, (thread_id,) if thread_id else ())
        out = {}
        for r in rows:
            d = dict(r)
            out[d["fact_id"]] = Fact(**d)
        return out

    def summaries(self, thread_id: str, before_turn: int | None = None) -> list[dict]:
        q = "SELECT * FROM summaries WHERE thread_id = ?"
        args: list = [thread_id]
        if before_turn is not None:
            q += " AND to_turn <= ?"
            args.append(before_turn)
        return [dict(r) for r in self.con.execute(q, args)]

    def edges(self, kind: str | None = None) -> list[tuple[str, str, str, float]]:
        q = "SELECT src, dst, kind, weight FROM edges" + (" WHERE kind = ?" if kind else "")
        return [tuple(r) for r in self.con.execute(q, (kind,) if kind else ())]

    def selection_counts(self) -> dict[str, tuple[int, int]]:
        """su_id → (times_selected, times_seen) for the ρ usage features."""
        rows = self.con.execute(
            "SELECT su_id, SUM(selected), COUNT(*) FROM selection_log GROUP BY su_id")
        return {r[0]: (int(r[1]), int(r[2])) for r in rows}

    def counts(self) -> dict[str, int]:
        c = self.con.cursor()
        return {
            t: c.execute(f"SELECT COUNT(*) FROM {t}").fetchone()[0]
            for t in ("semantic_units", "facts", "summaries", "edges", "selection_log")
        }

    # ------------------------------------------------------------------ salience
    def get_salience(self, thread_id: str, user_id: str, dim: int) -> np.ndarray:
        r = self.con.execute(
            "SELECT vec FROM user_salience WHERE thread_id = ? AND user_id = ?",
            (thread_id, user_id)).fetchone()
        return np.frombuffer(r[0], dtype=np.float32).copy() if r else np.zeros(dim, np.float32)

    def set_salience(self, thread_id: str, user_id: str, vec: np.ndarray, n: int) -> None:
        self.con.execute(
            "INSERT OR REPLACE INTO user_salience VALUES (?,?,?,?)",
            (thread_id, user_id, np.asarray(vec, np.float32).tobytes(), n))
        self.con.commit()

    # ------------------------------------------------------------------ internals
    @staticmethod
    def _su(r: sqlite3.Row) -> SemanticUnit:
        return SemanticUnit(
            su_id=r["su_id"], thread_id=r["thread_id"], turn_idx=r["turn_idx"],
            su_idx=r["su_idx"], user_id=r["user_id"], kind=r["kind"], text=r["text"],
            ts=r["ts"], fact_id=r["fact_id"],
        )

    def close(self) -> None:
        self.con.close()
