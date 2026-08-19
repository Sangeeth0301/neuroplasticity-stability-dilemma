-- The stability manifold ℳ.
--
-- The append-only property is enforced HERE, by triggers, not by convention in Python.
-- A store whose immutability depends on every caller behaving is not immutable — and the
-- entire stability argument of this architecture rests on it: knowledge that cannot be
-- overwritten cannot be catastrophically forgotten.

PRAGMA journal_mode = WAL;
PRAGMA foreign_keys = ON;

CREATE TABLE IF NOT EXISTS semantic_units (
    su_id       TEXT PRIMARY KEY,
    thread_id   TEXT NOT NULL,
    turn_idx    INTEGER NOT NULL,
    su_idx      INTEGER NOT NULL,
    user_id     TEXT NOT NULL,
    kind        TEXT NOT NULL,
    text        TEXT NOT NULL,
    ts          REAL NOT NULL,
    fact_id     TEXT,
    n_tokens    INTEGER NOT NULL,
    written_at  REAL NOT NULL
);
CREATE INDEX IF NOT EXISTS idx_su_thread_turn ON semantic_units(thread_id, turn_idx);
CREATE INDEX IF NOT EXISTS idx_su_fact ON semantic_units(fact_id);

-- The two triggers that make ℳ append-only.
CREATE TRIGGER IF NOT EXISTS su_no_update
BEFORE UPDATE ON semantic_units
BEGIN
    SELECT RAISE(ABORT, 'semantic_units is append-only: UPDATE is forbidden');
END;

CREATE TRIGGER IF NOT EXISTS su_no_delete
BEFORE DELETE ON semantic_units
BEGIN
    SELECT RAISE(ABORT, 'semantic_units is append-only: DELETE is forbidden');
END;

-- Facts as declared by the corpus (ground truth for evaluation only — the scorer never
-- sees fact_id at inference time).
CREATE TABLE IF NOT EXISTS facts (
    fact_id       TEXT PRIMARY KEY,
    thread_id     TEXT NOT NULL,
    owner_user    TEXT NOT NULL,
    entity        TEXT NOT NULL,
    attribute     TEXT NOT NULL,
    value         TEXT NOT NULL,
    regime        TEXT NOT NULL,
    introduced_turn INTEGER NOT NULL,
    supersedes    TEXT,
    superseded_by TEXT
);

-- Consolidation output. Summaries are ADDED; they never replace their sources, so
-- consolidation cannot destroy information.
CREATE TABLE IF NOT EXISTS summaries (
    summary_id  TEXT PRIMARY KEY,
    thread_id   TEXT NOT NULL,
    from_turn   INTEGER NOT NULL,
    to_turn     INTEGER NOT NULL,
    text        TEXT NOT NULL,
    n_tokens    INTEGER NOT NULL,
    n_sources   INTEGER NOT NULL,
    written_at  REAL NOT NULL
);

CREATE TABLE IF NOT EXISTS edges (
    src   TEXT NOT NULL,
    dst   TEXT NOT NULL,
    kind  TEXT NOT NULL,          -- summarises | supersedes | contradicts | shares_entity
    weight REAL NOT NULL DEFAULT 1.0,
    PRIMARY KEY (src, dst, kind)
);
CREATE INDEX IF NOT EXISTS idx_edges_src ON edges(src, kind);
CREATE INDEX IF NOT EXISTS idx_edges_dst ON edges(dst, kind);

-- Per-participant salience vectors (the only mutable table: an EMA is a running state,
-- not knowledge, and it is recomputable from the append-only units at any time).
CREATE TABLE IF NOT EXISTS user_salience (
    thread_id TEXT NOT NULL,
    user_id   TEXT NOT NULL,
    vec       BLOB NOT NULL,
    n_updates INTEGER NOT NULL,
    PRIMARY KEY (thread_id, user_id)
);

-- What the assembler chose, for the usage features in ρ and for prompt autopsy.
CREATE TABLE IF NOT EXISTS selection_log (
    run_id     TEXT NOT NULL,
    thread_id  TEXT NOT NULL,
    probe_turn INTEGER NOT NULL,
    su_id      TEXT NOT NULL,
    score      REAL NOT NULL,
    selected   INTEGER NOT NULL,
    rank       INTEGER NOT NULL
);
CREATE INDEX IF NOT EXISTS idx_sel_su ON selection_log(su_id);
