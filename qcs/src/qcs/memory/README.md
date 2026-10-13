# `memory/` — the stability side (memory M)

Everything the thread has ever said is kept here, word for word. Nothing in this package is
trained, so training cannot overwrite it. If a fact is not used, the cause is a ranking mistake,
which can be measured and corrected, and not a lost fact.

| File | What it does | Logic worth knowing |
|---|---|---|
| `schema.sql` | The SQLite tables. | `semantic_units` (the verbatim store) has two triggers, `su_no_update` and `su_no_delete`, that abort any `UPDATE` or `DELETE`. Append-only is therefore enforced **by the database**, not by good behaviour in Python. Other tables are derived data: `facts` (ground truth for evaluation only), `summaries`, `edges`, `user_salience` (a running state, recomputable at any time) and `selection_log`. |
| `store.py` | `MemoryStore`: all reads and writes. | `write_sus` uses `INSERT OR IGNORE`, so writing the same unit twice cannot change it. `sus(thread, before_turn)` gives causal reads. `write_summary` also records `summarises` edges to the source units. |
| `ann_index.py` | `ANNIndex`: candidate retrieval by cosine similarity (FAISS inner-product index). | `search(query, k, mask=...)` restricts the search to earlier turns, so **a candidate from the future can never be returned**. This layer sets the ceiling of the whole system: a fact that is not retrieved cannot be selected. Search is exact on purpose, so "the scorer missed it" is never confused with "the index never showed it". |
| `graph.py` | `EntityGraph`: entity and supersession links per thread. | A unit mentions an entity when all its name tokens occur in the text. `supersedes` / `contradicts` edges join the old and new statement of a changed fact, and both survive. `features(candidate, query)` returns the six bounded graph features that fill part of the relational block (superseded, supersedes, contradicted, shares entity, hops, degree). `stale_su_ids()` lists units whose fact was later replaced. |
| `salience.py` | `SalienceTracker`: what each participant's own turns are about. | An exponential moving average (alpha 0.05) of the projected embeddings of that user's turns, folded to 64 numbers by block means (deterministic, so it cannot drift). `separation()` checks that different users really get different vectors. |
| `consolidate.py` | Adds summary nodes every 50 turns. | Summaries are **added beside** the sources and never replace them, so consolidation cannot destroy information. The summary is extractive (it lists the facts stated in the window), so it can never invent a fact. `verify_sources_intact` proves the sources were not touched. |

## Tests

`tests/test_memory.py`: `UPDATE` and `DELETE` are rejected by the database; rewriting a unit does not change it;
consolidation adds and never modifies, and is idempotent; search respects causality; the graph marks superseded facts;
graph features are bounded; salience separates users.
