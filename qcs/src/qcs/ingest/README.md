# `ingest/` — from raw turns to semantic units

| File | What it does | Logic worth knowing |
|---|---|---|
| `loader.py` | Reads the JSONL corpus into typed objects: `load_turns`, `load_facts`, `load_sus`, `group_by_thread`. | `load_sus` returns segmented units, sorted by `(turn_idx, su_idx)` inside each thread. |
| `segment.py` | `segment_turn` cuts one turn into **semantic units** (SUs). | Sentences are packed greedily until adding the next one would pass the target of 40 words. The process is **lossless**: joining a turn's units reproduces the normalised turn text (`test_segment_roundtrip`). The only text change allowed anywhere in the project is collapsing whitespace (`normalize`). Unit ids look like `train-000:00042:0` (thread, zero-padded turn, unit index). |

A semantic unit is the atom of memory. It carries who said it, when, which turn it came from and,
for evaluation only, which fact it states.
