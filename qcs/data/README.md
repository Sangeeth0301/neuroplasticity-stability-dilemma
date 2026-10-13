# `data/` — the corpus and the frozen artefacts

| Path | Tracked in git? | What it is |
|---|---|---|
| `raw/fact_bank.jsonl` | yes | Every fact, with owner, entity, attribute, value and its supersession links. |
| `raw/threads_train.jsonl` | yes | The 24 training threads, one turn per line, including each probe's required facts and distance. |
| `raw/threads_test.jsonl` | yes | The 6 **sealed** test threads. Used only for the final numbers; nothing is tuned on it. |
| `processed/whiten.npz`, `projection.npz` | yes | The frozen encoding stack (whitening matrix and learned projection). |
| `processed/qcs.db` | yes | The append-only memory database (SQLite). |
| `processed/scoring_train.npz`, `scoring_test.npz` | no (large, regenerable) | The cached feature rows: for every probe, 64 candidates × 1,024 numbers, with labels and stale flags. |
| `interim/` | no | The sentence-embedding cache, rebuilt automatically. |

The raw corpus is deterministic: `python scripts/pipeline/gen_dataset.py` regenerates it exactly from the seed in `configs/base.yaml`.

To rebuild the cached feature files after a fresh clone:

```
python scripts/pipeline/build_scoring_cache.py
```

This needs the sentence-encoder weights (`sentence-transformers/all-MiniLM-L6-v2`, downloaded on first use). The rebuilt files are byte-identical to the ones used for every reported result.
