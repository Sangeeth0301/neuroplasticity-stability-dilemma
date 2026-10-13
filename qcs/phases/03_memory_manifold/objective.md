# Phase 03 · Memory manifold ℳ

## Objective
Storage that cannot destroy information — the whole stability argument rests here.

## What runs here
- `scripts/pipeline/build_memory.py`
- `src/qcs/memory/ — schema.sql, store, ann_index, consolidate, graph, salience`

## Exit criterion
Append-only proven by test; prefilter recall ≥ 0.95 at M = 256.

## Measured result
recall 1.000 · 3 mutation attempts rejected by the database · cosine puts a stale fact first 28.6% of the time

## Evidence
figures/phases/V09–V12, V10b

---

Full narrative for this phase: see `MASTER_DOC.md` and `deliverables/Ten_Phases_Dataset_Stability_Plasticity.docx`.
