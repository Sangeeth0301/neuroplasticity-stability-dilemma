# Phase 07 · Assembly and generation

## Objective
Turn scores into the best prompt under a hard token ceiling.

## What runs here
- `scripts/pipeline/run_assembly.py`
- `src/qcs/assemble/`

## Exit criterion
The budget is never exceeded and no stale fact is served unlabelled.

## Measured result
0 unlabelled stale facts · learned arms reach their ceiling at B=128 where baselines need B=512

## Evidence
figures/phases/V41–V45

---

Full narrative for this phase: see `MASTER_DOC.md` and `deliverables/Ten_Phases_Dataset_Stability_Plasticity.docx`.
