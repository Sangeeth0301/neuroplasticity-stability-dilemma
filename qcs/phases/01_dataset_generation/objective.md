# Phase 01 · Dataset generation

## Objective
Build a corpus where the answer to 'which memories were required?' is exact, not inferred.

## What runs here
- `scripts/pipeline/gen_dataset.py`
- `src/qcs/datagen/ — fact bank, thread planner, surface realiser, validator`

## Exit criterion
Oracle retriever scores 1.000 and a random one scores near chance.

## Measured result
oracle 1.000 · random 0.042 · 18,000 turns · 1,800 probes · 1,920 facts

## Evidence
figures/phases/V01–V04

---

Full narrative for this phase: see `MASTER_DOC.md` and `deliverables/Ten_Phases_Dataset_Stability_Plasticity.docx`.
