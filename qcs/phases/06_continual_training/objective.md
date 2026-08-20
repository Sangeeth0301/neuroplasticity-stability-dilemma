# Phase 06 · Continual training

## Objective
The experiment: a long task sequence with both properties instrumented at every step.

## What runs here
- `scripts/train_continual.py`
- `src/qcs/train/ — replay, anchor, loop`

## Exit criterion
Full metric suite logged for both arms across the task stream.

## Measured result
120 tasks × 3 arms · BWT positive for all · Tr(F) 4.75× quantum vs 0.28× and 0.06× classical

## Evidence
figures/phases/V21–V34 · figures/results/RES3, RES5

---

Full narrative for this phase: see `MASTER_DOC.md` and `deliverables/Ten_Phases_Dataset_Stability_Plasticity.docx`.
