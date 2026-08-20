# Phase 00 · Foundations

## Objective
Make every later experiment reproducible by someone else, and make randomness a controlled variable.

## What runs here
- `src/qcs/config.py — every tunable number, plus the architecture's invariants as assertions`
- `src/qcs/seeding.py — hashlib-derived sub-streams (Python's hash() is salted per process)`
- `src/qcs/logging_.py — one run directory per experiment, schema-guarded metrics`

## Exit criterion
Two identical-seed runs produce identical output.

## Measured result
7 tests, including two that assert the validator rejects invalid configurations.

---

Full narrative for this phase: see `MASTER_DOC.md` and `deliverables/Ten_Phases_Dataset_Stability_Plasticity.docx`.
