# Phase 02 · Encoding

## Objective
A deterministic, frozen map from text to a unit-norm 1024-dimensional vector.

## What runs here
- `scripts/pipeline/fit_encoder.py`
- `scripts/pipeline/study_whitening.py`
- `src/qcs/encode/`

## Exit criterion
Every triple yields a unit-norm 1024-vector; held-out retrieval retained.

## Measured result
held-out recall@20: 0.879 raw → 0.887 whitened → 0.996 projected · participation ratio 189/1024

## Evidence
figures/phases/V05–V08, V13

---

Full narrative for this phase: see `MASTER_DOC.md` and `deliverables/Ten_Phases_Dataset_Stability_Plasticity.docx`.
