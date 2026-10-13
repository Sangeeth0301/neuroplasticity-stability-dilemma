# `scripts/` — command-line entry points

Every script is a thin wrapper: it parses arguments, calls code in `src/qcs/`, and writes results. The real logic
lives in the package. **Run every script from the `qcs/` folder** (some write to relative paths such as `figures/`).

| Folder | What is in it | When you use it |
|---|---|---|
| [`pipeline/`](pipeline/README.md) | The build, phase by phase: dataset, encoder, memory, quantum core, learning rates, continual training, assembly, evaluation | To rebuild the system from nothing |
| [`experiments/`](experiments/README.md) | The final experiments: ranking and stability/plasticity, the evidence runs, the learning-rate fairness check, the readout sweep | To regenerate the final results |
| [`figures/`](figures/README.md) | Every figure script (original architecture and circuit diagrams, plus the final charts) | To redraw any figure |
| [`report/`](report/README.md) | Builds the results file, the presentation and the Word report | To rebuild the deliverables |
| `build_docs/` | Original (version 1) JavaScript generators for the archived documents | Only for the archive; see its README |

## Order of a full rebuild

```
pipeline/gen_dataset.py  →  fit_encoder.py  →  build_memory.py  →  study_trainability.py
   →  tune_lr.py  →  build_scoring_cache.py  →  train_continual.py  →  run_assembly.py  →  run_eval.py
experiments/final_eval.py, evidence_eval.py, plasticity_fairness.py   (3 seeds × 4 models)
figures/make_final_figures.py, make_evidence_figures.py  →  report/finalize_results.py
figures/make_slide_diagrams.py, make_slide_figures.py  →  report/build_presentation.py, build_team_report.js
```

`pipeline/` and `experiments/` read and write the `qcs` package's own folders (`data/`, `runs/`, `results/`),
so a script can be run on its own once the earlier phases have produced their files.
