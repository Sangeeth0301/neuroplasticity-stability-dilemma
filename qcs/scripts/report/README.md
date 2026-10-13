# `scripts/report/` — building the deliverables

These read the results in `results/` and the figures in `figures/final/` and write into `deliverables/`.

| Script | What it does | Output |
|---|---|---|
| `finalize_results.py` | Collects the raw results: the learning-rate fairness figure, the paired bootstrap on the test split, and the written results file. | `results/RESULTS_FINAL.md`, `figures/analysis/FIN5_fairness.*` |
| `build_presentation.py` | Builds the 24-slide deck with python-pptx (dark title slide, serif headings, light cards). Numbers come from `results/final_summary.json` and `evidence_summary.json`, so the deck cannot drift from the data. | `deliverables/Team_14_Dual_Manifold_ContextSync_Presentation.pptx` |
| `build_team_report.js` | Builds the Word report (A4, cover page with the Amrita logo, 10 pages) with the `docx` package. Run `npm install` once in `qcs/`. | `deliverables/Team_14_Dual_Manifold_ContextSync_Report.docx` |

```
python scripts/report/finalize_results.py
python scripts/report/build_presentation.py
node   scripts/report/build_team_report.js
```

The logo comes from `figures/branding/amrita_logo.png`. A PDF of the report is exported from the Word file.
