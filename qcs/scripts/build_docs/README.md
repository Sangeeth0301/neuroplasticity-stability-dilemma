# `scripts/build_docs/` — version 1 document generators (legacy)

These JavaScript files generated the original deliverables that are now archived in `deliverables/archive_v1/`:
the first report, the ten-phases document and the first deck. They are kept so the archive can be regenerated, but they describe the
**original** single-readout results and are superseded by the final report and presentation
(`scripts/report/`).

| File | Produces |
|---|---|
| `build_report.js` | Shared helpers for the Word documents (tables, figures, headings) |
| `report_body.js` | The version 1 technical report |
| `build_phases_doc.js` | The ten-phases document |
| `build_deck.js` | The version 1 deck |

Run with `npm install` once in `qcs/`, then for example `node scripts/build_docs/report_body.js`. They write into `deliverables/`;
move the results into `deliverables/archive_v1/` to keep the folder tidy.
