# `runs/` — the record of the build phases

One folder per build-phase run, named `<timestamp>_<git hash>_<tag>_<config hash>`. Each holds the configuration used
(`config.yaml`), the environment (`env.json`), a metrics log (`metrics.jsonl`, one line per task), the figures it drew (`figs/`)
and a summary JSON. Any figure of the project can be redrawn from the `metrics.jsonl` of its run.

| Run | Phase |
|---|---|
| `…phase1_dataset…` | Corpus generation and validation |
| `…phase2_encoder…`, `…phase2_whitening_study…` | Encoding stack and the whitening sweep |
| `…phase3_memory…` | Memory, graph, salience, retrieval recall |
| `…phase4_quantum…` | Quantum core: encoding, gradient variance, entanglement |
| `…phase6_continual…` | 120-task continual training of the three models (with checkpoints) |
| `…phase7_assembly…` | Prompt assembly and the budget sweep |
| `…phase8_eval…`, `…phase8_ablations…` | The original evaluation: seeds, ablations, noise, sealed test split |

These runs describe the **original** single-readout circuit. The final comparison (with the 10-qubit readout) is in
`../results/`. The experiment scripts write their throw-away logs to `runs/scratch/`, which is git-ignored and safe to delete.
