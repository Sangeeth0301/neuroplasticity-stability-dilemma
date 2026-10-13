# `scripts/experiments/` — the final experiments

These produce every number in the final report and presentation. Each script takes a **model** and a **seed**,
trains through the same 120-task stream, and writes one JSON file into `results/raw/`. They create throw-away run logs in
`runs/scratch/` (git-ignored), so `runs/` stays clean.

```
python scripts/experiments/final_eval.py        MODEL SEED     # MODEL: quantum | multi | linear | mlp
python scripts/experiments/evidence_eval.py     MODEL SEED
python scripts/experiments/plasticity_fairness.py MODEL SEED   # MODEL: multi | linear | mlp
python scripts/experiments/sweep_quantum.py NAME [--multi] [--depth N] [--epochs N] [--lr X] [--beta X] ...
```

`quantum` = the original circuit (1-qubit readout). `multi` = the final circuit (10-qubit readout).
Seeds used: 7, 17, 27. Results are deterministic: re-running a script reproduces its JSON exactly.

## What each script measures

| Script | Question it answers | Output (`results/raw/<folder>/MODEL_sSEED.json`) |
|---|---|---|
| `final_eval.py` | How good is the ranking, and do stability and plasticity hold as training goes on? | Ranking metrics on the training stream and the sealed test split; per-thread test scores (for the bootstrap); the stability score (MRR on the first 30 tasks) at tasks 1, 30, 60, 90, 120; the plasticity score at the same points; Fisher trace per task; the A-matrix. |
| `evidence_eval.py` | Does it learn new things and remember old ones, directly? | The score on each task **before** training on it (`pre`) and after (`post`); the **full forgetting matrix** (after every 10th task, every past task re-tested, no carried-forward values); test MRR by distance back to the needed fact, against a "last 100 turns only" baseline; probes with and without an outdated fact among the candidates. |
| `plasticity_fairness.py` | Are the classical models just using the wrong learning rate for the new rule? | The reversed-rule score after 40 and 80 steps at six learning rates (0.001 to 0.3); the best for each. |
| `sweep_quantum.py` | Which circuit settings work? | One JSON line appended to `results/exploration/sweep_results.jsonl`. This is how the 10-qubit readout was found; see `results/exploration/README.md`. |

## The reversed-rule (reversal) test, in detail

This is the plasticity measurement. After a chosen number of tasks the model is **copied** and trained for 40 Adam steps on a
rule it has never seen: for probes that have an outdated fact among the candidates, the labels are swapped, so the outdated
fact is now the correct one. Half of those probes are used to train the copy and half to score it. The training stream is not
touched; the copy is thrown away. A model that has lost its ability to learn scores low.

## Definitions to be careful with

- **"Updated facts"** in the results means probes that have at least one superseded (outdated) fact among their 64 candidates
  (`ex.is_stale.sum() > 0`). Updates exist only in `adversarial` threads, so these probes come from those threads.
- **Stability score** = MRR of the model on the first 30 tasks, measured at each checkpoint. **Plasticity score** = MRR on the held-out
  reversal probes after 40 steps. **Combined** = harmonic mean of the two.
