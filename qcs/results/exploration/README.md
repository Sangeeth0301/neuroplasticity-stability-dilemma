# Exploration log — how the final circuit was found

The original circuit (reading out one qubit) ranked clearly below a linear model of the same size. These runs were the search for
a fix. They used the **training stream only** (seed 7 first, then three seeds); the sealed test split was opened only for the final
comparison in `../RESULTS_FINAL.md`.

## 1. A failed idea: a fixed similarity prior

Idea: give the circuit the similarity signal directly and let it learn only the rest.
`logit = gain · (sum of the interaction block) + β⟨Z⟩ + b` (`HybridQRE`), with a classical control that uses the same prior plus a linear
residual (`HybridLinearScorer`). Seed 7, 120 tasks:

| Model | MRR | Rank-1 | Stale above current |
|---|---|---|---|
| hybrid circuit | 0.938 | 0.890 | 0.073 |
| hybrid linear (control) | 0.909 | 0.844 | 0.096 |
| cosine, no learning | 0.943 | 0.894 | 0.078 |

Both were worse than plain cosine similarity, so the idea was dropped. The code stays in `src/qcs/quantum/hybrid.py` as a record.

## 2. First sweep (seed 7, training stream, 8 variants)

Recorded from the console output when the runs finished (the raw log files were not kept).

| Variant | MRR | Rank-1 | Stale above | Final average | Fisher trace (end ÷ start) |
|---|---|---|---|---|---|
| base (original, 1-qubit readout) | 0.906 | 0.847 | 0.032 | 0.920 | 4.3× |
| β = 8 | 0.959 | 0.929 | 0.009 | 0.948 | 1.4× |
| depth 12 | 0.981 | 0.964 | 0.018 | 0.952 | 0.6× |
| 6 epochs per task | 0.881 | 0.806 | 0.018 | 0.951 | 8.7× |
| no anchor (λ = 0) | 0.881 | 0.810 | 0.037 | 0.927 | 2.9× |
| learning rate 0.05 | 0.792 | 0.676 | 0.091 | 0.723 | 7.7× |
| **10-qubit readout** | **0.988** | **0.976** | **0.000** | **0.992** | 3.3× |
| 10-qubit readout, 6 epochs | 0.975 | 0.954 | 0.023 | 0.989 | 0.6× |

The 10-qubit readout was the only variant that was good on ranking, stability and kept the Fisher trace growing.

## 3. Confirmation runs (`sweep_confirmation.jsonl`)

Produced by `scripts/experiments/sweep_quantum.py`.

| Run | Seed | Parameters | MRR | Rank-1 | Stale above | BWT | Final avg | Fisher trace ratio |
|---|---|---|---|---|---|---|---|---|
| base | 17 | 1,080 | 0.906 | 0.844 | 0.032 | +0.015 | 0.929 | 9.39× |
| base | 27 | 1,080 | 0.856 | 0.774 | 0.037 | +0.014 | 0.917 | 5.03× |
| 10-qubit readout | 17 | 1,080 | 0.975 | 0.953 | 0.009 | +0.001 | 0.984 | 1.65× |
| 10-qubit readout | 27 | 1,080 | 0.980 | 0.963 | 0.023 | +0.007 | 0.988 | 2.59× |
| 10-qubit, β = 8 | 7 | 1,080 | 0.978 | 0.957 | 0.014 | −0.007 | 0.986 | 0.20× |
| 10-qubit, β = 8 | 17 | 1,080 | 0.979 | 0.961 | 0.032 | +0.003 | 0.985 | 0.40× |
| 10-qubit, depth 12 | 7 | 1,620 | 0.982 | 0.965 | 0.014 | −0.001 | 0.991 | 30.26× |
| 10-qubit, depth 12 | 17 | 1,620 | 0.974 | 0.951 | 0.018 | −0.011 | 0.972 | 0.31× |

Conclusion: the 10-qubit readout with the default depth (8) and β (4) was kept. Deeper circuits and a larger β gave erratic
Fisher traces (30× on one seed, 0.2× to 0.4× on others) without a clear ranking benefit.

The seed-to-seed spread of the original circuit is large (MRR 0.856 to 0.906 here), which is why the final comparison uses
three seeds.
