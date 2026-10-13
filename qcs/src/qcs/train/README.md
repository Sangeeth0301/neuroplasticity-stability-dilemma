# `train/` — the continual-learning loop

This is the only place where parameters are updated. The encoder, whitening, projection and memory are
not even reachable from here, which is how "no gradient touches anything but the scorer" is enforced instead of promised.

| File | What it does | Logic worth knowing |
|---|---|---|
| `dataset.py` | `ScoringDataset`: turns the corpus into training rows. | For each probe: take the top-M **earlier** candidates from the memory index (an assertion checks that no future candidate leaks), build one 1,024-number feature row per candidate with `EncodingPipeline.features`, and label the candidates that state the needed fact (1) or not (0). It also records which candidates are stale and their token cost. Rows are stored as fp16 object arrays (probes have different candidate counts) and cached in `data/processed/scoring_{train,test}.npz`. |
| `loop.py` | `run_continual(model, stream, cfg, log, arm, kind, ...)`: the experiment itself. | See the walk-through below. Also contains `TaskStream` (groups probes into tasks of 6, in order) and `ContinualResult` (the A-matrix and BWT / forward transfer / plasticity index / final average). |
| `replay.py` | `ReservoirBuffer`: uniform replay over the **whole** history. | Reservoir sampling gives every example seen so far the same chance of being in the buffer, using fixed memory. A first-in-first-out buffer would only hold recent tasks and protect against nothing. |
| `anchor.py` | `FisherAnchor`, `TrustRegion`, `DriftDetector`. | **Anchor:** penalty `½ λ Σ Fₖ (θₖ − θ*ₖ)²`, where θ* is the angle set at the last checkpoint and F is the Fisher importance (normalised to mean 1 so λ means the same thing for every model). **Trust region:** after each task, if any angle moved more than 0.35, the whole movement is scaled back. **Drift detector:** population stability index on a 10-bin histogram of scores; when it passes 0.25 the reference is re-based so the anchor does not pin the model to its first tasks forever. |

## One task in `run_continual`

1. Build the task's training rows (`_flatten_task`): all positives plus up to 8 negatives per probe, stale ones first.
2. For 3 epochs, per batch of 32: add an equal number of replayed rows, compute `BCE(score, label) + anchor penalty`, take an Adam step.
3. Apply the trust region, then wrap the angles (quantum models only) and add the task to the replay buffer.
4. Score the task's own probes for the drift detector; if drift fired, or every 10th task, take a Fisher checkpoint (quantum Fisher for circuits, ordinary empirical Fisher for classical models).
5. Evaluate: `A[t, t]` on the current task, and 6 randomly chosen past tasks (3 probes each); other past entries carry the last known value.
6. Log loss, Fisher trace, parameter norms, anchor pull, clip rate and drift every task (the Fisher trace every 5 tasks).
7. Call the optional `on_task_end(t, model)` hook. The final experiments use it to run extra measurements at chosen tasks without changing training.

## Things to know

- The learning rate comes from `cfg.train.lr_by_arm[kind]`, so each model trains at its own tuned rate. `kind` must be passed explicitly: deriving it from a label such as `abl_no_anchor_s7` once trained every ablation at the wrong learning rate.
- Forward transfer comes out as 0 because tasks that have not arrived yet are never evaluated in the matrix; the final experiments measure the score on an arriving task directly (`scripts/experiments/evidence_eval.py`).

## Tests

`tests/test_train_assemble.py`: reservoir is uniform over history; replay sample shapes; the anchor is zero at the checkpoint and grows away from it; the anchor switches off at λ = 0; the trust region clips and reports; the drift detector fires on a shift and not on noise; BWT and forward transfer on a known matrix.
