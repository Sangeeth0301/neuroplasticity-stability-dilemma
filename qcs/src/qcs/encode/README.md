# `encode/` — text to a 1,024-number unit vector

The whole package is **frozen after Phase 2**. Nothing here learns during the continual run, so
any difference between models comes from the scorer and nothing else.

```
text ─► FrozenEncoder (MiniLM, 384) ─► Whitener (256) ─► Projection (256) ─┐
                                                                            ├─► assemble ─► 1,024 unit vector
 query/candidate/user info ─► build_rho (256 relational features) ─────────┘
```

| File | What it does | Logic worth knowing |
|---|---|---|
| `encoder.py` | `FrozenEncoder`: the sentence encoder (`all-MiniLM-L6-v2`, 384 numbers per text). | Parameters are set to `requires_grad=False`. If the model cannot be downloaded, a deterministic hashing fallback keeps the plumbing runnable; every run records which backend was used. The fallback is **not** a semantic encoder. |
| `cache.py` | `EmbeddingCache`: embeddings stored on disk, keyed by `sha256(text)`. | Keyed by content, not by position, so it survives regeneration and reordering. |
| `whiten.py` | `Whitener`: PCA whitening fitted once on held-out threads. | Raw sentence embeddings are very uneven (a few directions hold most of the variance). Amplitude encoding turns components into amplitudes, so uneven input would leave most of the register empty. Whitening uses an SVD with shrinkage `eps = 0.1` (chosen from the whitening study, not by default). `fit()` refuses to run twice, because refitting mid-run would be an invisible distribution shift. |
| `projection.py` | `Projection`: a linear map from 256 to 256 numbers trained once with InfoNCE. | Positives are (question, needed fact) pairs; negatives are hard in-thread distractors that mention the same entity. After training it is frozen. This is the last classical component that learns anything about meaning. |
| `relational.py` | `build_rho`: the 256 "relational" features. | Layout: salience `[0:64]`, recency `[64:80]`, graph `[80:112]`, speaker `[112:128]`, usage `[128:136]`, reserve `[136:256]`. **Every feature is bounded to [-1, 1]** before assembly, because amplitude encoding normalises the vector and would destroy raw magnitudes. Turn gaps are encoded as `tanh(log1p(gap)/6)`, `1/(1+gap)`, band flags and sin/cos phases. |
| `assemble.py` | `assemble(p_q, p_c, rho)`: builds the final feature vector. | Layout `[ question 256 | candidate 256 | interaction 256 | relational 256 ]`. The interaction block is the entry-by-entry product of the two embeddings (its sum is the cosine similarity). **Each block is scaled to a designed norm share** (0.35, 0.35, 0.20, 0.10) before the whole vector is set to length 1. Without this, one block swallowed the probability mass and participation fell to 61 of 1,024 states. |
| `pipeline.py` | `EncodingPipeline`: the whole chain as one object. | `triple_features()` returns one unit-norm row per candidate. `save()` / `load()` use `data/processed/whiten.npz` and `projection.npz`. |

## Why exactly 1,024 numbers

4 blocks × 256 = 1,024 = 2¹⁰. That is the size of a 10-qubit state, which is why the circuit has 10 qubits
(`Config.validate()` refuses any other combination).

## Tests

`tests/test_encode_shapes.py`: lossless segmentation, unit norm and 1,024 dimensions, block shares hit their targets,
relational features are bounded, the whitener refuses to refit and flattens the spectrum.
