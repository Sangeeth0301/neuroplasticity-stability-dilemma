# `assemble/` — from scores to a prompt

The scorer ranks candidates; this package decides which of them actually go into a prompt of fixed size, and
guarantees that outdated facts are never served silently.

| File | What it does | Logic worth knowing |
|---|---|---|
| `budget.py` | Token counting and the fixed sub-budgets. | `Budget(total=3072, system=256, pinned=256, recent=768)`; the space left for retrieved memory (`selected`, 1,792 tokens) is whatever remains. Context length therefore does not grow with thread length. `count_tokens` uses the model's real tokenizer when available and falls back to a word count; the choice is recorded. |
| `knapsack.py` | `greedy_knapsack`: selection by gain per token. | Maximise `Σ score·value − λ·redundancy` under the token budget. The gain is a modular term minus a maximum-similarity penalty (λ = 0.35), which is submodular, so the greedy rule has the standard (1 − 1/e) guarantee. `topk_baseline` (take the highest scores) is kept to show the redundancy penalty earns its place, because top-k can fill the budget with paraphrases of one fact. `value_prior` values a fact at 1.0, a summary at 0.7, filler at 0.35. |
| `pipeline.py` | `assemble(...)`, `render(...)` and `ExtractiveReader`. | **Stale-fact rule:** a superseded unit is either dropped, or shown labelled `OUTDATED` together with its replacement labelled `CURRENT`; it is never included unlabelled. Selected memory is shown in time order with `[user · turn]` tags, followed by recent turns and the question. `ExtractiveReader` stands in for a language model: it answers only if the needed fact's value is present in the prompt (ignoring lines marked `OUTDATED`). |

## Important limit

No language-model API is connected. The extractive reader measures exactly what this package is responsible for, whether the right memory reached the prompt, and it cannot flatter the system because it has no world knowledge. Replacing it with a real frozen model changes `ExtractiveReader.answer` and nothing else.

## Tests

`tests/test_train_assemble.py`: reserves must leave room; the knapsack never exceeds the budget, prefers gain per token over raw score and suppresses near-duplicates; top-k can waste the budget; facts outrank filler; the reader ignores facts marked outdated; token counts are positive and increase with length.
