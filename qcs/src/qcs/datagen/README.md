# `datagen/` — the synthetic chat corpus

We generate the chats ourselves because real logs do not say which earlier message a question
depends on. Here the plan decides everything first, and text is written afterwards, so the
correct answer to every question is known exactly.

| File | What it does | Logic worth knowing |
|---|---|---|
| `fact_bank.py` | Builds the facts of one thread and holds the sentence templates. | A fact is an (entity, attribute, value) triple with an owner. There are 7 attributes (deadline, owner, budget, status, location, tool, preference). Entities are two-word names such as "Helios dashboard", and **three facts share each entity** with different attributes, so a retriever cannot rely on the entity name alone. Regimes: `single_owner` (one participant owns all facts), `multi_owner` (facts spread across participants), `adversarial` (as multi-owner, plus updates added by the planner). |
| `thread_planner.py` | `plan_thread(...)`: decides what happens on every turn before any text exists. | Probe distance bands (10, 50, 200, 500 turns) are assigned **first**, round-robin, and the fact is then placed where it still leaves room for its probe. In `adversarial` threads about 35% of facts get an `update` turn 15 to 120 turns later ("Correction on the X: it's V now, not OLD"); the old fact gets `superseded_by`. Only facts that are *not* superseded are probed. In `multi_owner` and `adversarial` threads the person asking is someone other than the fact's owner. Remaining turns become `filler` or `distractor`. |
| `surface_realizer.py` | Text for filler and distractor turns. | Templates, not a language model, so the corpus is reproducible from a seed. Distractors name a **real entity with no usable value** ("The X came up in another thread, but nothing was decided"), which makes them close to the right answer in wording. |
| `generate.py` | `write_corpus(cfg)`: plans all threads and writes `data/raw/*.jsonl`. | Thread `i` uses regime `regimes[i % 3]` and random stream `child_rng(seed, split, i)`. Writes `fact_bank.jsonl`, `threads_train.jsonl` and the sealed `threads_test.jsonl`. |
| `validate.py` | Checks that the corpus is a valid measuring instrument (rules V1 to V7). | Every probe's needed fact appears strictly earlier (V1); no probe needs a superseded fact (V2); ids are unique (V3); turn indices are contiguous (V4); the distance bands are covered (V5); train and test threads do not overlap (V6); an oracle retriever scores 1.0 and a random one scores near chance (V7). |
| `stats.py` | Corpus statistics used by figures V01 to V04. | Turn kinds, words per turn, probe distances, speaker counts. |

## Numbers (from `configs/base.yaml`)

30 threads (24 train, 6 test) × 600 turns = 18,000 turns; 5 participants; 60 facts per thread;
filler 55%, distractor 25%; update share 35% (adversarial threads only).

## Tests

`tests/test_datagen.py` covers validity, calibration (oracle versus random), answerability,
determinism, and that the validator really catches a broken corpus.
