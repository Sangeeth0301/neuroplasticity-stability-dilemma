"""L5 · scores → prompt, and the frozen reader that answers from it.

Two things happen here that the scorer cannot do on its own:

  1. **Supersession filtering.** A stale fact is either dropped or included *explicitly
     labelled as outdated together with its replacement*. It is never served silently —
     that is a hard rule, not a ranking preference.
  2. **Budgeted assembly.** Reserved sub-budgets, exact token counts, provenance tags,
     chronological ordering within blocks.

On the reader: this project has no LLM API wired in, so `ExtractiveReader` stands in for
the frozen generator. It answers a probe only if the required fact's statement is present
in the assembled context. That measures exactly what this architecture is responsible for
— whether the right memory reached the prompt — and it cannot flatter the system, since it
has no world knowledge to fall back on. Swapping in a real frozen LLM changes `answer()`
and nothing else.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from ..types import Fact, SemanticUnit
from .budget import Budget, count_tokens
from .knapsack import Selection, greedy_knapsack, topk_baseline, value_prior

TEMPLATE_VERSION = "v1"

SYSTEM = ("You are answering from a shared team thread. Use only the recorded memory "
          "below. Each line is tagged with who said it and when.")


@dataclass
class AssembledPrompt:
    text: str
    selection: Selection
    tokens_total: int
    tokens_selected: int
    n_selected: int
    stale_included: int
    stale_labelled: int


def render(query: SemanticUnit, sus: list[SemanticUnit], sel: Selection,
           recent: list[SemanticUnit], stale_ids: set[str],
           replacements: dict[str, SemanticUnit] | None = None,
           exact_tokens: bool = True) -> AssembledPrompt:
    lines = [SYSTEM, "", "## Recorded memory"]
    stale_in, stale_lab = 0, 0
    for i in sorted(sel.idx, key=lambda j: sus[j].turn_idx):     # chronological within block
        su = sus[i]
        tag = f"[{su.user_id} · turn {su.turn_idx}]"
        if su.su_id in stale_ids:
            stale_in += 1
            repl = (replacements or {}).get(su.su_id)
            if repl is None:
                continue                                        # drop rather than mislead
            stale_lab += 1
            lines.append(f"{tag} OUTDATED: {su.text}")
            lines.append(f"[{repl.user_id} · turn {repl.turn_idx}] CURRENT: {repl.text}")
        else:
            lines.append(f"{tag} {su.text}")

    lines += ["", "## Recent turns"]
    lines += [f"[{s.user_id} · turn {s.turn_idx}] {s.text}" for s in recent]
    lines += ["", "## Question", f"[{query.user_id}] {query.text}"]

    text = "\n".join(lines)
    return AssembledPrompt(
        text=text,
        selection=sel,
        tokens_total=count_tokens(text, exact_tokens),
        tokens_selected=sel.tokens_used,
        n_selected=len(sel.idx),
        stale_included=stale_in,
        stale_labelled=stale_lab,
    )


def assemble(
    query: SemanticUnit,
    sus: list[SemanticUnit],
    scores: np.ndarray,
    tokens: np.ndarray,
    kinds: list[str],
    budget: Budget,
    cfg,
    vectors: np.ndarray | None = None,
    stale_ids: set[str] | None = None,
    replacements: dict[str, SemanticUnit] | None = None,
    recent_k: int = 6,
    strategy: str = "knapsack",
) -> AssembledPrompt:
    stale_ids = stale_ids or set()
    value = value_prior(kinds, cfg.assemble.value_prior)
    if strategy == "knapsack":
        sel = greedy_knapsack(scores, tokens, budget.selected, vectors=vectors, value=value,
                              redundancy_lambda=cfg.assemble.redundancy_lambda)
    elif strategy == "topk":
        sel = topk_baseline(scores, tokens, budget.selected)
    else:
        raise KeyError(strategy)

    recent = sorted([s for s in sus if s.turn_idx < query.turn_idx],
                    key=lambda s: -s.turn_idx)[:recent_k][::-1]
    return render(query, sus, sel, recent, stale_ids, replacements)


class ExtractiveReader:
    """The frozen generator stand-in. Answers only from what is in the context."""

    version = "extractive-v1"

    @staticmethod
    def answer(prompt: str, candidates: list[Fact]) -> str | None:
        """Return the value of whichever candidate fact is present in the prompt.

        Prefers a value tagged CURRENT so that a correctly-labelled stale pair does not
        cost a point.
        """
        current_block = "\n".join(line for line in prompt.splitlines()
                                  if "OUTDATED:" not in line)
        for f in candidates:
            if f.value and f.value in current_block:
                return f.value
        return None

    @staticmethod
    def correct(answer: str | None, gold: str) -> bool:
        return answer is not None and answer.strip().lower() == gold.strip().lower()
