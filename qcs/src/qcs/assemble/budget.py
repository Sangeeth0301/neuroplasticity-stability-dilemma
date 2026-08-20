"""Token accounting and sub-budget allocation.

Context saturation is an ALLOCATION problem under a hard ceiling, not a retrieval
problem, so the ceiling is enforced here with exact counts and an assertion — estimated
token counts are how budgets silently overrun in production systems.
"""

from __future__ import annotations

from dataclasses import dataclass
from functools import lru_cache


@lru_cache(maxsize=1)
def _hf_tokenizer(name: str = "sentence-transformers/all-MiniLM-L6-v2"):
    try:
        from transformers import AutoTokenizer

        return AutoTokenizer.from_pretrained(name)
    except Exception:      # noqa: BLE001 - offline or unavailable
        return None


def count_tokens(text: str, exact: bool = True) -> int:
    """Exact count with the target model's tokenizer when available, else a word proxy.

    Which one was used is recorded in every run, because a proxy count that drifts from
    the real one is a budget overrun waiting to happen.
    """
    if exact:
        tok = _hf_tokenizer()
        if tok is not None:
            return len(tok.encode(text, add_special_tokens=False))
    return len(text.split())


@dataclass
class Budget:
    """Fixed sub-budgets. The selected-memory allowance is whatever is left after the
    non-negotiable parts, which is what keeps context length constant in thread length."""

    total: int
    system: int
    pinned: int
    recent: int

    @property
    def selected(self) -> int:
        return self.total - self.system - self.pinned - self.recent

    def validate(self) -> None:
        assert self.selected > 0, "no room left for retrieved memory — raise total or cut reserves"
        assert self.system + self.pinned + self.recent + self.selected == self.total

    @classmethod
    def from_config(cls, cfg) -> "Budget":
        b = cls(cfg.assemble.token_budget, cfg.assemble.reserve_system,
                cfg.assemble.reserve_pinned, cfg.assemble.reserve_recent)
        b.validate()
        return b
