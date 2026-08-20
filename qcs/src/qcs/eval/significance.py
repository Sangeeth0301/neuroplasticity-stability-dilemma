"""Paired bootstrap over THREADS, not over probes.

Probes inside a thread share facts, participants and a template distribution, so they are
not independent. Resampling probes would shrink the confidence interval by roughly the
square root of the probes-per-thread and manufacture significance. The thread is the
independent unit, so the thread is what gets resampled.
"""

from __future__ import annotations

import numpy as np


def paired_bootstrap(
    per_probe_a: dict[str, list[float]],
    per_probe_b: dict[str, list[float]],
    n_boot: int = 5000,
    seed: int = 0,
) -> dict:
    """Both inputs are {thread_id: [per-probe metric]} for the same probes.

    Returns the mean difference (a − b), a 95% CI, and a two-sided p-value.
    """
    threads = sorted(set(per_probe_a) & set(per_probe_b))
    rng = np.random.default_rng(seed)

    def stat(sample: list[str]) -> float:
        a = np.concatenate([per_probe_a[t] for t in sample])
        b = np.concatenate([per_probe_b[t] for t in sample])
        return float(np.mean(a) - np.mean(b))

    observed = stat(threads)
    draws = np.array([stat(list(rng.choice(threads, len(threads), replace=True)))
                      for _ in range(n_boot)])
    lo, hi = np.percentile(draws, [2.5, 97.5])
    # two-sided p: how often the bootstrap distribution crosses zero
    p = 2.0 * min(float(np.mean(draws <= 0)), float(np.mean(draws >= 0)))
    return {
        "n_threads": len(threads),
        "difference": observed,
        "ci_low": float(lo),
        "ci_high": float(hi),
        "p_value": min(1.0, p),
        "significant": bool(lo > 0 or hi < 0),
    }


def per_thread_metric(score_fn, examples, metric_fn) -> dict[str, list[float]]:
    out: dict[str, list[float]] = {}
    for ex in examples:
        out.setdefault(ex.thread_id, []).append(float(metric_fn(np.asarray(score_fn(ex)), ex.y)))
    return out
