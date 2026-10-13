"""Run bookkeeping: one directory per experiment, one JSONL row per task.

Every figure in the project is regenerable from metrics.jsonl alone. If a plot
needs a number, log the number — never compute it inside plotting code.
"""

from __future__ import annotations

import json
import platform
import subprocess
import sys
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from . import paths

# Rows written to metrics.jsonl must be a subset of this schema. Guarded so that a
# typo in a key name fails loudly instead of silently producing an empty plot.
METRIC_KEYS = {
    "task_id", "arm", "phase", "split",
    "acc", "loss", "replay_loss", "auc", "recall_at_b", "answer_f1",
    "tr_fim", "grad_norm", "weight_norm", "weight_norm_unwrapped", "eff_rank",
    "anchor_pull", "omega", "clip_rate", "drift_psi", "checkpoint",
    "tokens_used", "n_selected", "latency_ms", "cohens_d", "lr", "extra",
}


def git_hash() -> str:
    try:
        out = subprocess.run(
            ["git", "rev-parse", "--short", "HEAD"],
            capture_output=True, text=True, cwd=paths.ROOT, timeout=5,
        )
        return out.stdout.strip() or "nogit"
    except Exception:
        return "nogit"


def env_info() -> dict[str, Any]:
    info: dict[str, Any] = {
        "python": sys.version.split()[0],
        "platform": platform.platform(),
        "git": git_hash(),
        "utc": datetime.now(timezone.utc).isoformat(timespec="seconds"),
    }
    for mod in ("numpy", "scipy", "torch", "pennylane", "sentence_transformers", "faiss"):
        try:
            m = __import__(mod)
            info[mod] = getattr(m, "__version__", "?")
        except Exception:
            info[mod] = None
    return info


@dataclass
class RunLogger:
    run_dir: Path

    @classmethod
    def create(cls, cfg, tag: str, root: Path | None = None) -> "RunLogger":
        """Make a run directory. `root` defaults to runs/; experiment scripts that only need
        throw-away logs pass runs/scratch (git-ignored) so runs/ stays a clean record."""
        stamp = datetime.now().strftime("%Y-%m-%dT%H-%M-%S")
        run_dir = (root or paths.RUNS) / f"{stamp}_{git_hash()}_{tag}_{cfg.hash()}"
        (run_dir / "figs").mkdir(parents=True, exist_ok=True)
        (run_dir / "checkpoints").mkdir(parents=True, exist_ok=True)
        cfg.save(run_dir / "config.yaml")
        (run_dir / "env.json").write_text(json.dumps(env_info(), indent=2), encoding="utf-8")
        return cls(run_dir)

    @property
    def figs(self) -> Path:
        return self.run_dir / "figs"

    def log(self, **row: Any) -> None:
        unknown = set(row) - METRIC_KEYS
        if unknown:
            raise KeyError(f"unknown metric keys {sorted(unknown)}; extend METRIC_KEYS deliberately")
        with open(self.run_dir / "metrics.jsonl", "a", encoding="utf-8") as fh:
            fh.write(json.dumps(row, default=float) + "\n")

    def note(self, name: str, obj: Any) -> None:
        """Dump an arbitrary JSON artefact next to the metrics."""
        (self.run_dir / f"{name}.json").write_text(
            json.dumps(obj, indent=2, default=str), encoding="utf-8"
        )


def write_jsonl(path: str | Path, rows) -> int:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    n = 0
    with open(path, "w", encoding="utf-8") as fh:
        for r in rows:
            fh.write(json.dumps(r, default=str) + "\n")
            n += 1
    return n


def read_jsonl(path: str | Path) -> list[dict]:
    with open(path, "r", encoding="utf-8") as fh:
        return [json.loads(line) for line in fh if line.strip()]
