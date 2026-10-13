"""Configuration: every number in the system lives in YAML and arrives through here.

Rule enforced by tests/test_config.py: no module outside this one may define a
tunable constant. If you find yourself typing a number into an algorithm, it
belongs in configs/base.yaml.
"""

from __future__ import annotations

import hashlib
import json
from dataclasses import asdict, dataclass, field, fields, is_dataclass
from pathlib import Path
from typing import Any, get_type_hints

import yaml

from . import paths


# --------------------------------------------------------------------------- #
# Sub-configs
# --------------------------------------------------------------------------- #
@dataclass
class DataConfig:
    n_threads_train: int = 24
    n_threads_test: int = 6
    turns_per_thread: int = 600
    n_users: int = 5
    n_facts_per_thread: int = 60
    probe_distances: list[int] = field(default_factory=lambda: [10, 50, 200, 500])
    probes_per_fact: int = 1
    filler_ratio: float = 0.55
    distractor_ratio: float = 0.25
    supersede_ratio: float = 0.35
    regimes: list[str] = field(default_factory=lambda: ["single_owner", "multi_owner", "adversarial"])
    su_target_tokens: int = 40


@dataclass
class EncodeConfig:
    model_name: str = "sentence-transformers/all-MiniLM-L6-v2"
    embed_dim: int = 384
    proj_dim: int = 256
    whiten_components: int = 256
    whiten_eps: float = 0.1          # shrinkage; 1e-6 amplifies noise directions (V06b study)
    feature_dim: int = 1024
    block_norm_targets: list[float] = field(default_factory=lambda: [0.35, 0.35, 0.20, 0.10])
    batch_size: int = 128


@dataclass
class MemoryConfig:
    candidate_M: int = 256
    consolidate_every: int = 50
    salience_alpha: float = 0.05
    ann_nlist: int = 64


@dataclass
class QuantumConfig:
    n_qubits: int = 10
    depth: int = 8
    ansatz: str = "brickwall"
    gate_form: str = "exp"               # exp (paper form) | product (shift-rule valid)
    readout_qubit: int = 9
    observable: str = "local_z"          # local_z | global_z
    beta_init: float = 4.0
    bias_init: float = 0.0
    init_scale: float = 0.05             # identity-block initialisation
    noise_p: float = 0.0


@dataclass
class TrainConfig:
    task_window_turns: int = 40
    epochs_per_task: int = 3
    batch_size: int = 32
    lr: float = 0.03
    lr_classical: float = 1e-3
    lr_by_arm: dict[str, float] = field(default_factory=dict)   # set by scripts/pipeline/tune_lr.py
    replay_capacity: int = 8000
    replay_ratio: float = 0.5
    lambda_omega: float = 1e-2
    trust_region_delta: float = 0.35
    drift_threshold: float = 0.25
    fim_probe_size: int = 256
    n_negatives: int = 8


@dataclass
class AssembleConfig:
    token_budget: int = 3072
    reserve_system: int = 256
    reserve_pinned: int = 256
    reserve_recent: int = 768
    redundancy_lambda: float = 0.35
    value_prior: dict[str, float] = field(
        default_factory=lambda: {"fact": 1.0, "summary": 0.7, "filler": 0.35}
    )


@dataclass
class Config:
    seed: int = 7
    run_tag: str = "dev"
    data: DataConfig = field(default_factory=DataConfig)
    encode: EncodeConfig = field(default_factory=EncodeConfig)
    memory: MemoryConfig = field(default_factory=MemoryConfig)
    quantum: QuantumConfig = field(default_factory=QuantumConfig)
    train: TrainConfig = field(default_factory=TrainConfig)
    assemble: AssembleConfig = field(default_factory=AssembleConfig)

    # ---------------- loading ---------------- #
    @classmethod
    def load(cls, *names: str) -> "Config":
        """Load configs/base.yaml then overlay each named config (without .yaml)."""
        merged: dict[str, Any] = {}
        for name in ("base", *names):
            path = paths.CONFIGS / f"{name}.yaml"
            if not path.exists():
                raise FileNotFoundError(f"config not found: {path}")
            with open(path, "r", encoding="utf-8") as fh:
                merged = _deep_merge(merged, yaml.safe_load(fh) or {})
        return cls.from_dict(merged)

    @classmethod
    def from_dict(cls, d: dict[str, Any]) -> "Config":
        return _build(cls, d)

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)

    def save(self, path: str | Path) -> None:
        with open(path, "w", encoding="utf-8") as fh:
            yaml.safe_dump(self.to_dict(), fh, sort_keys=False)

    def hash(self) -> str:
        blob = json.dumps(self.to_dict(), sort_keys=True).encode()
        return hashlib.sha256(blob).hexdigest()[:10]

    # ---------------- invariants ---------------- #
    def validate(self) -> None:
        q = self.quantum
        assert 2**q.n_qubits == self.encode.feature_dim, (
            f"feature_dim {self.encode.feature_dim} must equal 2**n_qubits "
            f"= {2**q.n_qubits} for lossless amplitude encoding"
        )
        assert 0 <= q.readout_qubit < q.n_qubits
        assert q.depth >= q.n_qubits // 2, (
            f"depth {q.depth} < light-cone floor {q.n_qubits // 2}: parameters on the far "
            "side of the register cannot influence the readout qubit (see Phase 5 study)"
        )
        assert abs(sum(self.encode.block_norm_targets) - 1.0) < 1e-9
        a = self.assemble
        assert a.reserve_system + a.reserve_pinned + a.reserve_recent < a.token_budget
        assert 0.0 <= self.train.replay_ratio < 1.0


# --------------------------------------------------------------------------- #
def _deep_merge(a: dict, b: dict) -> dict:
    out = dict(a)
    for k, v in b.items():
        out[k] = _deep_merge(out[k], v) if isinstance(out.get(k), dict) and isinstance(v, dict) else v
    return out


def _build(cls: type, d: dict[str, Any]) -> Any:
    kwargs: dict[str, Any] = {}
    # `from __future__ import annotations` turns field.type into a string, so resolve
    # the real types here rather than trusting the dataclass metadata.
    hints = get_type_hints(cls)
    known = {f.name for f in fields(cls)}
    for key, value in d.items():
        if key not in known:
            raise KeyError(f"unknown config key '{key}' for {cls.__name__}")
        ftype = hints[key]
        if is_dataclass(ftype) and isinstance(value, dict):
            kwargs[key] = _build(ftype, value)
        else:
            kwargs[key] = value
    return cls(**kwargs)
