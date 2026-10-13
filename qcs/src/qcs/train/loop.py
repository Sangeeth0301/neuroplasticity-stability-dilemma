"""The continual learning loop.

A "task" is a window of consecutive probes from the thread stream. Tasks arrive in order,
the model trains on each one once, and after every task it is re-evaluated on a sample of
ALL past tasks — which is what fills the A[i][j] matrix that backward transfer, forward
transfer and the plasticity index are computed from.

This is the only module that updates parameters. The frozen LLM, the encoder, the
whitening and the projection are not even reachable from here, which is how "no gradient
touches the LLM" is enforced rather than promised.
"""

from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np
import torch

from ..config import Config
from ..eval.metrics import evaluate_probes
from ..logging_ import RunLogger
from ..quantum.fisher import fim_trace, qfim_diagonal
from .anchor import DriftDetector, FisherAnchor, TrustRegion
from .replay import ReservoirBuffer


@dataclass
class TaskStream:
    """Probes grouped into sequential tasks. Order is chronological within a thread and
    threads are visited in order, so the stream is genuinely non-stationary."""

    examples: list
    probes_per_task: int

    def tasks(self) -> list[list]:
        out = []
        for i in range(0, len(self.examples), self.probes_per_task):
            chunk = self.examples[i:i + self.probes_per_task]
            if chunk:
                for e in chunk:
                    e.task_id = len(out)
                out.append(chunk)
        return out


@dataclass
class ContinualResult:
    arm: str
    A: np.ndarray                      # A[i][j] = metric on task j after training task i
    history: list[dict] = field(default_factory=list)

    def bwt(self) -> float:
        T = len(self.A)
        if T < 2:
            return 0.0
        return float(np.mean([self.A[T - 1, j] - self.A[j, j] for j in range(T - 1)]))

    def fwt(self) -> float:
        T = len(self.A)
        if T < 2:
            return 0.0
        return float(np.mean([self.A[j - 1, j] - self.A[0, j] for j in range(1, T)]))

    def plasticity_index(self, window: int = 20) -> float:
        d = np.array([self.A[i, i] for i in range(len(self.A))])
        w = min(window, len(d) // 2) or 1
        first, last = d[:w].mean(), d[-w:].mean()
        return float(last / first) if first > 0 else float("nan")

    def final_average(self) -> float:
        return float(np.mean(self.A[-1, : len(self.A)]))


def _batches(X, y, bs, rng):
    idx = rng.permutation(len(X))
    for i in range(0, len(X), bs):
        j = idx[i:i + bs]
        yield X[j], y[j]


def run_continual(
    model,
    stream: TaskStream,
    cfg: Config,
    log: RunLogger,
    arm: str,
    kind: str | None = None,
    eval_every: int = 1,
    eval_past: int = 6,
    eval_probes: int = 3,
    fisher_every: int = 10,
    fim_log_every: int = 5,
    seed: int = 0,
    verbose: bool = True,
    on_task_end=None,
) -> ContinualResult:
    rng = np.random.default_rng(seed)
    tasks = stream.tasks()
    T = len(tasks)
    # `kind` is the model family and MUST be passed explicitly. Deriving it from the
    # arm label silently mis-trained every ablation run: a label like "abl_no_anchor_s7"
    # parsed to "abl", so a quantum circuit was trained at the classical learning rate
    # and the whole ablation table was invalid.
    kind = kind or arm.split("_")[0]
    is_quantum = kind in ("quantum", "hybrid_quantum")
    lr = cfg.train.lr_by_arm.get(
        kind, cfg.train.lr if is_quantum else cfg.train.lr_classical)
    opt = torch.optim.Adam(model.parameters(), lr=lr)

    buffer = ReservoirBuffer(cfg.train.replay_capacity, cfg.encode.feature_dim, seed)
    anchor = FisherAnchor(cfg.train.lambda_omega)
    region = TrustRegion(cfg.train.trust_region_delta)
    drift = DriftDetector(cfg.train.drift_threshold)

    # A fixed probe batch, chosen once and never changed — Tr(F) is only comparable
    # across tasks and across arms if the probe set is identical throughout.
    probe_X = np.concatenate([e.X[:8] for e in stream.examples[:16]])[: cfg.train.fim_probe_size]
    probe_y = np.concatenate([e.y[:8] for e in stream.examples[:16]])[: cfg.train.fim_probe_size]

    A = np.zeros((T, T), dtype=np.float32)
    result = ContinualResult(arm=arm, A=A)
    last_tr_f = 0.0
    bce = torch.nn.BCELoss()

    for t, task in enumerate(tasks):
        Xt, yt = _flatten_task(task, cfg.train.n_negatives, rng)
        theta0 = region.start_task(model)

        if getattr(model, "trainable", True):
            for _ in range(cfg.train.epochs_per_task):
                for xb, yb in _batches(Xt, yt, cfg.train.batch_size, rng):
                    xr, yr = buffer.sample(int(len(xb) * cfg.train.replay_ratio /
                                               (1 - cfg.train.replay_ratio)))
                    if len(xr):
                        xb = np.concatenate([xb, xr])
                        yb = np.concatenate([yb, yr])
                    opt.zero_grad(set_to_none=True)
                    pred = model(xb).clamp(1e-6, 1 - 1e-6)
                    loss = bce(pred, torch.as_tensor(yb, dtype=torch.float32)) \
                        + anchor.penalty(model)
                    loss.backward()
                    grad_norm = model.grad_norm()
                    opt.step()
            region.apply(model, theta0)
            model.wrap_angles()
        else:
            grad_norm = 0.0

        buffer.add(Xt, yt, t)

        with torch.no_grad():
            scores = model(Xt).cpu().numpy()
        psi, fired = drift.update(scores)

        if fired or (t % fisher_every == 0):
            fdiag = (qfim_diagonal(model, probe_X, max_examples=8, n_sample=96)
                     if is_quantum else _classical_fisher(model, probe_X, probe_y))
            anchor.checkpoint(model, fdiag)

        # ---- evaluation: current task, plus a sample of every past task -------
        A[t, t] = _eval(model, task)
        past = rng.choice(t, size=min(eval_past, t), replace=False) if t else []
        for j in past:
            A[t, int(j)] = _eval(model, tasks[int(j)][:eval_probes])
        for j in range(t):
            if A[t, j] == 0 and j not in past:      # carry the last known value forward
                A[t, j] = A[t - 1, j]

        if on_task_end is not None:
            on_task_end(t, model)

        if t % eval_every == 0:
            # Tr(F) costs one backward pass per probe example, so it is refreshed on a
            # coarser grid than accuracy — it is a slow-moving quantity and the curve is
            # unchanged, while the loop gets several times faster.
            if t % fim_log_every == 0:
                last_tr_f = (fim_trace(model, probe_X, probe_y, max_examples=16)
                             if is_quantum else _classical_fim_trace(model, probe_X, probe_y))
            tr_f = last_tr_f
            log.log(task_id=t, arm=arm, phase="phase6", acc=float(A[t, t]),
                    loss=float(loss.detach()) if getattr(model, "trainable", True) else 0.0,
                    tr_fim=float(tr_f), grad_norm=float(grad_norm),
                    weight_norm=model.theta_norm(True),
                    weight_norm_unwrapped=model.theta_norm(False),
                    anchor_pull=anchor.pull(model), drift_psi=float(psi),
                    checkpoint=int(fired), clip_rate=region.clip_rate, lr=lr)
            result.history.append({"task": t, "acc": float(A[t, t]), "tr_fim": float(tr_f)})
            if verbose and t % max(1, T // 20) == 0:
                print(f"   [{arm}] task {t:4d}/{T}  acc {A[t, t]:.3f}  "
                      f"Tr(F) {tr_f:.3e}  ‖θ‖ {model.theta_norm(True):.2f}  "
                      f"pull {anchor.pull(model):.3f}")

    result.A = A
    log.note(f"continual_{arm}", {
        "bwt": result.bwt(), "fwt": result.fwt(),
        "plasticity_index": result.plasticity_index(),
        "final_average": result.final_average(),
        "clip_rate": region.clip_rate, "drift_events": drift.n_fired,
        "checkpoints": anchor.n_checkpoints, "buffer_tasks": buffer.task_coverage(),
        "n_params": int(model.n_circuit_params),
    })
    np.save(log.run_dir / f"cl_matrix_{arm}.npy", A)
    return result


def _flatten_task(task, n_neg: int, rng):
    Xs, ys = [], []
    for ex in task:
        pos = np.flatnonzero(ex.y > 0)
        neg = np.flatnonzero(ex.y == 0)
        if len(neg) > n_neg:
            hard = neg[ex.is_stale[neg] > 0][: n_neg // 2]
            rest = rng.choice(neg, n_neg - len(hard), replace=False)
            neg = np.concatenate([hard, rest])
        idx = np.concatenate([pos, neg])
        Xs.append(ex.X[idx])
        ys.append(ex.y[idx])
    return np.concatenate(Xs), np.concatenate(ys)


def _eval(model, task) -> float:
    with torch.no_grad():
        return evaluate_probes(lambda ex: model(ex.X).cpu().numpy(), task, ks=(1,))["mrr"]


def _classical_fisher(model, X, y) -> np.ndarray:
    """Empirical FIM diagonal for the Euclidean arms — ordinary EWC."""
    acc = torch.zeros_like(model.theta)
    for i in range(min(len(X), 24)):
        model.zero_grad(set_to_none=True)
        p = model(X[i:i + 1]).clamp(1e-6, 1 - 1e-6)
        yy = torch.as_tensor(y[i:i + 1], dtype=torch.float32)
        (yy * torch.log(p) + (1 - yy) * torch.log(1 - p)).sum().backward()
        if model.theta.grad is not None:
            acc += model.theta.grad.detach() ** 2
    return (acc / 24).cpu().numpy()


def _classical_fim_trace(model, X, y) -> float:
    return float(_classical_fisher(model, X, y).sum())
