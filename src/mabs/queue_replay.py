"""Queue / commit-carry replay utilities for offline mixture analysis.

Replays precomputed matched-edge arrays through the post-matching kernels
so mixture statistics and empty-output fractions can be reproduced without
re-running Stim/PyMatching. Also exposes a lightweight FIFO work-queue
model used when interpreting offered load ρ = E[τ_B] / (k C τ_c).
"""

from __future__ import annotations

from collections import deque
from dataclasses import dataclass
from typing import Callable, Literal, Sequence
import numpy as np

from mabs.kernels import CommitResult, get_kernel, kernel_vector

KernelName = Literal["naive", "loop", "vector", "auto"]


def replay_commits(
    edge_list: Sequence[np.ndarray],
    active_list: Sequence[np.ndarray],
    commit_regions: Sequence[tuple[int, int]],
    kernel: KernelName = "vector",
    auto_threshold: int = 64,
) -> list[CommitResult]:
    """Replay a commit/carry kernel over precomputed edges / actives / regions."""
    if not (len(edge_list) == len(active_list) == len(commit_regions)):
        raise ValueError("edge_list, active_list, commit_regions must align")
    fn: Callable = get_kernel(kernel, auto_threshold=auto_threshold)
    out: list[CommitResult] = []
    for edges, active, (lo, hi) in zip(edge_list, active_list, commit_regions):
        e = np.asarray(edges, dtype=np.int64)
        if e.size == 0:
            e = np.zeros((0, 2), dtype=np.int64)
        elif e.ndim == 1:
            e = e.reshape(-1, 2)
        a = np.asarray(active, dtype=np.int64)
        out.append(fn(e, a, lo, hi))
    return out


def replay_vector_commits(
    edge_list: Sequence[np.ndarray],
    active_list: Sequence[np.ndarray],
    commit_regions: Sequence[tuple[int, int]],
) -> list[CommitResult]:
    """Replay the vector kernel (compat wrapper)."""
    return replay_commits(edge_list, active_list, commit_regions, kernel="vector")


def batch_K_and_empty(results: Sequence[CommitResult]) -> tuple[np.ndarray, np.ndarray]:
    """Extract K and empty_output flags from replay results."""
    K = np.array([r.K for r in results], dtype=np.int64)
    empty = np.array([r.empty_output for r in results], dtype=bool)
    return K, empty


@dataclass
class QueueReplayStats:
    """Simple offered-load / backlog summary from a work-queue replay."""

    n_jobs: int
    mean_service_ns: float
    mean_wait_ns: float
    max_backlog: int
    offered_load: float


def replay_work_queue(
    service_times_ns: Sequence[float],
    *,
    interarrival_ns: float,
    k_workers: int = 1,
) -> QueueReplayStats:
    """Deterministic multi-server FIFO replay for mean wait / backlog.

    Jobs arrive every ``interarrival_ns`` (≈ C * τ_c). Each job has a measured
    service demand τ_B. With k identical workers the offered load is
    ρ = E[τ_B] / (k * interarrival).
    """
    times = [float(t) for t in service_times_ns]
    n = len(times)
    if n == 0:
        return QueueReplayStats(0, 0.0, 0.0, 0, float("nan"))
    if interarrival_ns <= 0:
        raise ValueError("interarrival_ns must be positive")
    if k_workers < 1:
        raise ValueError("k_workers must be >= 1")

    # Worker free-at times
    free_at = [0.0] * k_workers
    waits: list[float] = []
    backlog = 0
    max_backlog = 0
    pending: deque[tuple[float, float]] = deque()  # (arrival, service)

    for i, svc in enumerate(times):
        arrival = i * interarrival_ns
        # Drain completed work before this arrival
        while pending and min(free_at) <= arrival:
            # Advance the soonest free worker
            w = min(range(k_workers), key=lambda j: free_at[j])
            if free_at[w] > arrival:
                break
            _arr, _svc = pending.popleft()
            start = max(free_at[w], _arr)
            free_at[w] = start + _svc
            backlog = len(pending)

        pending.append((arrival, svc))
        backlog = len(pending)
        max_backlog = max(max_backlog, backlog)

        # Assign immediately if a worker is free
        w = min(range(k_workers), key=lambda j: free_at[j])
        if free_at[w] <= arrival and pending:
            _arr, _svc = pending.popleft()
            start = max(free_at[w], _arr)
            waits.append(start - _arr)
            free_at[w] = start + _svc
            backlog = len(pending)

    # Drain remaining queue in arrival order
    while pending:
        w = min(range(k_workers), key=lambda j: free_at[j])
        _arr, _svc = pending.popleft()
        start = max(free_at[w], _arr)
        waits.append(start - _arr)
        free_at[w] = start + _svc

    mean_service = float(np.mean(times))
    mean_wait = float(np.mean(waits)) if waits else 0.0
    rho = mean_service / (k_workers * interarrival_ns)
    return QueueReplayStats(
        n_jobs=n,
        mean_service_ns=mean_service,
        mean_wait_ns=mean_wait,
        max_backlog=max_backlog,
        offered_load=rho,
    )


# Re-export vector kernel helper for callers that only need the fast path
__all__ = [
    "replay_commits",
    "replay_vector_commits",
    "batch_K_and_empty",
    "QueueReplayStats",
    "replay_work_queue",
    "kernel_vector",
]
