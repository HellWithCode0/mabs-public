"""Post-matching commit/carry kernels with identical semantics.

After matching, commit edges whose earliest endpoint falls in the commit
region [commit_lo, commit_hi); carry unresolved syndrome into the next
window. Four kernels share that contract:

* naive  — element-by-element Python
* loop   — tuned per-edge Python loop
* vector — NumPy / table-driven; early return when K == 0
* auto   — dispatch loop vs vector by K vs threshold (default 64)
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Callable, Literal, Sequence
import numpy as np

KernelName = Literal["naive", "loop", "vector", "auto"]


@dataclass
class CommitResult:
    """Outcome of a post-matching commit/carry pass."""

    committed_edges: np.ndarray  # shape (M, 2) int64 endpoints (or empty)
    carry_detectors: np.ndarray  # detector indices to carry forward
    K: int  # matched edge count presented to the kernel
    empty_output: bool  # True when no edges were committed


def _earliest(a: int, b: int) -> int:
    # Convention: -1 is a boundary node (always "earliest" if present)
    if a < 0:
        return b if b >= 0 else a
    if b < 0:
        return a
    return a if a <= b else b


def _commit_mask_python(
    edges: Sequence[tuple[int, int]] | np.ndarray,
    commit_lo: int,
    commit_hi: int,
) -> list[tuple[int, int]]:
    committed: list[tuple[int, int]] = []
    for e in edges:
        a, b = int(e[0]), int(e[1])
        earliest = _earliest(a, b)
        # Boundary-only edges (both < 0) are committed if commit region nonempty
        if earliest < 0:
            if commit_lo < commit_hi:
                committed.append((a, b))
            continue
        if commit_lo <= earliest < commit_hi:
            committed.append((a, b))
    return committed


def _carry_from_edges(
    all_detectors: np.ndarray,
    committed: Sequence[tuple[int, int]] | np.ndarray,
    commit_lo: int,
    commit_hi: int,
) -> np.ndarray:
    """Carry detectors whose layer index is >= commit_hi (unresolved future).

    Approximate paper semantics: detectors in the look-ahead / non-commit
    region remain for the next window; committed-region detectors that were
    matched are resolved and dropped.
    """
    if all_detectors.size == 0:
        return np.array([], dtype=np.int64)
    # Detectors with index >= commit_hi are outside the commit region
    carry = all_detectors[all_detectors >= commit_hi]
    return np.asarray(carry, dtype=np.int64)


def kernel_naive(
    edges: np.ndarray,
    active_detectors: np.ndarray,
    commit_lo: int,
    commit_hi: int,
) -> CommitResult:
    """Element-by-element commit/carry."""
    K = int(edges.shape[0]) if edges.ndim == 2 else 0
    if K == 0:
        carry = _carry_from_edges(active_detectors, [], commit_lo, commit_hi)
        return CommitResult(
            committed_edges=np.zeros((0, 2), dtype=np.int64),
            carry_detectors=carry,
            K=0,
            empty_output=True,
        )
    committed = []
    i = 0
    while i < K:
        a = int(edges[i, 0])
        b = int(edges[i, 1])
        earliest = _earliest(a, b)
        if earliest < 0:
            if commit_lo < commit_hi:
                committed.append((a, b))
        elif commit_lo <= earliest < commit_hi:
            committed.append((a, b))
        i += 1
    arr = np.asarray(committed, dtype=np.int64).reshape(-1, 2) if committed else np.zeros((0, 2), dtype=np.int64)
    carry = _carry_from_edges(active_detectors, committed, commit_lo, commit_hi)
    return CommitResult(arr, carry, K, empty_output=arr.shape[0] == 0)


def kernel_loop(
    edges: np.ndarray,
    active_detectors: np.ndarray,
    commit_lo: int,
    commit_hi: int,
) -> CommitResult:
    """Tuned per-edge Python loop (local bindings, pre-sized list)."""
    K = 0 if edges.size == 0 else int(edges.shape[0])
    if K == 0:
        carry = _carry_from_edges(active_detectors, [], commit_lo, commit_hi)
        return CommitResult(
            committed_edges=np.zeros((0, 2), dtype=np.int64),
            carry_detectors=carry,
            K=0,
            empty_output=True,
        )
    lo = commit_lo
    hi = commit_hi
    e0 = edges[:, 0]
    e1 = edges[:, 1]
    out: list[tuple[int, int]] = []
    append = out.append
    for i in range(K):
        a = int(e0[i])
        b = int(e1[i])
        if a < 0:
            earliest = b if b >= 0 else a
        elif b < 0:
            earliest = a
        else:
            earliest = a if a <= b else b
        if earliest < 0:
            if lo < hi:
                append((a, b))
        elif lo <= earliest < hi:
            append((a, b))
    arr = np.asarray(out, dtype=np.int64).reshape(-1, 2) if out else np.zeros((0, 2), dtype=np.int64)
    carry = _carry_from_edges(active_detectors, out, lo, hi)
    return CommitResult(arr, carry, K, empty_output=arr.shape[0] == 0)


def kernel_vector(
    edges: np.ndarray,
    active_detectors: np.ndarray,
    commit_lo: int,
    commit_hi: int,
) -> CommitResult:
    """NumPy / table-driven commit; early return when K == 0."""
    if edges.size == 0:
        carry = _carry_from_edges(active_detectors, [], commit_lo, commit_hi)
        return CommitResult(
            committed_edges=np.zeros((0, 2), dtype=np.int64),
            carry_detectors=carry,
            K=0,
            empty_output=True,
        )
    K = int(edges.shape[0])
    a = edges[:, 0].astype(np.int64, copy=False)
    b = edges[:, 1].astype(np.int64, copy=False)
    # earliest endpoint: treat negatives specially
    both_neg = (a < 0) & (b < 0)
    a_neg = a < 0
    b_neg = b < 0
    earliest = np.where(a_neg, np.where(b_neg, a, b), np.where(b_neg, a, np.minimum(a, b)))
    in_commit = (earliest >= commit_lo) & (earliest < commit_hi)
    in_commit = in_commit | (both_neg & (commit_lo < commit_hi))
    # also commit boundary-single edges whose finite endpoint is in region
    # (already covered by earliest logic when one side is boundary)
    mask = in_commit
    committed = edges[mask]
    if committed.size == 0:
        committed = np.zeros((0, 2), dtype=np.int64)
    else:
        committed = committed.astype(np.int64, copy=False)
    carry = _carry_from_edges(active_detectors, committed, commit_lo, commit_hi)
    return CommitResult(committed, carry, K, empty_output=committed.shape[0] == 0)


def kernel_auto(
    edges: np.ndarray,
    active_detectors: np.ndarray,
    commit_lo: int,
    commit_hi: int,
    threshold: int = 64,
) -> CommitResult:
    """Dispatch loop vs vector by matched-edge count K vs threshold."""
    K = 0 if edges.size == 0 else int(edges.shape[0])
    if K < threshold:
        return kernel_loop(edges, active_detectors, commit_lo, commit_hi)
    return kernel_vector(edges, active_detectors, commit_lo, commit_hi)


def get_kernel(
    name: KernelName,
    auto_threshold: int = 64,
) -> Callable[[np.ndarray, np.ndarray, int, int], CommitResult]:
    name = name.lower()  # type: ignore[assignment]
    if name == "naive":
        return kernel_naive
    if name == "loop":
        return kernel_loop
    if name == "vector":
        return kernel_vector
    if name == "auto":

        def _auto(edges, active, lo, hi, _th=auto_threshold):
            return kernel_auto(edges, active, lo, hi, threshold=_th)

        return _auto
    raise ValueError(f"unknown kernel {name!r}; expected naive|loop|vector|auto")
