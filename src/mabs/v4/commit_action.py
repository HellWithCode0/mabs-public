"""Commit-region actions for the streaming XOR / carry path.

A CommitAction is the *effect* of matching edges on (obs_mask, carry) for one
window — usable directly by the same machinery as ``commit_window_edges``,
without re-interpreting raw edge arrays on every ExactPatternCache hit.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, List, Optional, Sequence, Tuple
import numpy as np


@dataclass(frozen=True)
class CommitAction:
    """Precomputed commit-region effect.

    Attributes
    ----------
    obs_xor:
        Bitmask XORed into the shot observable accumulator.
    carry_globals:
        Global detector indices to XOR in the carry buffer (sorted unique).
    n_committed:
        Number of edges that touched the commit region (for metrics).
    """

    obs_xor: int = 0
    carry_globals: Tuple[int, ...] = ()
    n_committed: int = 0

    @staticmethod
    def empty() -> "CommitAction":
        return CommitAction(0, (), 0)


def edges_to_commit_action(edges: np.ndarray, wm: Any) -> CommitAction:
    """Compile matching edges → CommitAction (same rules as commit_window_edges)."""
    if edges is None or len(edges) == 0:
        return CommitAction.empty()
    arr = np.asarray(edges, dtype=np.int64)
    if arr.ndim == 1:
        arr = arr.reshape(-1, 2)
    if arr.size == 0:
        return CommitAction.empty()

    stride = wm.key_stride
    incom = wm.in_commit_list
    gnode = wm.gnode_list
    table = wm.obs_by_pair
    n_local = wm.n_local
    obs = 0
    n_committed = 0
    carry_set: List[int] = []
    for u, v in arr.tolist():
        u = int(u)
        v = int(v)
        if u < 0:
            u = n_local
        if v < 0:
            v = n_local
        if u > n_local:
            u = n_local
        if v > n_local:
            v = n_local
        cu = incom[u]
        cv = incom[v]
        if cu or cv:
            gu = gnode[u]
            gv = gnode[v]
            obs ^= int(table.get(gu * stride + gv, 0))
            n_committed += 1
            if cu != cv:
                carry_set.append(int(gv if cu else gu))
    carry_globals = tuple(sorted(set(carry_set)))
    return CommitAction(obs_xor=int(obs), carry_globals=carry_globals, n_committed=n_committed)


def apply_commit_action(action: CommitAction, carry: np.ndarray) -> Tuple[int, int]:
    """Apply a cached CommitAction into carry; return (obs_xor, n_committed)."""
    if action is None or (action.obs_xor == 0 and not action.carry_globals and action.n_committed == 0):
        return 0, 0
    for g in action.carry_globals:
        gi = int(g)
        if 0 <= gi < carry.size:
            carry[gi] ^= 1
    return int(action.obs_xor), int(action.n_committed)


def commit_edges_or_action(
    edges: Optional[np.ndarray],
    action: Optional[CommitAction],
    wm: Any,
    carry: np.ndarray,
    *,
    carry_forward: bool = True,
) -> Tuple[int, int]:
    del carry_forward
    if action is not None:
        return apply_commit_action(action, carry)
    from mabs.streaming_windows import commit_window_edges
    return commit_window_edges(edges, wm, carry, carry_forward=True)
