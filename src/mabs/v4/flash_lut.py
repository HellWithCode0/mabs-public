"""FLASH CommitAction LUTs — boundary + pair keyed by raw detector ints.

Default CASCADE hot path (v4.2): store CommitActions, not edge lists.
Prewarm at Matching/window DEM build; O(1) lookup on K<=2.
Optional numba defect scan with numpy/Python fallback.
"""
from __future__ import annotations

from typing import Any, Dict, List, Optional, Tuple
import numpy as np

from mabs.v3.cluster import DetectorGraph, _get_graph_cache
from mabs.v3.local_decode import decode_one_defect, decode_two_defects
from mabs.v4.commit_action import CommitAction, edges_to_commit_action

try:
    from numba import njit

    @njit
    def _njit_scan_defects(buf: np.ndarray, n: int, out: np.ndarray) -> int:
        k = 0
        for i in range(n):
            if buf[i]:
                out[k] = i
                k += 1
        return k

    @njit
    def _njit_scan_defects_cap(buf: np.ndarray, n: int, out: np.ndarray, cap: int) -> int:
        """Record up to ``cap`` defect ids; stop once k reaches cap (FLASH K>=3)."""
        k = 0
        for i in range(n):
            if buf[i]:
                if k < cap:
                    out[k] = i
                k += 1
                if k >= cap:
                    return k
        return k

    @njit
    def _njit_buf_any(buf: np.ndarray, n: int) -> bool:
        for i in range(n):
            if buf[i]:
                return True
        return False

    _HAS_NUMBA = True
except Exception:  # pragma: no cover - graceful fallback
    _HAS_NUMBA = False
    _njit_scan_defects = None  # type: ignore
    _njit_scan_defects_cap = None  # type: ignore
    _njit_buf_any = None  # type: ignore


def has_numba() -> bool:
    return bool(_HAS_NUMBA)


def extract_defects(
    buf: np.ndarray,
    n: int,
    out: np.ndarray,
    *,
    prefer_numba: bool = True,
    cap: Optional[int] = None,
) -> int:
    """One-pass defect extract into preallocated ``out``; return K.

    If ``cap`` is set (FLASH uses 3), stop after finding ``cap`` defects —
    enough to decide K=0/1/2 vs K>=3 without scanning the rest.
    Uses numba when importable; else numpy.
    """
    if prefer_numba and _HAS_NUMBA:
        if cap is not None:
            return int(_njit_scan_defects_cap(buf, n, out, int(cap)))
        return int(_njit_scan_defects(buf, n, out))
    if cap is not None:
        c = int(cap)
        if c >= 1:
            # One C-level pass, then keep the first `cap`. The early-exit Python
            # loop this replaces indexed the buffer element by element and, on
            # a sparse syndrome, walked most of it: 10.4 us per window at d=5
            # against 1.8 us here, more than the blossom call it routed to.
            fired = np.flatnonzero(buf[:n])
            k = int(fired.size)
            take = k if k < c else c
            if take:
                out[:take] = fired[:take]
            return take
        # cap < 1 is never used by the router; keep the historical semantics
        # (and the numba kernel's) exactly rather than inventing new ones.
        k = 0
        for i in range(n):
            if buf[i]:
                if k < c:
                    out[k] = i
                k += 1
                if k >= c:
                    return k
        return k
    fired = np.flatnonzero(buf[:n])
    k = int(fired.size)
    if k:
        out[:k] = fired
    return k


def buf_nonempty(buf: np.ndarray, n: int, *, prefer_numba: bool = True) -> bool:
    """Fast emptiness probe (no allocation)."""
    if prefer_numba and _HAS_NUMBA:
        return bool(_njit_buf_any(buf, n))
    # count_nonzero is cheaper than np.any on uint8 windows here
    return bool(np.count_nonzero(buf[:n]))


PairKey = Tuple[int, int]


class FlashCommitLUT:
    """Per-window-matcher LUT: boundary[det] + pair[(a,b)] → CommitAction.

    Topology-safe by construction: one LUT instance per Matching / DetectorGraph
    (attached to the graph cache). Keys are raw local detector ints.
    """

    def __init__(self, wm: Any, graph: DetectorGraph):
        self.wm = wm
        self.graph = graph
        self.n_local = int(graph.n_local)
        self.boundary: List[Optional[CommitAction]] = [None] * self.n_local
        self.pairs: Dict[PairKey, CommitAction] = {}
        self.hits = 0
        self.misses = 0
        self.fills = 0
        self._defect_buf = np.empty(self.n_local, dtype=np.int64)
        self._prewarmed_boundary = False
        self._prewarmed_pairs = False

    @staticmethod
    def pair_key(a: int, b: int) -> PairKey:
        aa, bb = int(a), int(b)
        return (aa, bb) if aa <= bb else (bb, aa)

    def prewarm_boundary(self) -> int:
        """Precompute CommitAction for every single-defect → boundary path."""
        n = 0
        g = self.graph
        wm = self.wm
        for d in range(self.n_local):
            edges = decode_one_defect(g, d)
            if edges is None:
                self.boundary[d] = None
            else:
                self.boundary[d] = edges_to_commit_action(edges, wm)
                n += 1
        self._prewarmed_boundary = True
        return n

    def prewarm_pairs(self, radius: int) -> int:
        """Precompute pair CommitActions for graph hop-distance ≤ radius."""
        if radius <= 0:
            return 0
        g = self.graph
        wm = self.wm
        n_fill = 0
        adj = g.adj
        n_local = self.n_local
        for src in range(n_local):
            # BFS up to radius
            dist = {src: 0}
            q = [src]
            qi = 0
            while qi < len(q):
                u = q[qi]
                qi += 1
                du = dist[u]
                if du >= radius:
                    continue
                for v, _w in adj[u]:
                    if v < 0 or v >= n_local:
                        continue
                    if v in dist:
                        continue
                    dist[v] = du + 1
                    q.append(v)
            for dst, hop in dist.items():
                if dst <= src or hop < 1 or hop > radius:
                    continue
                key = (src, dst)
                if key in self.pairs:
                    continue
                edges = decode_two_defects(g, src, dst)
                if edges is None:
                    continue
                self.pairs[key] = edges_to_commit_action(edges, wm)
                n_fill += 1
        self._prewarmed_pairs = True
        self.fills += n_fill
        return n_fill

    def get_boundary(self, det: int) -> Optional[CommitAction]:
        d = int(det)
        if 0 <= d < self.n_local:
            act = self.boundary[d]
            if act is not None:
                self.hits += 1
                return act
        self.misses += 1
        edges = decode_one_defect(self.graph, d)
        if edges is None:
            return None
        act = edges_to_commit_action(edges, self.wm)
        if 0 <= d < self.n_local:
            self.boundary[d] = act
            self.fills += 1
        return act

    def get_pair(self, a: int, b: int) -> Optional[CommitAction]:
        key = self.pair_key(a, b)
        act = self.pairs.get(key)
        if act is not None:
            self.hits += 1
            return act
        self.misses += 1
        return None

    def fill_pair_from_edges(self, a: int, b: int, edges: np.ndarray) -> CommitAction:
        act = edges_to_commit_action(edges, self.wm)
        self.pairs[self.pair_key(a, b)] = act
        self.fills += 1
        return act

    def fill_pair_compute(self, a: int, b: int) -> Optional[CommitAction]:
        edges = decode_two_defects(self.graph, int(a), int(b))
        if edges is None:
            return None
        return self.fill_pair_from_edges(a, b, edges)

    @property
    def size(self) -> int:
        return sum(1 for a in self.boundary if a is not None) + len(self.pairs)


def get_flash_lut(wm: Any) -> FlashCommitLUT:
    """Get or build FlashCommitLUT attached to the window's DetectorGraph."""
    graph = _get_graph_cache(wm)
    lut = getattr(graph, "_flash_lut", None)
    if lut is None or lut.graph is not graph or lut.wm is not wm:
        lut = FlashCommitLUT(wm, graph)
        setattr(graph, "_flash_lut", lut)
    return lut


def prewarm_flash_lut(wm: Any, *, pair_radius: int = 0) -> FlashCommitLUT:
    lut = get_flash_lut(wm)
    if not lut._prewarmed_boundary:
        lut.prewarm_boundary()
    if pair_radius > 0 and not lut._prewarmed_pairs:
        lut.prewarm_pairs(pair_radius)
    return lut
