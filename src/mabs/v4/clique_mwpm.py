"""Exact MWPM on a K-node complete graph (2 ≤ K ≤ 6) with boundary.

Pairwise weights from Dijkstra on the existing DetectorGraph; boundary via
precomputed bound_dist / bound_paths. Pure-Python enumeration of matchings.
Falls through (returns None) when distance setup is incomplete.
"""

from __future__ import annotations

from itertools import combinations
from typing import Dict, List, Optional, Sequence, Tuple
import heapq
import numpy as np

from mabs.v3.cluster import DetectorGraph, _get_graph_cache
from mabs.v3.local_decode import (
    boundary_path_edges,
    decode_one_defect,
    decode_two_defects,
    syndrome_cleared_by_edges,
)


def _dijkstra_path(
    graph: DetectorGraph,
    source: int,
    target: int,
) -> Tuple[float, List[Tuple[int, int]]]:
    """Shortest path source→target (no boundary). Returns (weight, edges)."""
    if source == target:
        return 0.0, []
    dist: Dict[int, float] = {source: 0.0}
    prev: Dict[int, Optional[int]] = {source: None}
    pq: List[Tuple[float, int]] = [(0.0, source)]
    while pq:
        du, u = heapq.heappop(pq)
        if du != dist.get(u):
            continue
        if u == target:
            break
        for v, w in graph.adj[u]:
            if v < 0:
                continue
            nd = du + w
            if nd < dist.get(v, 1e300):
                dist[v] = nd
                prev[v] = u
                heapq.heappush(pq, (nd, v))
    if target not in dist:
        return float("inf"), []
    edges: List[Tuple[int, int]] = []
    x: Optional[int] = target
    while x is not None and x != source:
        p = prev.get(x)
        if p is None:
            return float("inf"), []
        edges.append((x, p))
        x = p
    return float(dist[target]), edges


def _pair_weight_and_edges(
    graph: DetectorGraph,
    a: int,
    b: int,
    cache: Dict[Tuple[int, int], Tuple[float, List[Tuple[int, int]]]],
) -> Tuple[float, List[Tuple[int, int]]]:
    key = (a, b) if a < b else (b, a)
    if key in cache:
        return cache[key]
    w, edges = _dijkstra_path(graph, a, b)
    cache[key] = (w, edges)
    return w, edges


def _perfect_matchings(n: int) -> List[List[Tuple[int, int]]]:
    """All perfect matchings on {0..n-1}, n even."""
    if n == 0:
        return [[]]
    if n % 2:
        return []

    def _rec(remaining: List[int]) -> List[List[Tuple[int, int]]]:
        if not remaining:
            return [[]]
        a = remaining[0]
        out: List[List[Tuple[int, int]]] = []
        for i in range(1, len(remaining)):
            b = remaining[i]
            rest = remaining[1:i] + remaining[i + 1 :]
            for m in _rec(rest):
                out.append([(a, b)] + m)
        return out

    return _rec(list(range(n)))


def _enumerate_boundary_matchings(k: int) -> List[Tuple[List[Tuple[int, int]], List[int]]]:
    """Yield (pair_list, singles_to_boundary) covering all K defects."""
    results: List[Tuple[List[Tuple[int, int]], List[int]]] = []
    nodes = list(range(k))
    for r in range(0, k + 1):
        if (k - r) % 2 != 0:
            continue
        for singles in combinations(nodes, r):
            paired = [i for i in nodes if i not in singles]
            for pm in _perfect_matchings(len(paired)):
                pairs = [(paired[a], paired[b]) for a, b in pm]
                results.append((pairs, list(singles)))
    return results


def clique_mwpm_edges(
    graph: DetectorGraph,
    defects: Sequence[int],
    *,
    max_k: int = 6,
) -> Optional[np.ndarray]:
    """Exact MWPM on ≤ max_k defects using complete-graph enumeration + paths.

    Returns local edge array (boundary as n_local) or None to fall through.
    """
    dets = [int(d) for d in defects]
    k = len(dets)
    if k == 0:
        return np.zeros((0, 2), dtype=np.int64)
    if k > max_k:
        return None
    if k == 1:
        return decode_one_defect(graph, dets[0])
    if k == 2:
        return decode_two_defects(graph, dets[0], dets[1])

    pair_cache: Dict[Tuple[int, int], Tuple[float, List[Tuple[int, int]]]] = {}
    W = np.full((k, k), 1e300)
    path_edges: Dict[Tuple[int, int], List[Tuple[int, int]]] = {}
    for i in range(k):
        for j in range(i + 1, k):
            w, edges = _pair_weight_and_edges(graph, dets[i], dets[j], pair_cache)
            if w < 1e300:
                W[i, j] = W[j, i] = w
                path_edges[(i, j)] = edges
                path_edges[(j, i)] = edges

    bdv = np.full(k, 1e300)
    be: List[Optional[List[Tuple[int, int]]]] = [None] * k
    for i, d in enumerate(dets):
        pe = boundary_path_edges(graph, d)
        if pe is None:
            return None
        be[i] = pe
        bd = graph.bound_dist.get(d)
        if bd is None:
            return None
        bdv[i] = float(bd)

    best_w = 1e300
    best_parts: Optional[List[List[Tuple[int, int]]]] = None

    for pairs, singles in _enumerate_boundary_matchings(k):
        w = 0.0
        parts: List[List[Tuple[int, int]]] = []
        ok = True
        for a, b in pairs:
            wij = W[a, b]
            if wij >= 1e300:
                ok = False
                break
            w += wij
            parts.append(path_edges[(a, b)])
        if not ok:
            continue
        for s in singles:
            if be[s] is None or bdv[s] >= 1e300:
                ok = False
                break
            w += float(bdv[s])
            parts.append(be[s])  # type: ignore[arg-type]
        if not ok:
            continue
        if w < best_w:
            best_w = w
            best_parts = parts

    if best_parts is None:
        return None
    all_edges: List[Tuple[int, int]] = []
    for part in best_parts:
        all_edges.extend(part)
    if not all_edges:
        return np.zeros((0, 2), dtype=np.int64)
    return np.asarray(all_edges, dtype=np.int64).reshape(-1, 2)


def clique_decode_window(
    wm,
    buf: np.ndarray,
    fired: np.ndarray,
    *,
    clique_cap: int = 6,
    check_syndrome: bool = True,
) -> Optional[np.ndarray]:
    """Decode a window with ≤ clique_cap defects via clique MWPM."""
    nd = int(fired.size)
    if nd == 0:
        return np.zeros((0, 2), dtype=np.int64)
    if nd > clique_cap:
        return None
    graph = _get_graph_cache(wm)
    edges = clique_mwpm_edges(graph, fired.tolist(), max_k=clique_cap)
    if edges is None:
        return None
    if check_syndrome and not syndrome_cleared_by_edges(wm.n_local, fired, edges):
        return None
    return edges
