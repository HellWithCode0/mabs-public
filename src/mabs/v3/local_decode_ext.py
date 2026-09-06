"""v3.1 extensions: induced-subgraph local decode + syndrome check."""
from __future__ import annotations

from typing import Dict, List, Optional, Tuple, Any
import numpy as np

from mabs.v3.cluster import DetectorGraph, _get_graph_cache
from mabs.v3.union_find import fired_induced_components
from mabs.v3.local_decode import (
    boundary_path_edges,
    decode_one_defect,
    decode_two_defects,
)

def boundary_path_array(graph: DetectorGraph, defect: int) -> Optional[np.ndarray]:
    if getattr(graph, "bound_paths", None) is not None and 0 <= defect < graph.n_local:
        return graph.bound_paths[defect]
    edges = boundary_path_edges(graph, defect)
    if edges is None:
        return None
    return np.asarray(edges, dtype=np.int64).reshape(-1, 2)

def decode_two_defects_fast(
    graph: DetectorGraph,
    a: int,
    b: int,
    edge_w: Optional[Dict[Tuple[int, int], float]] = None,
) -> Optional[np.ndarray]:
    key = (a, b) if a < b else (b, a)
    w_ab = float("inf")
    if edge_w is not None and key in edge_w:
        w_ab = edge_w[key]
    else:
        for v, w in graph.adj[a]:
            if v == b:
                w_ab = float(w)
                break
    wa = graph.bound_dist.get(a, float("inf"))
    wb = graph.bound_dist.get(b, float("inf"))
    if w_ab <= wa + wb and w_ab < float("inf"):
        return np.asarray([(a, b)], dtype=np.int64)
    pa = boundary_path_array(graph, a)
    pb = boundary_path_array(graph, b)
    if pa is None or pb is None:
        return None
    return np.vstack([pa, pb])

def decode_comp_induced(
    graph: DetectorGraph,
    comp: List[int],
    edge_w: Dict[Tuple[int, int], float],
) -> Optional[np.ndarray]:
    k = len(comp)
    if k == 0:
        return np.zeros((0, 2), dtype=np.int64)
    if k == 1:
        return decode_one_defect(graph, comp[0])
    if k == 2:
        return decode_two_defects_fast(graph, comp[0], comp[1], edge_w)
    if k > 4:
        return None
    be: List[np.ndarray] = []
    bdv: List[float] = []
    for d in comp:
        p = boundary_path_array(graph, d)
        if p is None:
            return None
        be.append(p)
        bdv.append(float(graph.bound_dist.get(d, 1e300)))
    idx = {d: i for i, d in enumerate(comp)}
    W = np.full((k, k), 1e300)
    for (a, b), w in edge_w.items():
        if a in idx and b in idx:
            i, j = idx[a], idx[b]
            W[i, j] = W[j, i] = w
    best: Optional[List[np.ndarray]] = None
    best_w = 1e300
    def consider(parts: List[np.ndarray], w: float) -> None:
        nonlocal best, best_w
        if w < best_w:
            best_w = w
            best = parts
    if k == 3:
        consider(be, sum(bdv))
        for i in range(3):
            others = [j for j in range(3) if j != i]
            a, b = others
            if W[a, b] < 1e300:
                pair = np.asarray([(comp[a], comp[b])], dtype=np.int64)
                consider([be[i], pair], bdv[i] + W[a, b])
    else:
        from itertools import combinations
        consider(be, sum(bdv))
        for a, b in combinations(range(4), 2):
            if W[a, b] >= 1e300:
                continue
            others = [i for i in range(4) if i not in (a, b)]
            pair = np.asarray([(comp[a], comp[b])], dtype=np.int64)
            consider([pair, be[others[0]], be[others[1]]], W[a, b] + bdv[others[0]] + bdv[others[1]])
        for (a, b), (c, d) in (((0, 1), (2, 3)), ((0, 2), (1, 3)), ((0, 3), (1, 2))):
            if W[a, b] >= 1e300 or W[c, d] >= 1e300:
                continue
            consider([
                np.asarray([(comp[a], comp[b])], dtype=np.int64),
                np.asarray([(comp[c], comp[d])], dtype=np.int64),
            ], W[a, b] + W[c, d])
    if best is None:
        return None
    return np.vstack(best)

def local_decode_induced(
    wm: Any,
    buf: np.ndarray,
    defects: np.ndarray,
    *,
    max_cluster: int = 4,
    max_defects: int = 8,
) -> Optional[np.ndarray]:
    nd = int(defects.size)
    if nd == 0:
        return np.zeros((0, 2), dtype=np.int64)
    if nd > max_defects:
        return None
    graph = _get_graph_cache(wm)
    if nd == 1:
        return decode_one_defect(graph, int(defects[0]))
    if nd == 2:
        return decode_two_defects(graph, int(defects[0]), int(defects[1]))
    comps, edge_w = fired_induced_components(graph, defects)
    if not comps:
        return None
    if any(len(c) > max_cluster for c in comps):
        return None
    parts: List[np.ndarray] = []
    for c in comps:
        edges = decode_comp_induced(graph, c, edge_w)
        if edges is None:
            return None
        if edges.size:
            parts.append(edges)
    if not parts:
        return np.zeros((0, 2), dtype=np.int64)
    return np.vstack(parts)

def syndrome_cleared_by_edges(
    n_local: int,
    defects: np.ndarray,
    edges: np.ndarray,
) -> bool:
    if defects.size == 0:
        return True
    syn = np.zeros(n_local + 1, dtype=np.uint8)
    for d in defects:
        di = int(d)
        if 0 <= di < n_local:
            syn[di] ^= 1
    arr = np.asarray(edges, dtype=np.int64).reshape(-1, 2)
    for u, v in arr:
        uu, vv = int(u), int(v)
        if 0 <= uu < n_local:
            syn[uu] ^= 1
        if 0 <= vv < n_local:
            syn[vv] ^= 1
    return not bool(np.any(syn[:n_local]))
