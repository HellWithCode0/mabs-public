"""Local decoders for small syndrome clusters (exact for tiny cases)."""

from __future__ import annotations

from typing import Any, Dict, List, Optional, Sequence, Tuple
import heapq
import numpy as np

from mabs.v3.cluster import Cluster, DetectorGraph, _get_graph_cache


def boundary_path_edges(graph: DetectorGraph, defect: int) -> Optional[List[Tuple[int, int]]]:
    if defect not in graph.bound_prev:
        return None
    n = graph.n_local
    edges: List[Tuple[int, int]] = []
    x = defect
    guard = 0
    while x != -1 and guard < n + 2:
        guard += 1
        p = graph.bound_prev.get(x)
        if p is None:
            return None
        if p == -1:
            edges.append((x, n))
            break
        edges.append((x, p))
        x = p
    else:
        return None
    return edges


def dijkstra_to_targets(
    graph: DetectorGraph,
    source: int,
    targets: set,
    *,
    allow_boundary: bool = True,
    node_limit: Optional[set] = None,
) -> Tuple[Optional[int], float, Dict[int, Optional[int]]]:
    dist: Dict[int, float] = {source: 0.0}
    prev: Dict[int, Optional[int]] = {source: None}
    pq: List[Tuple[float, int]] = [(0.0, source)]
    while pq:
        du, u = heapq.heappop(pq)
        if du != dist.get(u):
            continue
        if u != source and u in targets:
            return u, du, prev
        for v, w in graph.adj[u]:
            if v == -1:
                if allow_boundary and u != source:
                    return -1, du + w, {**prev, -1: u}
                continue
            if node_limit is not None and v not in node_limit and v not in targets:
                continue
            nd = du + w
            if nd < dist.get(v, 1e300):
                dist[v] = nd
                prev[v] = u
                heapq.heappush(pq, (nd, v))
    return None, float("inf"), prev


def decode_one_defect(graph: DetectorGraph, defect: int) -> Optional[np.ndarray]:
    edges = boundary_path_edges(graph, defect)
    if edges is None:
        return None
    return np.asarray(edges, dtype=np.int64).reshape(-1, 2)


def decode_two_defects(graph: DetectorGraph, a: int, b: int) -> Optional[np.ndarray]:
    """Exact MWPM on two defects: match-together vs each-to-boundary."""
    tgt, w_ab, prev_ab = dijkstra_to_targets(graph, a, {b}, allow_boundary=False)
    wa = graph.bound_dist.get(a)
    wb = graph.bound_dist.get(b)
    best: Optional[List[Tuple[int, int]]] = None
    best_w = float("inf")

    if tgt == b and w_ab < best_w:
        edges = []
        x: Optional[int] = b
        while x is not None and x != a:
            p = prev_ab.get(x)
            if p is None:
                edges = []
                break
            edges.append((x, p))
            x = p
        if edges:
            best = edges
            best_w = w_ab

    if wa is not None and wb is not None and (wa + wb) < best_w:
        ea = boundary_path_edges(graph, a)
        eb = boundary_path_edges(graph, b)
        if ea is not None and eb is not None:
            best = ea + eb

    if best is None:
        return None
    return np.asarray(best, dtype=np.int64).reshape(-1, 2)


def peel_cluster(
    graph: DetectorGraph,
    cluster: Cluster,
) -> Optional[np.ndarray]:
    defects = list(cluster.defects)
    if not defects:
        return np.zeros((0, 2), dtype=np.int64)
    if len(defects) == 1:
        return decode_one_defect(graph, defects[0])
    if len(defects) == 2:
        return decode_two_defects(graph, defects[0], defects[1])

    rem = set(defects)
    all_edges: List[Tuple[int, int]] = []
    node_limit = set(cluster.nodes) if cluster.nodes else None
    while rem:
        s = next(iter(rem))
        others = rem - {s}
        tgt, _w, prev = dijkstra_to_targets(
            graph, s, others, allow_boundary=True, node_limit=node_limit
        )
        if tgt is None:
            return None
        if tgt == -1:
            last = prev.get(-1)
            if last is None:
                return None
            edges = []
            x: Optional[int] = last
            while x is not None and x != s:
                p = prev.get(x)
                if p is None:
                    return None
                edges.append((x, p))
                x = p
            edges.append((last, graph.n_local))
            all_edges.extend(edges)
            rem.discard(s)
        else:
            edges = []
            x = tgt
            while x is not None and x != s:
                p = prev.get(x)
                if p is None:
                    return None
                edges.append((x, p))
                x = p
            all_edges.extend(edges)
            rem.discard(s)
            rem.discard(tgt)
    if not all_edges:
        return np.zeros((0, 2), dtype=np.int64)
    return np.asarray(all_edges, dtype=np.int64).reshape(-1, 2)


def local_decode_clusters(
    wm: Any,
    buf: np.ndarray,
    clusters: Sequence[Cluster],
    *,
    max_cluster_size: int = 2,
) -> Optional[np.ndarray]:
    if not clusters:
        return np.zeros((0, 2), dtype=np.int64)
    graph = _get_graph_cache(wm)
    parts: List[np.ndarray] = []
    for c in clusters:
        if c.size > max_cluster_size:
            return None
        edges = peel_cluster(graph, c)
        if edges is None:
            return None
        if edges.size:
            parts.append(edges)
    if not parts:
        return np.zeros((0, 2), dtype=np.int64)
    return np.vstack(parts)


def local_decode_few_defects(wm: Any, buf: np.ndarray, defects: np.ndarray) -> Optional[np.ndarray]:
    """Exact local MWPM for 1–2 defects without clustering overhead."""
    nd = int(defects.size)
    if nd == 0:
        return np.zeros((0, 2), dtype=np.int64)
    graph = _get_graph_cache(wm)
    if nd == 1:
        return decode_one_defect(graph, int(defects[0]))
    if nd == 2:
        return decode_two_defects(graph, int(defects[0]), int(defects[1]))
    return None


def blossom_edges_only(wm: Any, buf: np.ndarray) -> Tuple[np.ndarray, float]:
    """Single Sparse Blossom call — edges only (no second weight decode)."""
    try:
        pairs = wm.matching.decode_to_edges_array(buf)
    except Exception:
        return np.zeros((0, 2), dtype=np.int64), 0.0
    if pairs is None or len(pairs) == 0:
        return np.zeros((0, 2), dtype=np.int64), 0.0
    arr = np.asarray(pairs, dtype=np.int64)
    if arr.ndim == 1:
        arr = arr.reshape(-1, 2)
    return arr, float(arr.shape[0])
