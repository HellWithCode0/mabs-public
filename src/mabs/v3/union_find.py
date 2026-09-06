"""Union-find helpers and fast fired-subgraph clustering for SLEM v3.1."""

from __future__ import annotations

from typing import Dict, List, Tuple
import numpy as np

from mabs.v3.cluster import DetectorGraph


def find_root(parent: Dict[int, int], x: int) -> int:
    while parent[x] != x:
        parent[x] = parent[parent[x]]
        x = parent[x]
    return x


def union(parent: Dict[int, int], a: int, b: int) -> None:
    ra, rb = find_root(parent, a), find_root(parent, b)
    if ra != rb:
        parent[rb] = ra


def fired_induced_components(
    graph: DetectorGraph,
    fired: np.ndarray,
) -> Tuple[List[List[int]], Dict[Tuple[int, int], float]]:
    """Connected components on the induced subgraph of fired detectors.

    Also returns undirected edge weights among fired pairs (min weight if multi).
    Radius-0 clustering: two defects share a cluster iff a path of *fired*
    detectors connects them (no growth through quiescent nodes).
    """
    if fired.size == 0:
        return [], {}
    fds = [int(x) for x in np.asarray(fired, dtype=np.int64).tolist()]
    fset = set(fds)
    parent = {d: d for d in fds}
    edge_w: Dict[Tuple[int, int], float] = {}
    for u in fds:
        for v, w in graph.adj[u]:
            if v not in fset:
                continue
            a, b = (u, v) if u < v else (v, u)
            prev = edge_w.get((a, b))
            if prev is None or w < prev:
                edge_w[(a, b)] = float(w)
            union(parent, u, v)
    comps: Dict[int, List[int]] = {}
    for d in fds:
        comps.setdefault(find_root(parent, d), []).append(d)
    out = [sorted(c) for c in comps.values()]
    out.sort(key=lambda c: c[0])
    return out, edge_w


def max_component_size(comps: List[List[int]]) -> int:
    return max((len(c) for c in comps), default=0)
