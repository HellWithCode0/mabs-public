"""Syndrome clustering on the detector / matching graph."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Sequence, Tuple
import numpy as np


@dataclass
class DetectorGraph:
    """Weighted adjacency for one window matcher (local detector ids)."""

    n_local: int
    adj: List[List[Tuple[int, float]]]  # (neighbor, weight); -1 = boundary
    boundary: set
    # Precomputed nearest-boundary forest (multi-source Dijkstra).
    bound_dist: Dict[int, float] = field(default_factory=dict)
    bound_prev: Dict[int, int] = field(default_factory=dict)

    @staticmethod
    def from_matching(matching: Any, n_local: int) -> "DetectorGraph":
        adj: List[List[Tuple[int, float]]] = [[] for _ in range(n_local)]
        boundary: set = set()
        for u, v, attrs in matching.edges():
            uu = int(u)
            if uu < 0 or uu >= n_local:
                continue
            wt = float(attrs.get("weight", 1.0))
            if v is None or int(v) >= n_local:
                adj[uu].append((-1, wt))
                boundary.add(uu)
            else:
                vv = int(v)
                if vv < 0:
                    adj[uu].append((-1, wt))
                    boundary.add(uu)
                else:
                    adj[uu].append((vv, wt))
                    adj[vv].append((uu, wt))
        g = DetectorGraph(n_local=n_local, adj=adj, boundary=boundary)
        g._compute_boundary_forest()
        return g

    def _compute_boundary_forest(self) -> None:
        import heapq

        dist: Dict[int, float] = {}
        prev: Dict[int, int] = {}
        pq: List[Tuple[float, int]] = []
        for u in range(self.n_local):
            for v, w in self.adj[u]:
                if v == -1:
                    if w < dist.get(u, 1e300):
                        dist[u] = w
                        prev[u] = -1
                        heapq.heappush(pq, (w, u))
        while pq:
            du, u = heapq.heappop(pq)
            if du != dist.get(u):
                continue
            for v, w in self.adj[u]:
                if v < 0:
                    continue
                nd = du + w
                if nd < dist.get(v, 1e300):
                    dist[v] = nd
                    prev[v] = u
                    heapq.heappush(pq, (nd, v))
        self.bound_dist = dist
        self.bound_prev = prev


@dataclass
class Cluster:
    """One connected component of defects (optionally grown)."""

    defects: List[int]
    nodes: List[int]  # defects + grown neighborhood

    @property
    def size(self) -> int:
        return len(self.defects)

    @property
    def odd(self) -> bool:
        return len(self.defects) % 2 == 1


def build_detector_graph(matching: Any, n_local: int) -> DetectorGraph:
    return DetectorGraph.from_matching(matching, n_local)


def _get_graph_cache(wm: Any) -> DetectorGraph:
    g = getattr(wm, "_slem_graph", None)
    if g is None or g.n_local != wm.n_local:
        g = DetectorGraph.from_matching(wm.matching, wm.n_local)
        setattr(wm, "_slem_graph", g)
    return g


def cluster_syndrome(
    syndrome: np.ndarray,
    graph: DetectorGraph,
    *,
    radius: int = 1,
) -> List[Cluster]:
    """Connected components of fired detectors through a radius-grown neighborhood.

    Two defects are in the same cluster if a path of length ≤ ``radius`` hops
    (through any detectors) connects them, or they share a grown ball.
    """
    n = graph.n_local
    syn = np.asarray(syndrome, dtype=np.uint8).ravel()
    if syn.size < n:
        pad = np.zeros(n, dtype=np.uint8)
        pad[: syn.size] = syn
        syn = pad
    fired = np.flatnonzero(syn[:n]).astype(np.int64)
    if fired.size == 0:
        return []

    fired_set = set(int(x) for x in fired.tolist())

    # Grow neighborhood around all defects.
    nodes = set(fired_set)
    frontier = list(fired_set)
    for _ in range(max(0, int(radius))):
        nxt: List[int] = []
        for u in frontier:
            for v, _w in graph.adj[u]:
                if v < 0:
                    continue
                if v not in nodes:
                    nodes.add(v)
                    nxt.append(v)
        frontier = nxt

    # Union-find over defects connected via paths staying in ``nodes``.
    parent = {d: d for d in fired_set}

    def find(x: int) -> int:
        while parent[x] != x:
            parent[x] = parent[parent[x]]
            x = parent[x]
        return x

    def union(a: int, b: int) -> None:
        ra, rb = find(a), find(b)
        if ra != rb:
            parent[rb] = ra

    from collections import deque

    for s in fired_set:
        seen = {s}
        q: deque = deque([s])
        while q:
            u = q.popleft()
            for v, _w in graph.adj[u]:
                if v < 0 or v not in nodes or v in seen:
                    continue
                seen.add(v)
                if v in fired_set:
                    union(s, v)
                q.append(v)

    comps: Dict[int, List[int]] = {}
    for d in fired_set:
        comps.setdefault(find(d), []).append(d)

    # Per-cluster grown nodes: ball of radius around that cluster's defects.
    out: List[Cluster] = []
    for defs in comps.values():
        cnodes = set(defs)
        fr = list(defs)
        for _ in range(max(0, int(radius))):
            nxt2: List[int] = []
            for u in fr:
                for v, _w in graph.adj[u]:
                    if v < 0:
                        continue
                    if v not in cnodes:
                        cnodes.add(v)
                        nxt2.append(v)
            fr = nxt2
        out.append(Cluster(defects=sorted(defs), nodes=sorted(cnodes)))
    out.sort(key=lambda c: c.defects[0] if c.defects else 0)
    return out


def cluster_window_syndrome(wm: Any, buf: np.ndarray, *, radius: int = 1) -> List[Cluster]:
    graph = _get_graph_cache(wm)
    return cluster_syndrome(buf, graph, radius=radius)
