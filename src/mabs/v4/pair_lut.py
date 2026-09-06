"""Lazy pair distance/path LUT for exact K=2 (and boundary-pair) decode.

Replaces per-window pure-Python Dijkstra for repeated (det_a, det_b) pairs on
the same Matching / DetectorGraph. Populate on first miss; reuse forever for
that graph.
"""

from __future__ import annotations

from typing import Dict, Optional, Tuple
import numpy as np

from mabs.v3.cluster import DetectorGraph
from mabs.v3.local_decode import decode_one_defect, decode_two_defects

PairKey = Tuple[int, int]

class PairPathLUT:
    def __init__(self, graph: DetectorGraph):
        self.graph = graph
        self._store: Dict[PairKey, Optional[np.ndarray]] = {}
        self.hits = 0
        self.misses = 0
        self.fills = 0

    @staticmethod
    def key(a: int, b: int) -> PairKey:
        aa, bb = int(a), int(b)
        return (aa, bb) if aa <= bb else (bb, aa)

    def get_or_compute(self, a: int, b: int) -> Optional[np.ndarray]:
        k = self.key(a, b)
        if k in self._store:
            self.hits += 1
            return self._store[k]
        self.misses += 1
        aa, bb = k
        if aa == bb:
            edges = decode_one_defect(self.graph, aa)
        else:
            edges = decode_two_defects(self.graph, aa, bb)
        if edges is None:
            self._store[k] = None
        else:
            self._store[k] = np.asarray(edges, dtype=np.int64).reshape(-1, 2).copy()
        self.fills += 1
        return self._store[k]

    @property
    def size(self) -> int:
        return len(self._store)

    @property
    def hit_rate(self) -> float:
        t = self.hits + self.misses
        return 0.0 if t == 0 else self.hits / t

def get_pair_lut(graph: DetectorGraph) -> PairPathLUT:
    lut = getattr(graph, "_pair_lut", None)
    if lut is None or lut.graph is not graph:
        lut = PairPathLUT(graph)
        setattr(graph, "_pair_lut", lut)
    return lut

def decode_pair_cached(graph: DetectorGraph, a: int, b: int) -> Optional[np.ndarray]:
    return get_pair_lut(graph).get_or_compute(int(a), int(b))
