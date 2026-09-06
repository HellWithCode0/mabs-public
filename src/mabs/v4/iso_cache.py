"""Exact syndrome-pattern cache for repeated sparse windows.

``ExactPatternCache`` memoizes absolute/global detector patterns — exact
memoization, **not** translation-isomorphism. Keys are exact defect-id tuples
(optionally scoped by ``det_lo``). Optional global-key API kept for experiments.
LRU / size-capped (~10k). Metrics: hits / misses / hit_rate.
"""

from __future__ import annotations

from collections import OrderedDict
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Sequence, Tuple, Union
import numpy as np


CacheKey = Tuple


def canonicalize_defects(defects: Sequence[int]) -> Tuple[Tuple[int, ...], int]:
    if defects is None:
        return (), 0
    arr = [int(x) for x in defects]
    if not arr:
        return (), 0
    return tuple(sorted(arr)), min(arr)


def relative_key(defects: Sequence[int]) -> Tuple[int, ...]:
    arr = [int(x) for x in defects]
    if not arr:
        return ()
    origin = min(arr)
    return tuple(sorted(d - origin for d in arr))


def local_to_global(fired: Sequence[int], gnode_list: Sequence[int]) -> List[int]:
    return [int(gnode_list[int(i)]) for i in fired]


def edges_local_to_global(
    edges: np.ndarray,
    gnode_list: Sequence[int],
    n_local: int,
) -> np.ndarray:
    if edges is None or len(edges) == 0:
        return np.zeros((0, 2), dtype=np.int64)
    arr = np.asarray(edges, dtype=np.int64).reshape(-1, 2)
    out = np.empty_like(arr)
    for i, (u, v) in enumerate(arr):
        uu, vv = int(u), int(v)
        out[i, 0] = -1 if uu < 0 or uu >= n_local else int(gnode_list[uu])
        out[i, 1] = -1 if vv < 0 or vv >= n_local else int(gnode_list[vv])
    return out


def edges_global_to_local(
    edges: np.ndarray,
    global_to_local: Dict[int, int],
    n_local: int,
) -> Optional[np.ndarray]:
    if edges is None or len(edges) == 0:
        return np.zeros((0, 2), dtype=np.int64)
    arr = np.asarray(edges, dtype=np.int64).reshape(-1, 2)
    out = np.empty_like(arr)
    get = global_to_local.get
    for i in range(arr.shape[0]):
        uu = int(arr[i, 0]); vv = int(arr[i, 1])
        if uu < 0:
            ou = n_local
        else:
            ou = get(uu)
            if ou is None:
                return None
        if vv < 0:
            ov = n_local
        else:
            ov = get(vv)
            if ov is None:
                return None
        out[i, 0] = ou
        out[i, 1] = ov
    return out


def _build_g2l(gnode_list: Sequence[int]) -> Dict[int, int]:
    return {int(g): i for i, g in enumerate(gnode_list)}


@dataclass
class ExactPatternCache:
    """LRU exact-pattern cache for matching edges (not isomorphism).

    Primary API uses window-local keys ``(det_lo, sorted_locals)``.
    """

    max_size: int = 10_000
    hits: int = 0
    misses: int = 0
    _store: "OrderedDict[CacheKey, np.ndarray]" = field(default_factory=OrderedDict)

    def clear(self) -> None:
        self._store.clear()
        self.hits = 0
        self.misses = 0

    @property
    def size(self) -> int:
        return len(self._store)

    @property
    def hit_rate(self) -> float:
        t = self.hits + self.misses
        return 0.0 if t == 0 else self.hits / t

    def _put(self, key: CacheKey, arr: np.ndarray) -> None:
        if key in self._store:
            self._store.move_to_end(key)
            self._store[key] = arr
        else:
            self._store[key] = arr
            while len(self._store) > self.max_size:
                self._store.popitem(last=False)

    def lookup_local(
        self,
        det_lo: int,
        defects: Sequence[int],
    ) -> Optional[np.ndarray]:
        key = (int(det_lo), tuple(sorted(int(x) for x in defects)))
        if key[1] == ():
            self.hits += 1
            return np.zeros((0, 2), dtype=np.int64)
        if key not in self._store:
            self.misses += 1
            return None
        self.hits += 1
        self._store.move_to_end(key)
        return self._store[key]

    def store_local(
        self,
        det_lo: int,
        defects: Sequence[int],
        edges: np.ndarray,
    ) -> None:
        key = (int(det_lo), tuple(sorted(int(x) for x in defects)))
        if key[1] == ():
            return
        arr = np.asarray(edges, dtype=np.int64).reshape(-1, 2).copy()
        self._put(key, arr)

    def lookup_global(self, global_defects: Sequence[int]) -> Optional[np.ndarray]:
        key = ("g", tuple(sorted(int(x) for x in global_defects)))
        if key[1] == ():
            self.hits += 1
            return np.zeros((0, 2), dtype=np.int64)
        if key not in self._store:
            self.misses += 1
            return None
        self.hits += 1
        self._store.move_to_end(key)
        return np.asarray(self._store[key], dtype=np.int64).reshape(-1, 2).copy()

    def store_global(self, global_defects: Sequence[int], global_edges: np.ndarray) -> None:
        key = ("g", tuple(sorted(int(x) for x in global_defects)))
        if key[1] == ():
            return
        self._put(key, np.asarray(global_edges, dtype=np.int64).reshape(-1, 2).copy())

    def lookup(
        self,
        defects: Sequence[int],
        n_local: int,
        gnode_list: Optional[Sequence[int]] = None,
        g2l: Optional[Dict[int, int]] = None,
        det_lo: Optional[int] = None,
    ) -> Optional[np.ndarray]:
        if det_lo is not None:
            return self.lookup_local(det_lo, defects)
        if gnode_list is not None:
            gdefs = local_to_global(defects, gnode_list)
            gedges = self.lookup_global(gdefs)
            if gedges is None:
                return None
            if g2l is None:
                g2l = _build_g2l(gnode_list)
            return edges_global_to_local(gedges, g2l, n_local)
        key = tuple(sorted(int(x) for x in defects))
        if not key:
            self.hits += 1
            return np.zeros((0, 2), dtype=np.int64)
        if key not in self._store:
            self.misses += 1
            return None
        self.hits += 1
        self._store.move_to_end(key)
        return np.asarray(self._store[key], dtype=np.int64).reshape(-1, 2).copy()

    def store(
        self,
        defects: Sequence[int],
        edges: np.ndarray,
        n_local: int,
        gnode_list: Optional[Sequence[int]] = None,
        det_lo: Optional[int] = None,
    ) -> None:
        if det_lo is not None:
            self.store_local(det_lo, defects, edges)
            return
        if gnode_list is not None:
            gdefs = local_to_global(defects, gnode_list)
            gedges = edges_local_to_global(edges, gnode_list, n_local)
            self.store_global(gdefs, gedges)
            return
        key = tuple(sorted(int(x) for x in defects))
        if not key:
            return
        self._put(key, np.asarray(edges, dtype=np.int64).reshape(-1, 2).copy())

    def stats(self) -> Dict[str, float]:
        return {
            "cache_hits": float(self.hits),
            "cache_misses": float(self.misses),
            "cache_hit_rate": self.hit_rate,
            "cache_size": float(self.size),
        }


IsoCache = ExactPatternCache
