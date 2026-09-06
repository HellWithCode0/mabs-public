"""Exact syndrome-pattern cache (ExactPatternCache).

Absolute keys store **CommitAction** values (obs XOR + carry flips) applied
directly by the streaming commit path — no edge re-interpretation on hit.

Relative keys (opt-in) store **relative local edges**; on hit they are
translated by the current pattern origin then compiled to a CommitAction.
Relative mode is only valid under the same topology id (priority 4).

## Cache key schema (topology-safe)

Absolute (default ``canonicalize_relative=False``)::

    (topo_id, sorted_local_defects: Tuple[int, ...])

Relative (``canonicalize_relative=True``)::

    (topo_id, "rel", relative_offsets: Tuple[int, ...])

``topo_id``::

    (d, window_depth, commit_stride, n_local, det_lo, graph_token)

Frequency admission: insert only after ``admit_after`` misses (default 2).
"""

from __future__ import annotations

from collections import OrderedDict
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Sequence, Tuple, Union
import numpy as np

from mabs.v4.commit_action import CommitAction, edges_to_commit_action

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

def edges_local_to_global(edges: np.ndarray, gnode_list: Sequence[int], n_local: int) -> np.ndarray:
    if edges is None or len(edges) == 0:
        return np.zeros((0, 2), dtype=np.int64)
    arr = np.asarray(edges, dtype=np.int64).reshape(-1, 2)
    out = np.empty_like(arr)
    for i, (u, v) in enumerate(arr):
        uu, vv = int(u), int(v)
        out[i, 0] = -1 if uu < 0 or uu >= n_local else int(gnode_list[uu])
        out[i, 1] = -1 if vv < 0 or vv >= n_local else int(gnode_list[vv])
    return out

def edges_global_to_local(edges: np.ndarray, global_to_local: Dict[int, int], n_local: int) -> Optional[np.ndarray]:
    if edges is None or len(edges) == 0:
        return np.zeros((0, 2), dtype=np.int64)
    arr = np.asarray(edges, dtype=np.int64).reshape(-1, 2)
    out = np.empty_like(arr)
    get = global_to_local.get
    for i in range(arr.shape[0]):
        uu = int(arr[i, 0]); vv = int(arr[i, 1])
        if uu < 0: ou = n_local
        else:
            ou = get(uu)
            if ou is None: return None
        if vv < 0: ov = n_local
        else:
            ov = get(vv)
            if ov is None: return None
        out[i, 0] = ou; out[i, 1] = ov
    return out

def make_topo_id(*, d: int, window_depth: int, commit_stride: int, n_local: int, det_lo: int, graph_token: Any) -> Tuple:
    return (int(d), int(window_depth), int(commit_stride), int(n_local), int(det_lo), graph_token)

def topo_id_from_wm(wm: Any, *, d: int, graph_token: Optional[Any] = None) -> Tuple:
    depth = int(wm.spec.end_layer - wm.spec.start_layer)
    stride = int(wm.commit_end - wm.commit_start)
    tok = graph_token if graph_token is not None else id(wm.matching)
    return make_topo_id(d=d, window_depth=depth, commit_stride=max(1, stride), n_local=int(wm.n_local), det_lo=int(wm.det_lo), graph_token=tok)

def build_pattern_key(topo_id: Tuple, defects: Sequence[int], *, canonicalize_relative: bool = False) -> CacheKey:
    dets = tuple(sorted(int(x) for x in defects))
    if not dets: return (topo_id, ())
    if canonicalize_relative: return (topo_id, "rel", relative_key(dets))
    return (topo_id, dets)

def _shift_edges(edges: np.ndarray, origin: int, n_local: int) -> np.ndarray:
    if edges is None or len(edges) == 0: return np.zeros((0, 2), dtype=np.int64)
    arr = np.asarray(edges, dtype=np.int64).reshape(-1, 2).copy()
    for i in range(arr.shape[0]):
        for j in range(2):
            v = int(arr[i, j])
            if 0 <= v < n_local: arr[i, j] = v + origin
    return arr

def _unshift_edges(edges: np.ndarray, origin: int, n_local: int) -> np.ndarray:
    if edges is None or len(edges) == 0: return np.zeros((0, 2), dtype=np.int64)
    arr = np.asarray(edges, dtype=np.int64).reshape(-1, 2).copy()
    for i in range(arr.shape[0]):
        for j in range(2):
            v = int(arr[i, j])
            if 0 <= v < n_local: arr[i, j] = v - origin
    return arr

@dataclass
class _CacheValue:
    action: Optional[CommitAction] = None
    rel_edges: Optional[np.ndarray] = None

@dataclass
class ExactPatternCache:
    max_size: int = 10_000
    admit_after: int = 2
    canonicalize_relative: bool = False
    hits: int = 0
    misses: int = 0
    admits: int = 0
    rejects: int = 0
    _store: "OrderedDict[CacheKey, _CacheValue]" = field(default_factory=OrderedDict)
    _freq: Dict[CacheKey, int] = field(default_factory=dict)

    def clear(self) -> None:
        self._store.clear(); self._freq.clear()
        self.hits = self.misses = self.admits = self.rejects = 0

    @property
    def size(self) -> int:
        return len(self._store)

    @property
    def hit_rate(self) -> float:
        t = self.hits + self.misses
        return 0.0 if t == 0 else self.hits / t

    def _put(self, key: CacheKey, val: _CacheValue) -> None:
        if key in self._store:
            self._store.move_to_end(key); self._store[key] = val
        else:
            self._store[key] = val
            while len(self._store) > self.max_size:
                self._store.popitem(last=False)

    def lookup_action(self, topo_id: Tuple, defects: Sequence[int], wm: Optional[Any] = None) -> Optional[CommitAction]:
        dets = [int(x) for x in defects]
        key = build_pattern_key(topo_id, dets, canonicalize_relative=self.canonicalize_relative)
        if not dets:
            self.hits += 1; return CommitAction.empty()
        if key not in self._store:
            self.misses += 1; self._freq[key] = self._freq.get(key, 0) + 1; return None
        self.hits += 1; self._store.move_to_end(key)
        val = self._store[key]
        if val.action is not None: return val.action
        if val.rel_edges is None or wm is None: return None
        return edges_to_commit_action(_shift_edges(val.rel_edges, min(dets), int(wm.n_local)), wm)

    def offer_store_action(self, topo_id: Tuple, defects: Sequence[int], action: CommitAction, *, edges: Optional[np.ndarray] = None, n_local: Optional[int] = None, force: bool = False) -> bool:
        dets = [int(x) for x in defects]
        key = build_pattern_key(topo_id, dets, canonicalize_relative=self.canonicalize_relative)
        if not dets: return False
        freq = self._freq.get(key, 0)
        if (not force) and freq < self.admit_after and key not in self._store:
            self.rejects += 1; return False
        if self.canonicalize_relative:
            if edges is None:
                self.rejects += 1; return False
            nl = int(n_local) if n_local is not None else (max(dets) + 1)
            self._put(key, _CacheValue(action=None, rel_edges=_unshift_edges(edges, min(dets), nl)))
        else:
            self._put(key, _CacheValue(action=action, rel_edges=None))
        self.admits += 1; return True

    def store_from_edges(self, topo_id: Tuple, defects: Sequence[int], edges: np.ndarray, wm: Any, *, force: bool = False) -> bool:
        return self.offer_store_action(topo_id, defects, edges_to_commit_action(edges, wm), edges=edges, n_local=int(wm.n_local), force=force)

    def lookup_local(self, det_lo: int, defects: Sequence[int]) -> Optional[CommitAction]:
        return self.lookup_action(("legacy", int(det_lo)), defects)

    def store_local(self, det_lo: int, defects: Sequence[int], edges: np.ndarray, wm: Any = None) -> None:
        topo = ("legacy", int(det_lo))
        if wm is not None: self.store_from_edges(topo, defects, edges, wm, force=True)
        else: self.offer_store_action(topo, defects, CommitAction.empty(), edges=edges, force=True)

    def lookup(self, defects: Sequence[int], n_local: int = 0, gnode_list: Optional[Sequence[int]] = None, g2l: Optional[Dict[int, int]] = None, det_lo: Optional[int] = None, topo_id: Optional[Tuple] = None, wm: Any = None):
        del n_local, gnode_list, g2l
        if topo_id is None: topo_id = ("legacy", int(det_lo) if det_lo is not None else -1)
        return self.lookup_action(topo_id, defects, wm=wm)

    def store(self, defects: Sequence[int], edges: np.ndarray, n_local: int = 0, gnode_list: Optional[Sequence[int]] = None, det_lo: Optional[int] = None, topo_id: Optional[Tuple] = None, wm: Any = None, *, force: bool = True) -> None:
        del n_local, gnode_list
        if topo_id is None: topo_id = ("legacy", int(det_lo) if det_lo is not None else -1)
        if wm is not None: self.store_from_edges(topo_id, defects, edges, wm, force=force)
        else: self.offer_store_action(topo_id, defects, CommitAction.empty(), edges=edges, force=force)

    def stats(self) -> Dict[str, float]:
        return {"cache_hits": float(self.hits), "cache_misses": float(self.misses), "cache_hit_rate": self.hit_rate, "cache_size": float(self.size), "cache_admits": float(self.admits), "cache_rejects": float(self.rejects)}

IsoCache = ExactPatternCache
