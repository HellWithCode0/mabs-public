"""CASCADE core: config, costs, pair/clique/peel helpers (v4.1)."""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Tuple
import numpy as np

from mabs.streaming import CircuitBundle
from mabs.streaming_graph import ensure_window_schedule
from mabs.v3.cluster import _get_graph_cache
from mabs.v3.local_decode import syndrome_cleared_by_edges
from mabs.v3.union_find import fired_induced_components
from mabs.v4.clique_mwpm import clique_mwpm_edges
from mabs.v4.exact_pattern_cache import ExactPatternCache
from mabs.v4.pair_lut import decode_pair_cached, get_pair_lut


# Estimated Python control costs (µs-equivalent units) vs one blossom budget.
_COST_EMPTY = 0.1
_COST_CACHE = 0.5
_COST_PAIR = 2.0
_COST_CLIQUE_BASE = 5.0
_COST_CLIQUE_K = 3.0
_COST_PEEL = 8.0
_COST_BLOSSOM_DEFAULT = 25.0


@dataclass
class CASCADEConfig:
    window_factor: float = 3.0
    commit_factor: float = 1.0
    # Default 2: exact 1–2 defect path (stage Pareto).
    clique_cap: int = 2
    cache_size: int = 10_000
    use_cache: bool = True
    use_clique: bool = True
    use_pair_lut: bool = True
    # OFF by default — component-wise matching can inflate LER.
    use_component_clique: bool = False
    # Peel easy induced components; blossom when residual hard remains.
    use_peel: bool = True
    peel_cap: int = 4
    # When residual non-empty: blossom full window (LER-safe). residual_only is research.
    residual_only_blossom: bool = False
    check_syndrome: bool = True
    prefer_correctness: bool = True
    prewarm_graphs: bool = True
    cache_escalations: bool = True
    cache_k_max: int = 6
    # Frequency admission for ExactPatternCache (≥2 seen before insert).
    cache_admit_after: int = 2
    # Opt-in relative keys under topology id (default absolute).
    canonicalize_relative: bool = False
    # Cheap defect-count gate before MABS logic (default based on clique_cap).
    gate_max: Optional[int] = None  # None → max(12, clique_cap * 4)
    # Cost-aware routing: escalate if estimated local cost ≥ blossom budget.
    cost_aware: bool = True
    blossom_cost_budget: float = _COST_BLOSSOM_DEFAULT
    # Research: adaptive window depth (default off — LER risk if mis-tuned).
    adaptive_depth: bool = False
    w_small_factor: float = 2.0
    w_large_factor: float = 3.0
    adapt_density_threshold: float = 0.02

    def resolved_gate_max(self) -> int:
        if self.gate_max is not None:
            return int(self.gate_max)
        return max(12, int(self.clique_cap) * 4)


@dataclass
class CASCADEState:
    n_windows: int = 0
    n_empty: int = 0
    n_cache: int = 0
    n_pair: int = 0
    n_clique: int = 0
    n_peel: int = 0
    n_residual: int = 0
    n_escalate: int = 0
    n_gate_escalate: int = 0
    n_cost_escalate: int = 0
    # Route decision counters
    route_counts: Dict[str, int] = field(default_factory=dict)
    graphs_ready: bool = False
    cache: ExactPatternCache = field(
        default_factory=lambda: ExactPatternCache(max_size=10_000, admit_after=2)
    )

    def _bump_route(self, name: str) -> None:
        self.route_counts[name] = self.route_counts.get(name, 0) + 1

    @property
    def escalate_rate(self) -> float:
        return 0.0 if self.n_windows == 0 else self.n_escalate / self.n_windows

    @property
    def empty_rate(self) -> float:
        return 0.0 if self.n_windows == 0 else self.n_empty / self.n_windows

    @property
    def cache_hit_rate(self) -> float:
        return self.cache.hit_rate

    @property
    def clique_rate(self) -> float:
        return 0.0 if self.n_windows == 0 else self.n_clique / self.n_windows

    @property
    def cache_rate(self) -> float:
        return 0.0 if self.n_windows == 0 else self.n_cache / self.n_windows

    @property
    def pair_rate(self) -> float:
        return 0.0 if self.n_windows == 0 else self.n_pair / self.n_windows

    @property
    def peel_rate(self) -> float:
        return 0.0 if self.n_windows == 0 else self.n_peel / self.n_windows

    @property
    def residual_rate(self) -> float:
        return 0.0 if self.n_windows == 0 else self.n_residual / self.n_windows


def prewarm_cascade_graphs(bundle: CircuitBundle, window_depth: int, commit_stride: int) -> List:
    windows = ensure_window_schedule(bundle, window_depth, commit_stride)
    for wm in windows:
        g = _get_graph_cache(wm)
        get_pair_lut(g)
    return windows


def _estimate_local_cost(n_defects: int, *, clique_cap: int, use_peel: bool) -> float:
    if n_defects == 0:
        return _COST_EMPTY
    if n_defects <= 2:
        return _COST_PAIR
    if n_defects <= clique_cap:
        return _COST_CLIQUE_BASE + _COST_CLIQUE_K * (n_defects ** 2)
    if use_peel:
        return _COST_PEEL + _COST_CLIQUE_K * n_defects
    return _COST_BLOSSOM_DEFAULT * 2.0


def _try_pair_lut(wm, fired: np.ndarray, *, check: bool) -> Optional[np.ndarray]:
    nd = int(fired.size)
    if nd == 0:
        return np.zeros((0, 2), dtype=np.int64)
    graph = _get_graph_cache(wm)
    if nd == 1:
        edges = decode_pair_cached(graph, int(fired[0]), int(fired[0]))
    elif nd == 2:
        edges = decode_pair_cached(graph, int(fired[0]), int(fired[1]))
    else:
        return None
    if edges is None:
        return None
    if check and not syndrome_cleared_by_edges(wm.n_local, fired, edges):
        return None
    return edges


def _try_clique(wm, fired: np.ndarray, *, clique_cap: int, check: bool) -> Optional[np.ndarray]:
    nd = int(fired.size)
    if nd == 0:
        return np.zeros((0, 2), dtype=np.int64)
    if nd > clique_cap:
        return None
    if nd <= 2:
        return _try_pair_lut(wm, fired, check=check)
    graph = _get_graph_cache(wm)
    edges = clique_mwpm_edges(graph, fired.tolist(), max_k=clique_cap)
    if edges is None:
        return None
    if check and not syndrome_cleared_by_edges(wm.n_local, fired, edges):
        return None
    return edges


def _peel_easy_residual(
    wm,
    buf: np.ndarray,
    fired: np.ndarray,
    *,
    peel_cap: int,
    check: bool,
    prefer_correctness: bool = True,
) -> Tuple[Optional[np.ndarray], List[int], str]:
    """Peel easy induced components; return (easy_edges|None, residual_locals, tag).

    LER-safe default (``prefer_correctness=True``): only peel a **single**
    induced component of size ≤ peel_cap (exact clique/pair on that defect set).
    Multi-component local decode can miss cross-quiescent matchings → residual
    (full-window blossom). Opt-out via prefer_correctness=False.
    """
    graph = _get_graph_cache(wm)
    comps, _edge_w = fired_induced_components(graph, fired)
    if not comps:
        return None, fired.tolist(), "fail"

    if prefer_correctness and len(comps) > 1:
        # Do not partially commit easy comps — blossom full window.
        return np.zeros((0, 2), dtype=np.int64), fired.tolist(), "residual"

    parts: List[np.ndarray] = []
    residual: List[int] = []
    for c in comps:
        if len(c) <= peel_cap:
            if len(c) <= 2:
                arr = np.asarray(c, dtype=np.int64)
                edges = _try_pair_lut(wm, arr, check=check)
            else:
                edges = clique_mwpm_edges(graph, c, max_k=peel_cap)
                if edges is not None and check:
                    if not syndrome_cleared_by_edges(wm.n_local, np.asarray(c), edges):
                        edges = None
            if edges is None:
                residual.extend(c)
            elif edges.size:
                parts.append(edges)
            # size-0 edges (shouldn't happen) still "peeled"
        else:
            residual.extend(c)
    easy = np.zeros((0, 2), dtype=np.int64) if not parts else np.vstack(parts)
    if not residual:
        return easy, [], "peel"
    return easy, residual, "residual"


def _window_depth_choice(
    bundle: CircuitBundle,
    det: np.ndarray,
    carry: np.ndarray,
    start_layer: int,
    *,
    config: CASCADEConfig,
    d: int,
) -> int:
    """Research adaptive depth: shrink/expand using density peek (v2-style)."""
    w_large = max(1, int(round(config.w_large_factor * d)))
    w_small = max(1, int(round(config.w_small_factor * d)))
    if w_small > w_large:
        w_small = w_large
    if not config.adaptive_depth:
        return max(1, int(round(config.window_factor * d)))
    n_layers = bundle.n_layers
    peek_end = min(start_layer + w_large, n_layers)
    peek_lo = int(bundle.layer_lo[start_layer])
    peek_hi = int(bundle.layer_lo[peek_end])
    if peek_hi > peek_lo:
        dens = float(np.bitwise_xor(det[peek_lo:peek_hi], carry[peek_lo:peek_hi]).sum()) / float(
            peek_hi - peek_lo
        )
    else:
        dens = 0.0
    # High density → larger window; low → small (confidence-like).
    return w_large if dens >= config.adapt_density_threshold else w_small
