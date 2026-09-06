"""CASCADE — Cached Approximate Sparse Correction with Amortized Deferred Escalation.

MABS v4 streaming decoder:

Easy path:
  empty → iso-cache (global detector key) → exact few-defect / clique MWPM
  (default ``clique_cap=2`` for stage Pareto; raise to 3–6 to cut escalate further)

Hard path:
  Per-window Sparse Blossom (LER-safe). Blossom outcomes are cached so repeated
  global syndromes skip blossom (amortized escalation).

``use_component_clique`` is off by default (induced components ignore
cross-quiescent matchings → LER risk). ``defer_hard`` is experimental.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import List, Optional
import numpy as np

from mabs.streaming import CircuitBundle, ShotDecodeResult, WindowRecord
from mabs.streaming_graph import (
    commit_window_edges,
    ensure_window_schedule,
    mask_to_obs_array,
)
from mabs.timing import NestedTimers
from mabs.v3.cluster import _get_graph_cache
from mabs.v3.local_decode import (
    blossom_edges_only,
    local_decode_few_defects,
    syndrome_cleared_by_edges,
)
from mabs.v3.union_find import fired_induced_components
from mabs.v4.clique_mwpm import clique_mwpm_edges
from mabs.v4.iso_cache import IsoCache


@dataclass
class CASCADEConfig:
    window_factor: float = 3.0
    commit_factor: float = 1.0
    # Default 2: exact 1–2 defect path (stage-competitive with v3).
    # Set 3–6 to cut escalate further at a pure-Python Dijkstra cost.
    clique_cap: int = 2
    cache_size: int = 10_000
    use_cache: bool = True
    use_clique: bool = True
    # OFF by default — component-wise matching can inflate LER.
    use_component_clique: bool = False
    check_syndrome: bool = True
    prefer_correctness: bool = True
    prewarm_graphs: bool = True
    defer_hard: bool = False
    cache_escalations: bool = True
    # Only cache/lookup hard windows with K in (clique_cap, cache_k_max].
    cache_k_max: int = 6


@dataclass
class CASCADEState:
    n_windows: int = 0
    n_empty: int = 0
    n_cache: int = 0
    n_clique: int = 0
    n_escalate: int = 0
    n_defer_flush: int = 0
    graphs_ready: bool = False
    cache: IsoCache = field(default_factory=lambda: IsoCache(max_size=10_000))

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


def prewarm_cascade_graphs(bundle: CircuitBundle, window_depth: int, commit_stride: int) -> List:
    windows = ensure_window_schedule(bundle, window_depth, commit_stride)
    for wm in windows:
        _get_graph_cache(wm)
    return windows


def _try_component_clique(wm, fired: np.ndarray, *, clique_cap: int, check: bool) -> Optional[np.ndarray]:
    graph = _get_graph_cache(wm)
    comps, _edge_w = fired_induced_components(graph, fired)
    if not comps or any(len(c) > clique_cap for c in comps):
        return None
    parts: List[np.ndarray] = []
    for c in comps:
        if len(c) <= 2:
            edges = local_decode_few_defects(wm, wm.buf, np.asarray(c, dtype=np.int64))
        else:
            edges = clique_mwpm_edges(graph, c, max_k=clique_cap)
        if edges is None:
            return None
        if edges.size:
            parts.append(edges)
    out = np.zeros((0, 2), dtype=np.int64) if not parts else np.vstack(parts)
    if check and not syndrome_cleared_by_edges(wm.n_local, fired, out):
        return None
    return out


def _try_clique(wm, fired: np.ndarray, *, clique_cap: int, check: bool) -> Optional[np.ndarray]:
    nd = int(fired.size)
    if nd == 0:
        return np.zeros((0, 2), dtype=np.int64)
    if nd > clique_cap:
        return None
    if nd <= 2:
        edges = local_decode_few_defects(wm, wm.buf, fired)
    else:
        graph = _get_graph_cache(wm)
        edges = clique_mwpm_edges(graph, fired.tolist(), max_k=clique_cap)
    if edges is None:
        return None
    if check and not syndrome_cleared_by_edges(wm.n_local, fired, edges):
        return None
    return edges


def stream_shot_cascade_timed(
    bundle: CircuitBundle,
    syndrome: np.ndarray,
    *,
    config: Optional[CASCADEConfig] = None,
    state: Optional[CASCADEState] = None,
    campaign: int = 0,
    shot: int = 0,
    observable_flips: Optional[np.ndarray] = None,
) -> ShotDecodeResult:
    if config is None:
        config = CASCADEConfig()
    if state is None:
        state = CASCADEState(cache=IsoCache(max_size=config.cache_size))
    elif state.cache.max_size != config.cache_size:
        state.cache.max_size = config.cache_size

    d = bundle.d
    n_det = bundle.n_detectors
    n_obs = max(int(bundle.num_observables), 1)
    w = max(1, int(round(config.window_factor * d)))
    C = max(1, int(round(config.commit_factor * d)))

    if config.prewarm_graphs and not state.graphs_ready:
        windows = prewarm_cascade_graphs(bundle, w, C)
        state.graphs_ready = True
    else:
        windows = ensure_window_schedule(bundle, w, C)

    det = np.asarray(syndrome, dtype=np.uint8).ravel()
    carry = np.zeros(n_det + 1, dtype=np.uint8)
    obs_mask = 0
    records: list[WindowRecord] = []
    deferred: list = []

    for w_idx, wm in enumerate(windows):
        lo, hi, n = wm.det_lo, wm.det_hi, wm.n_local
        buf = wm.buf
        timers = NestedTimers()
        escalated = False
        path = "empty"
        edges = np.zeros((0, 2), dtype=np.int64)
        weight = 0.0
        skip_commit = False
        gnodes = wm.gnode_list
        g2l = getattr(wm, "_cascade_g2l", None)
        if g2l is None:
            g2l = {int(g): i for i, g in enumerate(gnodes)}
            setattr(wm, "_cascade_g2l", g2l)

        with timers.input():
            np.bitwise_xor(det[lo:hi], carry[lo:hi], out=buf[:n])
            if buf.size > n:
                buf[n:] = 0
            carry[lo:hi] = 0
            fired = np.flatnonzero(buf[:n])
            n_defects = int(fired.size)
            dens = float(n_defects) / float(n) if n else 0.0

        with timers.match():
            if n_defects == 0:
                path = "empty"
                state.n_empty += 1
            else:
                handled = False

                # Cache only for hard patterns (K > clique_cap): easy local is
                # already faster than dict lookup + remap.
                if (
                    config.use_cache
                    and n_defects > config.clique_cap
                    and n_defects <= config.cache_k_max
                ):
                    cached = state.cache.lookup(fired.tolist(), n, gnode_list=gnodes, g2l=g2l)
                    if cached is not None:
                        edges = np.asarray(cached, dtype=np.int64).reshape(-1, 2).copy()
                        weight = float(edges.shape[0])
                        path = "cache"
                        state.n_cache += 1
                        handled = True

                if not handled and config.use_clique and n_defects <= config.clique_cap:
                    cl = _try_clique(
                        wm, fired, clique_cap=config.clique_cap, check=config.check_syndrome
                    )
                    if cl is not None:
                        edges = cl
                        weight = float(edges.shape[0])
                        path = "clique"
                        state.n_clique += 1
                        handled = True

                if (
                    not handled
                    and config.use_component_clique
                    and config.use_clique
                    and n_defects > config.clique_cap
                ):
                    cl = _try_component_clique(
                        wm, fired, clique_cap=config.clique_cap, check=config.check_syndrome
                    )
                    if cl is not None:
                        edges = cl
                        weight = float(edges.shape[0])
                        path = "clique"
                        state.n_clique += 1
                        handled = True
                        if config.use_cache:
                            state.cache.store(fired.tolist(), edges, n, gnode_list=gnodes)

                if not handled:
                    if config.defer_hard and w_idx < len(windows) - 1:
                        deferred.append((wm, buf.copy(), fired.copy()))
                        path = "defer"
                        skip_commit = True
                    else:
                        edges, weight = blossom_edges_only(wm, buf)
                        escalated = True
                        path = "escalate"
                        state.n_escalate += 1
                        if (
                            config.cache_escalations
                            and config.use_cache
                            and n_defects <= config.cache_k_max
                        ):
                            state.cache.store(fired.tolist(), edges, n, gnode_list=gnodes)

        with timers.post():
            if skip_commit:
                n_committed = 0
            else:
                if deferred and path != "defer":
                    for wm_i, buf_i, fired_i in deferred:
                        edges_i, _ = blossom_edges_only(wm_i, buf_i)
                        state.n_escalate += 1
                        state.n_defer_flush += 1
                        if config.cache_escalations and config.use_cache and fired_i.size:
                            state.cache.store(fired_i.tolist(), edges_i, wm_i.n_local, gnode_list=wm_i.gnode_list)
                        pm, _ = commit_window_edges(
                            edges_i, wm_i, carry, carry_forward=True
                        )
                        obs_mask ^= pm
                    deferred.clear()
                part_mask, n_committed = commit_window_edges(
                    edges, wm, carry, carry_forward=True
                )
                obs_mask ^= part_mask

        state.n_windows += 1
        sample = timers.sample()
        K = int(edges.shape[0]) if not skip_commit else 0
        conf = {
            "empty": 1.0,
            "cache": 0.9,
            "clique": 0.85,
            "defer": 0.4,
            "escalate": 0.35,
        }.get(path, 0.5)
        rec = WindowRecord(
            d=d,
            campaign=campaign,
            shot=shot,
            window_index=w_idx,
            K=K,
            tau_input_ns=sample.tau_input_ns,
            tau_match_ns=sample.tau_match_ns,
            tau_post_ns=sample.tau_post_ns,
            tau_stage_ns=sample.tau_stage_ns,
            empty_output=(n_committed == 0),
            commit_lo=wm.commit_start,
            commit_hi=wm.commit_end,
            n_committed=n_committed,
            matching_weight=weight,
            syndrome_density=dens,
            window_depth=wm.spec.end_layer - wm.spec.start_layer,
            retried=escalated,
            confidence=conf,
        )
        setattr(rec, "escalated", escalated)
        setattr(rec, "cascade_path", path)
        records.append(rec)

    if deferred:
        for wm_i, buf_i, fired_i in deferred:
            edges_i, _ = blossom_edges_only(wm_i, buf_i)
            state.n_escalate += 1
            state.n_defer_flush += 1
            if config.cache_escalations and config.use_cache and fired_i.size:
                state.cache.store(fired_i.tolist(), edges_i, wm_i.n_local, gnode_list=wm_i.gnode_list)
            pm, _ = commit_window_edges(edges_i, wm_i, carry, carry_forward=True)
            obs_mask ^= pm
        deferred.clear()

    pred = mask_to_obs_array(obs_mask, n_obs)
    if observable_flips is None:
        return ShotDecodeResult(
            predicted_observables=pred[: bundle.num_observables],
            records=records,
            logical_error=False,
        )

    obs = np.asarray(observable_flips, dtype=np.uint8).ravel()
    if obs.size and pred.size < obs.size:
        pad = np.zeros(obs.size, dtype=np.uint8)
        pad[: pred.size] = pred
        pred = pad
    logical_error = bool(obs.size and np.any(pred[: obs.size] != obs))
    return ShotDecodeResult(
        predicted_observables=pred[: obs.size] if obs.size else pred[: bundle.num_observables],
        records=records,
        logical_error=logical_error,
    )


stream_shot_cascade = stream_shot_cascade_timed
