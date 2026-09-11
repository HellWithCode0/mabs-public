"""CASCADE streaming shot decode (v4.2 FLASH default + FULL opt-in)."""
from __future__ import annotations
from dataclasses import replace
from typing import Optional
import numpy as np
from mabs.streaming import CircuitBundle, ShotDecodeResult, WindowRecord
from mabs.streaming_graph import ensure_window_schedule, mask_to_obs_array
from mabs.streaming_windows import commit_window_edges
from mabs.timing import NestedTimers
from mabs.v3.local_decode import blossom_edges_only
from mabs.v4.commit_action import CommitAction, apply_commit_action
from mabs.v4.exact_pattern_cache import ExactPatternCache, topo_id_from_wm
from mabs.v4.flash_lut import extract_defects, get_flash_lut
from mabs.v4.cascade_core import (
    CASCADEConfig, CASCADEState, _window_depth_choice, prewarm_cascade_graphs,
)
from mabs.v4.cascade_route import route_window, route_window_flash_sticky
from mabs.v4.cluster_route import (
    apply_cluster_flips, fused_available, get_cluster_table, route_clusters_buf,
    route_clusters_fused,
)

_EMPTY_EDGES = np.zeros((0, 2), dtype=np.int64)
_EMPTY_ACTION = CommitAction.empty()


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
        state = CASCADEState(
            cache=ExactPatternCache(
                max_size=config.cache_size,
                admit_after=config.cache_admit_after,
                canonicalize_relative=config.canonicalize_relative,
            )
        )
    else:
        state.cache.max_size = config.cache_size
        state.cache.admit_after = config.cache_admit_after
        state.cache.canonicalize_relative = config.canonicalize_relative

    # Sticky resets each shot
    state.reset_shot_sticky()

    d = bundle.d
    n_det = bundle.n_detectors
    n_obs = max(int(bundle.num_observables), 1)
    C = max(1, int(round(config.commit_factor * d)))
    flash = config.is_flash()
    gate_max = config.resolved_gate_max()
    sticky_on = bool(config.sticky_blossom)
    fill_on_miss = bool(config.fill_pair_lut_on_miss)
    prefer_numba = bool(config.prefer_numba)

    if config.adaptive_depth:
        windows = None
        w_default = max(1, int(round(config.window_factor * d)))
        if config.prewarm_graphs and not state.graphs_ready:
            prewarm_cascade_graphs(bundle, w_default, C, config=config)
            w_small = max(1, int(round(config.w_small_factor * d)))
            prewarm_cascade_graphs(bundle, w_small, C, config=config)
            state.graphs_ready = True
    else:
        w = max(1, int(round(config.window_factor * d)))
        if config.prewarm_graphs and not state.graphs_ready:
            windows = prewarm_cascade_graphs(bundle, w, C, config=config)
            state.graphs_ready = True
        else:
            windows = ensure_window_schedule(bundle, w, C)

    det = np.asarray(syndrome, dtype=np.uint8).ravel()
    carry = np.zeros(n_det + 1, dtype=np.uint8)
    obs_mask = 0
    records: list[WindowRecord] = []
    timers = NestedTimers()

    def _iter_windows():
        if windows is not None:
            for wm in windows:
                yield wm
            return
        from mabs.streaming_graph import get_window_matcher, make_window_spec
        start = 0
        w_idx = 0
        while start < bundle.n_layers:
            depth = _window_depth_choice(bundle, det, carry, start, config=config, d=d)
            spec = make_window_spec(bundle, start, depth, C, index=w_idx)
            wm = get_window_matcher(bundle, spec)
            yield wm
            if wm.commit_end >= bundle.n_layers:
                break
            start += C
            if start >= bundle.n_layers or C <= 0:
                break
            w_idx += 1

    if flash:
        # Pre-bind LUTs for fixed schedules (avoid per-window getattr)
        flash_luts = None
        if windows is not None:
            flash_luts = [get_flash_lut(wm) for wm in windows]
        cluster_on = config.resolved_cluster_route()
        cluster_margin = float(config.cluster_margin)
        cluster_max = int(config.cluster_max_nodes)
        cluster_tabs = None
        if cluster_on and windows is not None:
            cluster_tabs = [get_cluster_table(wm, max_nodes=cluster_max) for wm in windows]

        for w_idx, wm in enumerate(_iter_windows()):
            lo, hi, n = wm.det_lo, wm.det_hi, wm.n_local
            buf = wm.buf
            lut = flash_luts[w_idx] if flash_luts is not None else get_flash_lut(wm)
            defect_buf = lut._defect_buf
            fused = None
            timers.reset()
            with timers.input():
                np.bitwise_xor(det[lo:hi], carry[lo:hi], out=buf[:n])
                if buf.size > n:
                    buf[n:] = 0
                carry[lo:hi] = 0
            with timers.match():
                if sticky_on and state.sticky_this_shot:
                    path, edges, action, escalated, weight = route_window_flash_sticky(
                        wm, buf, state=state
                    )
                else:
                    # Inlined FLASH route (avoid call overhead on hot path)
                    n_defects = extract_defects(
                        buf, n, defect_buf, prefer_numba=prefer_numba, cap=3
                    )
                    if n_defects == 0:
                        path, edges, action, escalated, weight = (
                            "empty", _EMPTY_EDGES, _EMPTY_ACTION, False, 0.0
                        )
                        state.n_empty += 1
                        state._bump_route("empty")
                    elif n_defects == 1:
                        act = lut.get_boundary(int(defect_buf[0]))
                        if act is not None:
                            path, edges, action, escalated, weight = (
                                "boundary", _EMPTY_EDGES, act, False, 0.0
                            )
                            state.n_boundary += 1
                            state.n_pair += 1
                            state._bump_route("boundary")
                        else:
                            edges, weight = blossom_edges_only(wm, buf)
                            path, action, escalated = "escalate", None, True
                            state.n_escalate += 1
                            state._bump_route("escalate")
                            if sticky_on:
                                state.sticky_this_shot = True
                    elif n_defects == 2:
                        a = int(defect_buf[0])
                        b = int(defect_buf[1])
                        act = lut.get_pair(a, b)
                        if act is not None:
                            path, edges, action, escalated, weight = (
                                "pair", _EMPTY_EDGES, act, False, 0.0
                            )
                            state.n_pair += 1
                            state._bump_route("pair")
                        else:
                            edges, weight = blossom_edges_only(wm, buf)
                            if fill_on_miss and edges is not None and edges.size:
                                act = lut.fill_pair_from_edges(a, b, edges)
                                path, action, escalated = "pair_fill", act, False
                                state.n_pair += 1
                                state._bump_route("pair_fill")
                            else:
                                path, action, escalated = "escalate", None, True
                                state.n_escalate += 1
                                state._bump_route("escalate")
                                if sticky_on:
                                    state.sticky_this_shot = True
                    else:
                        # K >= 3: CLUSTER route when every cluster is a singleton
                        # or pair and the dual certificate proves the decomposed
                        # matching optimal; otherwise blossom (commit like w3d).
                        act = None
                        if cluster_on:
                            ctab = (
                                cluster_tabs[w_idx] if cluster_tabs is not None
                                else get_cluster_table(wm, max_nodes=cluster_max)
                            )
                            if prefer_numba and fused_available(ctab):
                                # One compiled call: shape, certificate, lookups,
                                # XOR composition. Carry flips wait for post.
                                fused = route_clusters_fused(
                                    buf, n, ctab, margin=cluster_margin,
                                    stats=state.cluster_stats,
                                )
                            else:
                                act = route_clusters_buf(
                                    buf, n, ctab, lut, margin=cluster_margin,
                                    stats=state.cluster_stats, prefer_numba=prefer_numba,
                                )
                        if fused is not None or act is not None:
                            path, edges, action, escalated, weight = (
                                "cluster", _EMPTY_EDGES, act, False, 0.0
                            )
                            state.n_cluster += 1
                            state._bump_route("cluster")
                        else:
                            edges, weight = blossom_edges_only(wm, buf)
                            path, action, escalated = "escalate", None, True
                            state.n_escalate += 1
                            state._bump_route("escalate")
                            if sticky_on:
                                state.sticky_this_shot = True
            with timers.post():
                if fused is not None:
                    part_mask, n_committed = fused[0], fused[1]
                    apply_cluster_flips(carry, ctab, fused[2])
                elif action is not None:
                    part_mask, n_committed = apply_commit_action(action, carry)
                else:
                    part_mask, n_committed = commit_window_edges(
                        edges, wm, carry, carry_forward=True
                    )
                obs_mask ^= part_mask
            state.n_windows += 1
            # Instrumentation only, so it sits outside every timed interval.
            # The capped extract stops at 3 and the sticky path never counts,
            # so neither can supply the true density; buf still holds the
            # window syndrome here because commit touches only carry.
            dens = float(np.count_nonzero(buf[:n])) / float(n) if n else 0.0
            sample = timers.sample()
            conf = {
                "empty": 1.0,
                "sticky_empty": 1.0,
                "boundary": 0.92,
                "pair": 0.9,
                "pair_fill": 0.88,
                "cluster": 0.88,
                "escalate": 0.35,
                "sticky": 0.35,
            }.get(path, 0.5)
            rec = WindowRecord(
                d=d, campaign=campaign, shot=shot, window_index=w_idx, K=int(n_committed),
                tau_input_ns=sample.tau_input_ns, tau_match_ns=sample.tau_match_ns,
                tau_post_ns=sample.tau_post_ns, tau_stage_ns=sample.tau_stage_ns,
                empty_output=(n_committed == 0), commit_lo=wm.commit_start, commit_hi=wm.commit_end,
                n_committed=n_committed, matching_weight=weight, syndrome_density=dens,
                window_depth=wm.spec.end_layer - wm.spec.start_layer, retried=escalated, confidence=conf,
            )
            setattr(rec, "escalated", escalated)
            setattr(rec, "cascade_path", path)
            records.append(rec)
    else:
        # FULL 4.1 path
        for w_idx, wm in enumerate(_iter_windows()):
            lo, hi, n = wm.det_lo, wm.det_hi, wm.n_local
            buf = wm.buf
            timers.reset()
            topo = topo_id_from_wm(wm, d=d)
            with timers.input():
                np.bitwise_xor(det[lo:hi], carry[lo:hi], out=buf[:n])
                if buf.size > n:
                    buf[n:] = 0
                carry[lo:hi] = 0
                fired = np.flatnonzero(buf[:n])
                n_defects = int(fired.size)
            with timers.match():
                path, edges, action, escalated, weight = route_window(
                    wm, buf, fired, n_defects, config=config, state=state, topo=topo, gate_max=gate_max,
                )
            with timers.post():
                if action is not None:
                    part_mask, n_committed = apply_commit_action(action, carry)
                else:
                    part_mask, n_committed = commit_window_edges(edges, wm, carry, carry_forward=True)
                obs_mask ^= part_mask
            state.n_windows += 1
            # Instrumentation only, so it sits outside every timed interval.
            dens = float(n_defects) / float(n) if n else 0.0
            sample = timers.sample()
            conf = {
                "empty": 1.0, "cache": 0.9, "pair": 0.88, "clique": 0.85, "peel": 0.8,
                "escalate": 0.35, "gate_escalate": 0.3, "cost_escalate": 0.3, "residual_escalate": 0.35,
            }.get(path, 0.5)
            rec = WindowRecord(
                d=d, campaign=campaign, shot=shot, window_index=w_idx, K=int(n_committed),
                tau_input_ns=sample.tau_input_ns, tau_match_ns=sample.tau_match_ns,
                tau_post_ns=sample.tau_post_ns, tau_stage_ns=sample.tau_stage_ns,
                empty_output=(n_committed == 0), commit_lo=wm.commit_start, commit_hi=wm.commit_end,
                n_committed=n_committed, matching_weight=weight, syndrome_density=dens,
                window_depth=wm.spec.end_layer - wm.spec.start_layer, retried=escalated, confidence=conf,
            )
            setattr(rec, "escalated", escalated)
            setattr(rec, "cascade_path", path)
            records.append(rec)

    pred = mask_to_obs_array(obs_mask, n_obs)
    if observable_flips is None:
        return ShotDecodeResult(
            predicted_observables=pred[: bundle.num_observables], records=records, logical_error=False,
        )
    obs = np.asarray(observable_flips, dtype=np.uint8).ravel()
    if obs.size and pred.size < obs.size:
        pad = np.zeros(obs.size, dtype=np.uint8)
        pad[: pred.size] = pred
        pred = pad
    logical_error = bool(obs.size and np.any(pred[: obs.size] != obs))
    return ShotDecodeResult(
        predicted_observables=pred[: obs.size] if obs.size else pred[: bundle.num_observables],
        records=records, logical_error=logical_error,
    )


stream_shot_cascade = stream_shot_cascade_timed


def stream_shot_cascade_adapt(bundle, syndrome, **kwargs):
    """Research entry: CASCADE with adaptive_depth=True.

    Copies the caller's config rather than mutating it, so a config reused for
    a non-adaptive run in the same campaign is not silently switched over.
    """
    config = kwargs.pop("config", None)
    if config is None:
        config = CASCADEConfig(adaptive_depth=True)
    elif not config.adaptive_depth:
        config = replace(config, adaptive_depth=True)
    return stream_shot_cascade_timed(bundle, syndrome, config=config, **kwargs)
