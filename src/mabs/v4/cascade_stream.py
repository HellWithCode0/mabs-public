"""CASCADE streaming shot decode (v4.1 router)."""
from __future__ import annotations
from typing import Optional
import numpy as np
from mabs.streaming import CircuitBundle, ShotDecodeResult, WindowRecord
from mabs.streaming_graph import ensure_window_schedule, mask_to_obs_array
from mabs.timing import NestedTimers
from mabs.v4.commit_action import apply_commit_action
from mabs.v4.exact_pattern_cache import ExactPatternCache, topo_id_from_wm
from mabs.v4.cascade_core import (
    CASCADEConfig, CASCADEState, _window_depth_choice, prewarm_cascade_graphs,
)
from mabs.v4.cascade_route import route_window

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

    d = bundle.d
    n_det = bundle.n_detectors
    n_obs = max(int(bundle.num_observables), 1)
    C = max(1, int(round(config.commit_factor * d)))
    gate_max = config.resolved_gate_max()

    if config.adaptive_depth:
        windows = None
        w_default = max(1, int(round(config.window_factor * d)))
        if config.prewarm_graphs and not state.graphs_ready:
            prewarm_cascade_graphs(bundle, w_default, C)
            w_small = max(1, int(round(config.w_small_factor * d)))
            prewarm_cascade_graphs(bundle, w_small, C)
            state.graphs_ready = True
    else:
        w = max(1, int(round(config.window_factor * d)))
        if config.prewarm_graphs and not state.graphs_ready:
            windows = prewarm_cascade_graphs(bundle, w, C)
            state.graphs_ready = True
        else:
            windows = ensure_window_schedule(bundle, w, C)

    det = np.asarray(syndrome, dtype=np.uint8).ravel()
    carry = np.zeros(n_det + 1, dtype=np.uint8)
    obs_mask = 0
    records: list[WindowRecord] = []

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

    for w_idx, wm in enumerate(_iter_windows()):
        lo, hi, n = wm.det_lo, wm.det_hi, wm.n_local
        buf = wm.buf
        timers = NestedTimers()
        topo = topo_id_from_wm(wm, d=d)
        with timers.input():
            np.bitwise_xor(det[lo:hi], carry[lo:hi], out=buf[:n])
            if buf.size > n:
                buf[n:] = 0
            carry[lo:hi] = 0
            fired = np.flatnonzero(buf[:n])
            n_defects = int(fired.size)
            dens = float(n_defects) / float(n) if n else 0.0
        with timers.match():
            path, edges, action, escalated, weight = route_window(
                wm, buf, fired, n_defects, config=config, state=state, topo=topo, gate_max=gate_max,
            )
        with timers.post():
            if action is not None:
                part_mask, n_committed = apply_commit_action(action, carry)
            else:
                from mabs.streaming_windows import commit_window_edges
                part_mask, n_committed = commit_window_edges(edges, wm, carry, carry_forward=True)
            obs_mask ^= part_mask
        state.n_windows += 1
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
    """Research entry: CASCADE with adaptive_depth=True."""
    config = kwargs.pop("config", None)
    if config is None:
        config = CASCADEConfig(adaptive_depth=True)
    else:
        config.adaptive_depth = True
    return stream_shot_cascade_timed(bundle, syndrome, config=config, **kwargs)
