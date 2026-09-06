"""Sparse Local Escalation Matching (SLEM) — MABS v3.1 streaming decoder.

Default: exact 1–2 defect local MWPM + single blossom escalate.
Optional cluster local via use_cluster_local (needs local_decode_induced).
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
from mabs.v3.escalate import EscalationPolicy, adapt_policy
from mabs.v3.local_decode import blossom_edges_only, local_decode_few_defects


@dataclass
class SLEMConfig:
    window_factor: float = 3.0
    commit_factor: float = 1.0
    policy: EscalationPolicy = field(default_factory=EscalationPolicy)
    prefer_correctness: bool = True
    local_defect_cap: int = 2
    use_cluster_local: bool = False
    check_syndrome: bool = True
    prewarm_graphs: bool = True


@dataclass
class SLEMState:
    n_windows: int = 0
    n_empty: int = 0
    n_local: int = 0
    n_escalate: int = 0
    policy: EscalationPolicy = field(default_factory=EscalationPolicy)
    graphs_ready: bool = False

    @property
    def escalate_rate(self) -> float:
        return 0.0 if self.n_windows == 0 else self.n_escalate / self.n_windows

    @property
    def empty_rate(self) -> float:
        return 0.0 if self.n_windows == 0 else self.n_empty / self.n_windows

    @property
    def local_rate(self) -> float:
        return 0.0 if self.n_windows == 0 else self.n_local / self.n_windows


def prewarm_slem_graphs(bundle: CircuitBundle, window_depth: int, commit_stride: int) -> List:
    windows = ensure_window_schedule(bundle, window_depth, commit_stride)
    for wm in windows:
        _get_graph_cache(wm)
    return windows


def stream_shot_slem_timed(
    bundle: CircuitBundle,
    syndrome: np.ndarray,
    *,
    config: Optional[SLEMConfig] = None,
    state: Optional[SLEMState] = None,
    campaign: int = 0,
    shot: int = 0,
    observable_flips: Optional[np.ndarray] = None,
) -> ShotDecodeResult:
    if config is None:
        config = SLEMConfig()
    if state is None:
        state = SLEMState(policy=EscalationPolicy(
            max_local_cluster=config.policy.max_local_cluster,
            max_local_defects=config.policy.max_local_defects,
            max_local_density=config.policy.max_local_density,
            cluster_radius=config.policy.cluster_radius,
            target_escalate_rate=config.policy.target_escalate_rate,
            tune=config.policy.tune,
            tune_every=config.policy.tune_every,
            tune_step=config.policy.tune_step,
        ))

    d = bundle.d
    n_det = bundle.n_detectors
    n_obs = max(int(bundle.num_observables), 1)
    w = max(1, int(round(config.window_factor * d)))
    C = max(1, int(round(config.commit_factor * d)))
    policy = state.policy
    local_cap = max(0, min(int(config.local_defect_cap), int(policy.max_local_defects)))

    if config.prewarm_graphs and not state.graphs_ready:
        windows = prewarm_slem_graphs(bundle, w, C)
        state.graphs_ready = True
    else:
        windows = ensure_window_schedule(bundle, w, C)

    det = np.asarray(syndrome, dtype=np.uint8).ravel()
    carry = np.zeros(n_det + 1, dtype=np.uint8)
    obs_mask = 0
    records: list[WindowRecord] = []

    for w_idx, wm in enumerate(windows):
        lo, hi, n = wm.det_lo, wm.det_hi, wm.n_local
        buf = wm.buf
        timers = NestedTimers()
        escalated = False
        path = "empty"
        edges = np.zeros((0, 2), dtype=np.int64)
        weight = 0.0

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
                used_local = False
                if local_cap > 0 and n_defects <= local_cap and dens <= policy.max_local_density:
                    local_edges = local_decode_few_defects(wm, buf, fired)
                    if local_edges is not None:
                        edges = local_edges
                        weight = float(edges.shape[0])
                        path = "local"
                        state.n_local += 1
                        used_local = True
                if not used_local and config.use_cluster_local:
                    try:
                        from mabs.v3.local_decode import local_decode_induced, syndrome_cleared_by_edges
                        if n_defects <= policy.max_local_defects:
                            local_edges = local_decode_induced(
                                wm, buf, fired,
                                max_cluster=policy.max_local_cluster,
                                max_defects=policy.max_local_defects,
                            )
                            if local_edges is not None:
                                if (not config.check_syndrome) or syndrome_cleared_by_edges(wm.n_local, fired, local_edges):
                                    edges = local_edges
                                    weight = float(edges.shape[0])
                                    path = "local"
                                    state.n_local += 1
                                    used_local = True
                    except ImportError:
                        pass
                if not used_local:
                    edges, weight = blossom_edges_only(wm, buf)
                    escalated = True
                    path = "escalate"
                    state.n_escalate += 1

        with timers.post():
            part_mask, n_committed = commit_window_edges(edges, wm, carry, carry_forward=True)
            obs_mask ^= part_mask

        state.n_windows += 1
        if policy.tune and state.n_windows % max(policy.tune_every, 1) == 0:
            adapt_policy(policy, state.escalate_rate)

        sample = timers.sample()
        K = int(edges.shape[0])
        conf = 1.0 if path == "empty" else (0.85 if path == "local" else 0.35)
        rec = WindowRecord(
            d=d, campaign=campaign, shot=shot, window_index=w_idx, K=K,
            tau_input_ns=sample.tau_input_ns, tau_match_ns=sample.tau_match_ns,
            tau_post_ns=sample.tau_post_ns, tau_stage_ns=sample.tau_stage_ns,
            empty_output=(n_committed == 0), commit_lo=wm.commit_start,
            commit_hi=wm.commit_end, n_committed=n_committed,
            matching_weight=weight, syndrome_density=dens,
            window_depth=wm.spec.end_layer - wm.spec.start_layer,
            retried=escalated, confidence=conf,
        )
        setattr(rec, "escalated", escalated)
        setattr(rec, "n_clusters", 1 if n_defects else 0)
        setattr(rec, "slem_path", path)
        records.append(rec)

    pred = mask_to_obs_array(obs_mask, n_obs)
    if observable_flips is None:
        return ShotDecodeResult(predicted_observables=pred[: bundle.num_observables], records=records, logical_error=False)
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


stream_shot_slem = stream_shot_slem_timed
