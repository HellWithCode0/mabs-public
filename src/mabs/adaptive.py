"""MABS-Adaptive: Mixture-Aware Adaptive Boundary Streaming.

ADaPT-style adaptivity on truncated window DEMs + carry-forward.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Optional
import numpy as np

from mabs.confidence import (
    ConfidenceConfig, confidence_score, predecode_confidence, should_retry,
)
from mabs.streaming import CircuitBundle, ShotDecodeResult, WindowRecord
from mabs.streaming_graph import (
    commit_window_edges, get_window_matcher, make_window_spec,
    mask_to_obs_array, window_match_edges,
)
from mabs.timing import NestedTimers


@dataclass
class AdaptiveConfig:
    w_large_factor: float = 3.0
    w_small_factor: float = 2.0
    commit_factor: float = 1.0
    confidence: ConfidenceConfig = field(default_factory=ConfidenceConfig)
    max_retries_per_window: int = 1
    use_predecode_gate: bool = True
    predecode_threshold: float = 0.30
    tune_threshold: bool = True
    target_retry_rate: float = 0.05
    tune_step: float = 0.02
    tune_every: int = 32


@dataclass
class AdaptiveState:
    n_windows: int = 0
    n_retries: int = 0
    threshold: float = 0.40

    @property
    def retry_rate(self) -> float:
        return 0.0 if self.n_windows == 0 else self.n_retries / self.n_windows


def _window_sizes(d: int, cfg: AdaptiveConfig):
    w_large = max(1, int(round(cfg.w_large_factor * d)))
    w_small = max(1, int(round(cfg.w_small_factor * d)))
    if w_small > w_large:
        w_small = w_large
    C = max(1, int(round(cfg.commit_factor * d)))
    return w_small, w_large, C


def stream_shot_adaptive_timed(
    bundle: CircuitBundle,
    syndrome: np.ndarray,
    *,
    adaptive: Optional[AdaptiveConfig] = None,
    state: Optional[AdaptiveState] = None,
    kernel_name: str = "auto",
    auto_threshold: int = 64,
    campaign: int = 0,
    shot: int = 0,
    observable_flips: Optional[np.ndarray] = None,
) -> ShotDecodeResult:
    del kernel_name, auto_threshold
    if adaptive is None:
        adaptive = AdaptiveConfig()
    if state is None:
        state = AdaptiveState(threshold=adaptive.confidence.threshold)

    d = bundle.d
    n_layers = bundle.n_layers
    n_det = bundle.n_detectors
    n_obs = max(int(bundle.num_observables), 1)
    w_small, w_large, C = _window_sizes(d, adaptive)

    det = np.asarray(syndrome, dtype=np.uint8).ravel()
    carry = np.zeros(n_det + 1, dtype=np.uint8)
    obs_mask = 0
    records: list[WindowRecord] = []
    w_idx = 0
    start = 0
    conf_cfg = adaptive.confidence

    while start < n_layers:
        carry_snapshot = carry.copy()
        obs_snapshot = obs_mask

        peek_end = min(start + w_large, n_layers)
        peek_lo = int(bundle.layer_lo[start])
        peek_hi = int(bundle.layer_lo[peek_end])
        if peek_hi > peek_lo:
            eff = np.bitwise_xor(det[peek_lo:peek_hi], carry[peek_lo:peek_hi])
            peek_dens = float(eff.sum()) / float(peek_hi - peek_lo)
        else:
            peek_dens = 0.0
        pre_q = predecode_confidence(peek_dens, cfg=conf_cfg)
        initial_w = w_large if (adaptive.use_predecode_gate and pre_q < adaptive.predecode_threshold) else w_small

        attempt_w = initial_w
        retried = False
        acc_in = acc_match = acc_post = 0
        chosen: dict[str, Any] = {}

        for attempt in range(adaptive.max_retries_per_window + 1):
            carry = carry_snapshot.copy()
            obs_mask = obs_snapshot
            spec = make_window_spec(bundle, start, attempt_w, C, index=w_idx)
            wm = get_window_matcher(bundle, spec)
            lo, hi, n = wm.det_lo, wm.det_hi, wm.n_local
            buf = wm.buf
            timers = NestedTimers()
            with timers.input():
                np.bitwise_xor(det[lo:hi], carry[lo:hi], out=buf[:n])
                if buf.size > n:
                    buf[n:] = 0
                carry[lo:hi] = 0
                dens = float(buf[:n].sum()) / float(n) if n else 0.0
            with timers.match():
                edges, weight = window_match_edges(wm, buf)
            with timers.post():
                part_mask, n_committed = commit_window_edges(edges, wm, carry, carry_forward=True)
                obs_mask ^= part_mask
            sample = timers.sample()
            acc_in += sample.tau_input_ns
            acc_match += sample.tau_match_ns
            acc_post += sample.tau_post_ns
            K = int(edges.shape[0])
            empty = n_committed == 0
            q = confidence_score(K=K, empty_output=empty, matching_weight=weight,
                                 syndrome_density=dens, cfg=conf_cfg, distance=d)
            chosen = dict(
                K=K, empty=empty, n_committed=n_committed, weight=weight, dens=dens, q=q,
                commit_lo=wm.commit_start, commit_hi=wm.commit_end,
                w=spec.end_layer - spec.start_layer, carry=carry, obs_mask=obs_mask,
            )
            need_retry = (
                attempt_w < w_large and should_retry(
                    K=K, matching_weight=weight, syndrome_density=dens, Q=q,
                    threshold=state.threshold, distance=d, cfg=conf_cfg,
                )
            )
            if (not need_retry or attempt >= adaptive.max_retries_per_window or spec.end_layer >= n_layers):
                break
            attempt_w = w_large
            retried = True

        carry = chosen["carry"]
        obs_mask = chosen["obs_mask"]
        state.n_windows += 1
        if retried:
            state.n_retries += 1
        if adaptive.tune_threshold and state.n_windows % max(adaptive.tune_every, 1) == 0:
            if state.retry_rate > adaptive.target_retry_rate + 0.05:
                state.threshold = max(0.15, state.threshold - adaptive.tune_step)
            elif state.retry_rate < adaptive.target_retry_rate - 0.03:
                state.threshold = min(0.8, state.threshold + adaptive.tune_step)

        records.append(WindowRecord(
            d=d, campaign=campaign, shot=shot, window_index=w_idx, K=chosen["K"],
            tau_input_ns=acc_in, tau_match_ns=acc_match, tau_post_ns=acc_post,
            tau_stage_ns=acc_in + acc_match + acc_post, empty_output=chosen["empty"],
            commit_lo=chosen["commit_lo"], commit_hi=chosen["commit_hi"],
            n_committed=chosen["n_committed"], matching_weight=chosen["weight"],
            syndrome_density=chosen["dens"], window_depth=chosen["w"],
            retried=retried, confidence=chosen["q"],
        ))
        w_idx += 1
        if chosen["commit_hi"] >= n_layers:
            break
        start += C
        if start >= n_layers or C <= 0:
            break

    pred = mask_to_obs_array(obs_mask, n_obs)
    obs = np.asarray(observable_flips, dtype=np.uint8).ravel() if observable_flips is not None else np.zeros(bundle.num_observables, dtype=np.uint8)
    if obs.size:
        if pred.size < obs.size:
            pad = np.zeros(obs.size, dtype=np.uint8)
            pad[: pred.size] = pred
            pred = pad
        logical_error = bool(np.any(pred[: obs.size] != obs))
        pred_out = pred[: obs.size]
    else:
        logical_error = False
        pred_out = pred[: bundle.num_observables]
    return ShotDecodeResult(predicted_observables=pred_out, records=records, logical_error=logical_error)


stream_shot_adaptive = stream_shot_adaptive_timed
