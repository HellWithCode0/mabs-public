"""MABS-Adaptive: Mixture-Aware Adaptive Boundary Streaming.

ADaPT-style adaptivity on top of correct sliding-window MWPM:

1. **Pre-decode gate** (primary): from residual syndrome density in the
   upcoming peek region, choose ``w_small`` (default 2·d) or ``w_large``
   (3·d) in a *single* pass.
2. **Rare post-decode retry**: only if the small-window decode looks
   extremely hard (``should_retry``), re-decode once with ``w_large``.
3. Track observed retry rate; optionally nudge the Q threshold.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Optional
import numpy as np

from mabs.confidence import (
    ConfidenceConfig,
    confidence_score,
    predecode_confidence,
    should_retry,
)
from mabs.kernels import get_kernel
from mabs.streaming import (
    CircuitBundle,
    ShotDecodeResult,
    WindowRecord,
    _commit_edges_to_prediction,
    _edges_and_weight,
)
from mabs.timing import NestedTimers


@dataclass
class AdaptiveConfig:
    """Adaptive window policy parameters."""

    w_large_factor: float = 3.0
    w_small_factor: float = 2.0
    commit_factor: float = 1.0
    confidence: ConfidenceConfig = field(default_factory=ConfidenceConfig)
    max_retries_per_window: int = 1
    use_predecode_gate: bool = True
    # Density confidence below this → start on large window (single pass)
    predecode_threshold: float = 0.30
    tune_threshold: bool = True
    target_retry_rate: float = 0.05
    tune_step: float = 0.02
    tune_every: int = 32


@dataclass
class AdaptiveState:
    """Mutable online stats for retry-rate tracking / threshold tuning."""

    n_windows: int = 0
    n_retries: int = 0
    threshold: float = 0.40

    @property
    def retry_rate(self) -> float:
        if self.n_windows == 0:
            return 0.0
        return self.n_retries / self.n_windows


def _window_sizes(d: int, cfg: AdaptiveConfig) -> tuple[int, int, int]:
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
    """Adaptive decode; stage times accumulate across rare retries."""
    if adaptive is None:
        adaptive = AdaptiveConfig()
    if state is None:
        state = AdaptiveState(threshold=adaptive.confidence.threshold)

    kernel = get_kernel(kernel_name, auto_threshold=auto_threshold)  # type: ignore[arg-type]
    d = bundle.d
    n_layers = bundle.n_layers
    n_det = bundle.n_detectors
    n_fault = max(int(bundle.matching.num_fault_ids), 1)
    w_small, w_large, C = _window_sizes(d, adaptive)

    residual = np.asarray(syndrome, dtype=np.uint8).copy()
    pred_obs = np.zeros(n_fault, dtype=np.uint8)
    records: list[WindowRecord] = []
    w_idx = 0
    start = 0
    conf_cfg = adaptive.confidence

    while start < n_layers:
        residual_snapshot = residual.copy()
        pred_snapshot = pred_obs.copy()

        peek_end = min(start + w_large, n_layers)
        peek_active = (bundle.detector_time >= start) & (
            bundle.detector_time < peek_end
        )
        n_peek = int(peek_active.sum())
        peek_dens = (
            float(residual[peek_active].sum()) / float(n_peek) if n_peek else 0.0
        )
        pre_q = predecode_confidence(peek_dens, cfg=conf_cfg)

        # Single-pass preference: dense → large; sparse → small
        if adaptive.use_predecode_gate and pre_q < adaptive.predecode_threshold:
            initial_w = w_large
        else:
            initial_w = w_small

        attempt_w = initial_w
        retried = False
        acc_in = acc_match = acc_post = 0
        chosen: dict[str, Any] = {}

        for attempt in range(adaptive.max_retries_per_window + 1):
            residual = residual_snapshot.copy()
            pred_obs = pred_snapshot.copy()

            end = min(start + attempt_w, n_layers)
            commit_lo = start
            commit_hi = min(start + C, end)
            if end >= n_layers:
                commit_hi = end

            timers = NestedTimers()
            with timers.input():
                active = np.zeros(n_det, dtype=bool)
                for ell in range(start, end):
                    active[bundle.layers[ell]] = True
                active_idx = np.flatnonzero(active).astype(np.int64)
                syn = np.zeros(n_det, dtype=np.uint8)
                if active_idx.size:
                    syn[active_idx] = residual[active_idx]
                dens = (
                    float(syn.sum()) / float(active_idx.size)
                    if active_idx.size
                    else 0.0
                )
            with timers.match():
                edges, weight = _edges_and_weight(bundle.matching, syn)
            with timers.post():
                n_committed, edge_layers = _commit_edges_to_prediction(
                    edges,
                    bundle=bundle,
                    commit_lo=commit_lo,
                    commit_hi=commit_hi,
                    residual=residual,
                    pred_obs=pred_obs,
                )
                residual[
                    (bundle.detector_time >= commit_lo)
                    & (bundle.detector_time < commit_hi)
                ] = 0
                result = kernel(edge_layers, active_idx, commit_lo, commit_hi)

            sample = timers.sample()
            acc_in += sample.tau_input_ns
            acc_match += sample.tau_match_ns
            acc_post += sample.tau_post_ns

            q = confidence_score(
                K=result.K,
                empty_output=result.empty_output,
                matching_weight=weight,
                syndrome_density=dens,
                cfg=conf_cfg,
                distance=d,
            )
            thr = state.threshold
            chosen = dict(
                result=result,
                n_committed=n_committed,
                weight=weight,
                dens=dens,
                q=q,
                commit_lo=commit_lo,
                commit_hi=commit_hi,
                w=end - start,
                residual=residual,
                pred_obs=pred_obs,
            )

            # Retry only if we used small window and decode looks extremely hard
            need_retry = (
                attempt_w < w_large
                and should_retry(
                    K=result.K,
                    matching_weight=weight,
                    syndrome_density=dens,
                    Q=q,
                    threshold=thr,
                    distance=d,
                    cfg=conf_cfg,
                )
            )
            if (
                not need_retry
                or attempt >= adaptive.max_retries_per_window
                or end >= n_layers
            ):
                break
            attempt_w = w_large
            retried = True

        residual = chosen["residual"]
        pred_obs = chosen["pred_obs"]
        result = chosen["result"]

        state.n_windows += 1
        if retried:
            state.n_retries += 1
        if adaptive.tune_threshold and state.n_windows % max(adaptive.tune_every, 1) == 0:
            if state.retry_rate > adaptive.target_retry_rate + 0.05:
                # Too many retries → be more trusting of small windows
                state.threshold = max(0.15, state.threshold - adaptive.tune_step)
            elif state.retry_rate < adaptive.target_retry_rate - 0.03:
                state.threshold = min(0.8, state.threshold + adaptive.tune_step)

        records.append(
            WindowRecord(
                d=d,
                campaign=campaign,
                shot=shot,
                window_index=w_idx,
                K=result.K,
                tau_input_ns=acc_in,
                tau_match_ns=acc_match,
                tau_post_ns=acc_post,
                tau_stage_ns=acc_in + acc_match + acc_post,
                empty_output=result.empty_output,
                commit_lo=chosen["commit_lo"],
                commit_hi=chosen["commit_hi"],
                n_committed=chosen["n_committed"],
                matching_weight=chosen["weight"],
                syndrome_density=chosen["dens"],
                window_depth=chosen["w"],
                retried=retried,
                confidence=chosen["q"],
            )
        )
        w_idx += 1
        if chosen["commit_hi"] >= n_layers:
            break
        start += C
        if start >= n_layers or C <= 0:
            break

    obs = (
        np.asarray(observable_flips, dtype=np.uint8).ravel()
        if observable_flips is not None
        else np.zeros(bundle.num_observables, dtype=np.uint8)
    )
    pred = pred_obs[: max(obs.size, bundle.num_observables)]
    if obs.size:
        logical_error = bool(np.any(pred[: obs.size] != obs))
        pred_out = pred[: obs.size]
    else:
        logical_error = False
        pred_out = pred[: bundle.num_observables]
    return ShotDecodeResult(
        predicted_observables=pred_out,
        records=records,
        logical_error=logical_error,
    )


# Back-compat alias
stream_shot_adaptive = stream_shot_adaptive_timed
