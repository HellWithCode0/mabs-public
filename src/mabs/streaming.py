"""Correct sliding-window surface-code decode with Stim + PyMatching.

Residual overlapping recovery on full DEM + PyMatching. See package README /
results/SUMMARY.md. Re-exports graph helpers for a stable ``mabs.streaming`` API.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Optional
import numpy as np

from mabs.kernels import CommitResult, get_kernel
from mabs.timing import NestedTimers, TimingSample
from mabs.streaming_graph import (
    CircuitBundle,
    WindowRecord,
    build_surface_code_bundle,
    _build_edge_faults,
    _commit_edges_to_prediction,
    _earliest_layer,
    _edges_and_weight,
)

# Re-export public graph API
__all__ = [
    "WindowRecord",
    "CircuitBundle",
    "ShotDecodeResult",
    "build_surface_code_bundle",
    "stream_shot",
    "sample_syndromes",
    "sample_syndromes_with_observables",
    "batch_decode_observables",
]


@dataclass
class ShotDecodeResult:
    """One-shot streaming decode outcome."""

    predicted_observables: np.ndarray
    records: list[WindowRecord]
    logical_error: bool = False


def stream_shot(
    bundle: CircuitBundle,
    syndrome: np.ndarray,
    *,
    window_depth: int,
    commit_stride: int,
    kernel_name: str = "auto",
    auto_threshold: int = 64,
    campaign: int = 0,
    shot: int = 0,
    observable_flips: Optional[np.ndarray] = None,
) -> list[WindowRecord] | ShotDecodeResult:
    """Stream one shot with fixed window depth (correct residual commit).

    If ``observable_flips`` is provided, returns ``ShotDecodeResult`` including
    LER bit; otherwise returns ``list[WindowRecord]`` (back-compat).
    """
    kernel = get_kernel(kernel_name, auto_threshold=auto_threshold)  # type: ignore[arg-type]
    d = bundle.d
    n_layers = bundle.n_layers
    n_det = bundle.n_detectors
    n_fault = max(int(bundle.matching.num_fault_ids), 1)
    records: list[WindowRecord] = []
    residual = np.asarray(syndrome, dtype=np.uint8).copy()
    pred_obs = np.zeros(n_fault, dtype=np.uint8)

    w_idx = 0
    start = 0
    while start < n_layers:
        end = min(start + window_depth, n_layers)
        commit_lo = start
        commit_hi = min(start + commit_stride, end)
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
            n_active = int(active_idx.size)
            dens = float(syn.sum()) / float(n_active) if n_active else 0.0

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
            # Drop any leftover commit-region residual (explained or empty).
            residual[(bundle.detector_time >= commit_lo) & (bundle.detector_time < commit_hi)] = 0
            result: CommitResult = kernel(
                edge_layers, active_idx, commit_lo, commit_hi
            )

        sample: TimingSample = timers.sample()
        rec = WindowRecord(
            d=d,
            campaign=campaign,
            shot=shot,
            window_index=w_idx,
            K=result.K,
            tau_input_ns=sample.tau_input_ns,
            tau_match_ns=sample.tau_match_ns,
            tau_post_ns=sample.tau_post_ns,
            tau_stage_ns=sample.tau_stage_ns,
            empty_output=result.empty_output,
            commit_lo=commit_lo,
            commit_hi=commit_hi,
            n_committed=n_committed,
            matching_weight=weight,
            syndrome_density=dens,
            window_depth=end - start,
            retried=False,
            confidence=1.0,
        )
        records.append(rec)
        w_idx += 1

        if commit_hi >= n_layers:
            break
        start += commit_stride
        if start >= n_layers or commit_stride <= 0:
            break

    if observable_flips is None:
        return records

    obs = np.asarray(observable_flips, dtype=np.uint8).ravel()
    pred = pred_obs[: len(obs)] if obs.size else pred_obs[:0]
    if obs.size and pred.size < obs.size:
        pad = np.zeros(obs.size, dtype=np.uint8)
        pad[: pred.size] = pred
        pred = pad
    logical_error = bool(obs.size and np.any(pred[: obs.size] != obs))
    return ShotDecodeResult(
        predicted_observables=pred[: obs.size] if obs.size else pred_obs[: bundle.num_observables],
        records=records,
        logical_error=logical_error,
    )


def sample_syndromes(
    bundle: CircuitBundle,
    shots: int,
    seed: int,
) -> np.ndarray:
    """Sample detector syndrome bits; shape (shots, n_detectors)."""
    sampler = bundle.circuit.compile_detector_sampler(seed=seed)
    dets = sampler.sample(shots=shots, bit_packed=False)
    return np.asarray(dets, dtype=np.uint8)


def sample_syndromes_with_observables(
    bundle: CircuitBundle,
    shots: int,
    seed: int,
) -> tuple[np.ndarray, np.ndarray]:
    """Sample detectors and observable flips; shapes (shots, n_det), (shots, n_obs)."""
    sampler = bundle.circuit.compile_detector_sampler(seed=seed)
    dets, obs = sampler.sample(shots=shots, separate_observables=True, bit_packed=False)
    return np.asarray(dets, dtype=np.uint8), np.asarray(obs, dtype=np.uint8)


def batch_decode_observables(
    bundle: CircuitBundle,
    syndromes: np.ndarray,
) -> np.ndarray:
    """Batch MWPM observable predictions; shape (shots, n_obs)."""
    return np.asarray(bundle.matching.decode_batch(syndromes), dtype=np.uint8)
