"""Correct sliding-window surface-code decode with Stim + PyMatching.

Uses truncated per-window DEMs (past-drop / future-truncate) with carry-forward
commit — see ``streaming_graph`` and package README / results/SUMMARY.md.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Optional
import numpy as np

from mabs.timing import NestedTimers, TimingSample
from mabs.streaming_graph import (
    CircuitBundle,
    WindowRecord,
    build_surface_code_bundle,
    commit_window_edges,
    ensure_window_schedule,
    get_window_matcher,
    make_window_spec,
    mask_to_obs_array,
    window_match_edges,
    _build_edge_faults,
    _commit_edges_to_prediction,
    _earliest_layer,
    _edges_and_weight,
)

__all__ = [
    "WindowRecord",
    "CircuitBundle",
    "ShotDecodeResult",
    "build_surface_code_bundle",
    "stream_shot",
    "sample_syndromes",
    "sample_syndromes_with_observables",
    "batch_decode_observables",
    "ensure_window_schedule",
    "get_window_matcher",
    "make_window_spec",
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
):
    """Stream one shot with fixed window depth (truncated window DEM + carry)."""
    del kernel_name, auto_threshold
    d = bundle.d
    n_det = bundle.n_detectors
    n_obs = max(int(bundle.num_observables), 1)
    records: list[WindowRecord] = []

    windows = ensure_window_schedule(bundle, window_depth, commit_stride)
    det = np.asarray(syndrome, dtype=np.uint8).ravel()
    carry = np.zeros(n_det + 1, dtype=np.uint8)
    obs_mask = 0

    for w_idx, wm in enumerate(windows):
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

        sample: TimingSample = timers.sample()
        K = int(edges.shape[0])
        records.append(WindowRecord(
            d=d, campaign=campaign, shot=shot, window_index=w_idx, K=K,
            tau_input_ns=sample.tau_input_ns, tau_match_ns=sample.tau_match_ns,
            tau_post_ns=sample.tau_post_ns, tau_stage_ns=sample.tau_stage_ns,
            empty_output=(n_committed == 0), commit_lo=wm.commit_start,
            commit_hi=wm.commit_end, n_committed=n_committed, matching_weight=weight,
            syndrome_density=dens, window_depth=wm.spec.end_layer - wm.spec.start_layer,
            retried=False, confidence=1.0,
        ))

    pred = mask_to_obs_array(obs_mask, n_obs)
    if observable_flips is None:
        return records

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


def sample_syndromes(bundle: CircuitBundle, shots: int, seed: int) -> np.ndarray:
    sampler = bundle.circuit.compile_detector_sampler(seed=seed)
    return np.asarray(sampler.sample(shots=shots, bit_packed=False), dtype=np.uint8)


def sample_syndromes_with_observables(bundle: CircuitBundle, shots: int, seed: int):
    sampler = bundle.circuit.compile_detector_sampler(seed=seed)
    dets, obs = sampler.sample(shots=shots, separate_observables=True, bit_packed=False)
    return np.asarray(dets, dtype=np.uint8), np.asarray(obs, dtype=np.uint8)


def batch_decode_observables(bundle: CircuitBundle, syndromes: np.ndarray) -> np.ndarray:
    return np.asarray(bundle.matching.decode_batch(syndromes), dtype=np.uint8)
