"""Tests for MABS-Adaptive streaming and LER vs batch."""

from __future__ import annotations

import numpy as np

from mabs.adaptive import AdaptiveConfig, AdaptiveState, stream_shot_adaptive_timed
from mabs.streaming import (
    batch_decode_observables,
    build_surface_code_bundle,
    sample_syndromes_with_observables,
    stream_shot,
)
from mabs.streaming import ShotDecodeResult


def test_fixed_w3d_matches_batch_ler_d3():
    d, p, shots = 3, 0.001, 80
    bundle = build_surface_code_bundle(d=d, noise=p, rounds=10 * d)
    dets, obs = sample_syndromes_with_observables(bundle, shots=shots, seed=11)
    batch = batch_decode_observables(bundle, dets)
    batch_err = int(np.sum(np.any(batch != obs, axis=1)))
    stream_err = 0
    agree = 0
    for i in range(shots):
        out = stream_shot(
            bundle,
            dets[i],
            window_depth=3 * d,
            commit_stride=d,
            observable_flips=obs[i],
        )
        assert isinstance(out, ShotDecodeResult)
        if out.logical_error:
            stream_err += 1
        if np.array_equal(out.predicted_observables, batch[i]):
            agree += 1
    # Near-batch: allow tiny disagreement at this shot count
    assert agree / shots >= 0.95
    assert abs(stream_err - batch_err) <= max(2, batch_err)


def test_adaptive_runs_and_tracks_retry():
    d = 3
    bundle = build_surface_code_bundle(d=d, noise=0.001, rounds=10 * d)
    dets, obs = sample_syndromes_with_observables(bundle, shots=20, seed=3)
    state = AdaptiveState(threshold=0.40)
    cfg = AdaptiveConfig(max_retries_per_window=1)
    errors = 0
    for i in range(20):
        out = stream_shot_adaptive_timed(
            bundle,
            dets[i],
            adaptive=cfg,
            state=state,
            observable_flips=obs[i],
        )
        assert out.records
        for r in out.records:
            assert r.tau_stage_ns == r.tau_input_ns + r.tau_match_ns + r.tau_post_ns
            assert 0.0 <= r.confidence <= 1.0
        if out.logical_error:
            errors += 1
    assert state.n_windows > 0
    assert 0.0 <= state.retry_rate <= 1.0
    assert errors >= 0
