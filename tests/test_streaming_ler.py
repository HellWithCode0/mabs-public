"""Streaming vs batch LER / observable agreement tests."""

from __future__ import annotations

import numpy as np
import pytest

from mabs.adaptive import AdaptiveConfig, AdaptiveState, stream_shot_adaptive_timed
from mabs.streaming import (
    ShotDecodeResult,
    batch_decode_observables,
    build_surface_code_bundle,
    get_window_matcher,
    sample_syndromes_with_observables,
    stream_shot,
)


@pytest.mark.parametrize("window_factor", [2.0, 3.0])
def test_streaming_observables_match_batch_d3(window_factor: float):
    """d=3, p=1e-3: streaming predictions must match batch on identical shots."""
    d, p, shots = 3, 1e-3, 200
    bundle = build_surface_code_bundle(d=d, noise=p, rounds=10 * d)
    dets, obs = sample_syndromes_with_observables(bundle, shots=shots, seed=101)
    batch = batch_decode_observables(bundle, dets)
    w = max(1, int(round(window_factor * d)))
    C = d
    agree = 0
    stream_err = batch_err = 0
    for i in range(shots):
        out = stream_shot(
            bundle,
            dets[i],
            window_depth=w,
            commit_stride=C,
            observable_flips=obs[i],
        )
        assert isinstance(out, ShotDecodeResult)
        if np.array_equal(out.predicted_observables, batch[i]):
            agree += 1
        if out.logical_error:
            stream_err += 1
        if np.any(batch[i] != obs[i]):
            batch_err += 1
    assert agree == shots, f"agreement {agree}/{shots}"
    assert stream_err == batch_err


def test_streaming_ler_within_2x_batch_d3():
    """Many shots: streaming w=3d LER within ~2× of batch (exact match expected)."""
    d, p, shots = 3, 1e-3, 500
    bundle = build_surface_code_bundle(d=d, noise=p, rounds=10 * d)
    dets, obs = sample_syndromes_with_observables(bundle, shots=shots, seed=202)
    batch = batch_decode_observables(bundle, dets)
    batch_err = int(np.sum(np.any(batch != obs, axis=1)))
    stream_err = 0
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
    # Exact agreement on error count at this seed/budget
    assert stream_err == batch_err
    if batch_err > 0:
        assert stream_err / shots <= 2.0 * (batch_err / shots) + 1e-12


def test_window_matcher_cache_and_truncation():
    from mabs.streaming import make_window_spec

    d = 3
    bundle = build_surface_code_bundle(d=d, noise=0.001, rounds=10 * d)
    spec = make_window_spec(bundle, 0, 3 * d, d)
    wm = get_window_matcher(bundle, spec)
    assert wm.n_local > 0
    assert wm.n_local < bundle.n_detectors
    # Cache hit
    wm2 = get_window_matcher(bundle, spec)
    assert wm2 is wm


def test_adaptive_matches_batch_d3():
    d, p, shots = 3, 1e-3, 100
    bundle = build_surface_code_bundle(d=d, noise=p, rounds=10 * d)
    dets, obs = sample_syndromes_with_observables(bundle, shots=shots, seed=7)
    batch = batch_decode_observables(bundle, dets)
    state = AdaptiveState(threshold=0.40)
    cfg = AdaptiveConfig()
    agree = 0
    for i in range(shots):
        out = stream_shot_adaptive_timed(
            bundle,
            dets[i],
            adaptive=cfg,
            state=state,
            observable_flips=obs[i],
        )
        if np.array_equal(out.predicted_observables, batch[i]):
            agree += 1
    assert agree == shots
