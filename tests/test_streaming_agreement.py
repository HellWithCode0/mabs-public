"""Streaming vs batch agreement on identical shots (window DEM + carry)."""

from __future__ import annotations

import numpy as np

from mabs.streaming import (
    batch_decode_observables,
    build_surface_code_bundle,
    sample_syndromes_with_observables,
    stream_shot,
)


def test_stream_w3d_agrees_with_batch_d3():
    d = 3
    p = 0.002
    shots = 100
    bundle = build_surface_code_bundle(d=d, noise=p, rounds=10 * d)
    dets, obs = sample_syndromes_with_observables(bundle, shots=shots, seed=7)
    pred_b = batch_decode_observables(bundle, dets)
    w, C = 3 * d, d
    agree = 0
    for i in range(shots):
        out = stream_shot(
            bundle,
            dets[i],
            window_depth=w,
            commit_stride=C,
            observable_flips=obs[i],
        )
        if np.array_equal(out.predicted_observables[: obs.shape[1]], pred_b[i]):
            agree += 1
    # Truncated window DEM + carry should match batch on almost all shots
    assert agree / shots >= 0.95, f"agreement {agree}/{shots} below 95%"


def test_stream_w3d_near_batch_ler_d5_smoke():
    d = 5
    p = 0.002
    shots = 40
    bundle = build_surface_code_bundle(d=d, noise=p, rounds=10 * d)
    dets, obs = sample_syndromes_with_observables(bundle, shots=shots, seed=11)
    pred_b = batch_decode_observables(bundle, dets)
    batch_err = int(np.sum(np.any(pred_b != obs, axis=1)))
    w, C = 3 * d, d
    stream_err = 0
    for i in range(shots):
        out = stream_shot(
            bundle,
            dets[i],
            window_depth=w,
            commit_stride=C,
            observable_flips=obs[i],
        )
        if out.logical_error:
            stream_err += 1
    # At this shot count, stream errors should not explode vs batch
    assert stream_err <= batch_err + 2, (
        f"stream_err={stream_err} batch_err={batch_err}"
    )
