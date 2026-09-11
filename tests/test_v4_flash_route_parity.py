"""Guard: the inlined FLASH hot path must agree with route_window_flash.

``stream_shot_cascade`` inlines the FLASH route to avoid a call on the hot
path, so the same decision logic exists twice. Nothing else exercises
``route_window_flash``, so without this test the two copies can drift apart
silently and the readable one stops describing the measured one.
"""

from __future__ import annotations

import numpy as np

from mabs.streaming import (
    build_surface_code_bundle,
    ensure_window_schedule,
    sample_syndromes_with_observables,
)
from mabs.streaming_graph import mask_to_obs_array
from mabs.streaming_windows import commit_window_edges
from mabs.v4.cascade import (
    CASCADEConfig,
    CASCADEState,
    prewarm_cascade_graphs,
    stream_shot_cascade,
)
from mabs.v4.cascade_route import route_window_flash, route_window_flash_sticky
from mabs.v4.commit_action import apply_commit_action
from mabs.v4.flash_lut import extract_defects, get_flash_lut


def _reference_shot(bundle, syndrome, windows, config, state):
    """Same shot, driven through route_window_flash instead of the inline copy."""
    state.reset_shot_sticky()
    n_det = bundle.n_detectors
    det = np.asarray(syndrome, dtype=np.uint8).ravel()
    carry = np.zeros(n_det + 1, dtype=np.uint8)
    obs_mask = 0
    for wm in windows:
        lo, hi, n = wm.det_lo, wm.det_hi, wm.n_local
        buf = wm.buf
        lut = get_flash_lut(wm)
        np.bitwise_xor(det[lo:hi], carry[lo:hi], out=buf[:n])
        if buf.size > n:
            buf[n:] = 0
        carry[lo:hi] = 0
        if config.sticky_blossom and state.sticky_this_shot:
            _path, edges, action, _esc, _w = route_window_flash_sticky(
                wm, buf, state=state
            )
        else:
            k = extract_defects(
                buf, n, lut._defect_buf, prefer_numba=config.prefer_numba, cap=3
            )
            _path, edges, action, _esc, _w = route_window_flash(
                wm, buf, lut._defect_buf, k, config=config, state=state, lut=lut
            )
        if action is not None:
            part_mask, _n_committed = apply_commit_action(action, carry)
        else:
            part_mask, _n_committed = commit_window_edges(
                edges, wm, carry, carry_forward=True
            )
        obs_mask ^= part_mask
        state.n_windows += 1
    return mask_to_obs_array(obs_mask, max(int(bundle.num_observables), 1))


def _run_parity(sticky: bool, cluster: bool = False):
    d, p, shots = 3, 2e-3, 40
    cfg = CASCADEConfig(sticky_blossom=sticky, cluster_route=cluster)

    # Two bundles, because a pair-LUT fill on one run would otherwise warm the
    # LUT the other run reads and turn a pair_fill into a pair.
    inline_bundle = build_surface_code_bundle(d=d, noise=p, rounds=10 * d)
    ref_bundle = build_surface_code_bundle(d=d, noise=p, rounds=10 * d)
    syn, _obs = sample_syndromes_with_observables(inline_bundle, shots=shots, seed=77)
    prewarm_cascade_graphs(inline_bundle, 3 * d, d, config=cfg)
    prewarm_cascade_graphs(ref_bundle, 3 * d, d, config=cfg)
    ref_windows = ensure_window_schedule(ref_bundle, 3 * d, d)

    inline_state = CASCADEState()
    inline_state.graphs_ready = True
    ref_state = CASCADEState()
    ref_state.graphs_ready = True

    for s in range(shots):
        out = stream_shot_cascade(
            inline_bundle, syn[s], config=cfg, state=inline_state, shot=s
        )
        ref_pred = _reference_shot(ref_bundle, syn[s], ref_windows, cfg, ref_state)
        assert np.array_equal(
            out.predicted_observables, ref_pred[: out.predicted_observables.size]
        ), f"shot {s}: inline and route_window_flash disagree on observables"

    assert inline_state.route_counts == ref_state.route_counts
    assert inline_state.n_windows == ref_state.n_windows
    assert inline_state.n_escalate == ref_state.n_escalate
    assert inline_state.n_empty == ref_state.n_empty
    assert inline_state.n_pair == ref_state.n_pair
    assert inline_state.n_boundary == ref_state.n_boundary
    assert inline_state.n_sticky == ref_state.n_sticky
    assert inline_state.n_cluster == ref_state.n_cluster
    assert inline_state.cluster_stats.as_dict() == ref_state.cluster_stats.as_dict()
    # The run must actually reach the branches this test is guarding.
    assert inline_state.route_counts.get("escalate", 0) > 0
    assert inline_state.n_empty > 0
    if cluster:
        assert inline_state.n_cluster > 0
    else:
        assert inline_state.n_cluster == 0


def test_inline_flash_matches_route_window_flash():
    _run_parity(sticky=False)


def test_inline_flash_matches_route_window_flash_sticky():
    _run_parity(sticky=True)


def test_inline_flash_matches_route_window_flash_cluster():
    _run_parity(sticky=False, cluster=True)


def test_inline_flash_matches_route_window_flash_cluster_sticky():
    _run_parity(sticky=True, cluster=True)
