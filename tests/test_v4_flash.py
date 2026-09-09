"""Tests for MABS v4.2 CASCADE FLASH (default)."""
from __future__ import annotations

import numpy as np
import pytest

from mabs.streaming import (
    batch_decode_observables,
    build_surface_code_bundle,
    ensure_window_schedule,
    sample_syndromes_with_observables,
)
from mabs.v3.cluster import _get_graph_cache
from mabs.v3.local_decode import blossom_edges_only, syndrome_cleared_by_edges
from mabs.v4.commit_action import apply_commit_action, edges_to_commit_action
from mabs.v4.cascade import CASCADEConfig, CASCADEState, stream_shot_cascade, prewarm_cascade_graphs
from mabs.v4.flash_lut import (
    FlashCommitLUT,
    get_flash_lut,
    prewarm_flash_lut,
    extract_defects,
    has_numba,
)


def test_flash_is_default():
    cfg = CASCADEConfig()
    assert cfg.is_flash()
    assert cfg.mode == "flash"
    assert cfg.sticky_blossom is False
    assert not hasattr(cfg, "defer_hard") or getattr(cfg, "defer_hard", False) is False


def test_full_mode_opt_in():
    assert CASCADEConfig(mode="full").is_flash() is False
    assert CASCADEConfig(use_flash=False).is_flash() is False
    assert CASCADEConfig(mode="full", use_flash=True).is_flash() is True


def test_lut_boundary_and_pair_vs_blossom():
    d = 5
    bundle = build_surface_code_bundle(d=d, noise=0.001, rounds=10 * d)
    wms = ensure_window_schedule(bundle, 3 * d, d)
    wm = wms[0]
    lut = prewarm_flash_lut(wm, pair_radius=2)
    rng = np.random.default_rng(42)
    n = wm.n_local
    buf = wm.buf
    for k in (0, 1, 2):
        for _ in range(30):
            buf[:] = 0
            dets = rng.choice(n, size=k, replace=False) if k else []
            for di in dets:
                buf[int(di)] = 1
            if k == 0:
                assert not np.count_nonzero(buf[:n])
                continue
            edges_b, _ = blossom_edges_only(wm, buf)
            carry1 = np.zeros(bundle.n_detectors + 1, dtype=np.uint8)
            carry2 = np.zeros_like(carry1)
            from mabs.streaming_windows import commit_window_edges
            obs_b, n_b = commit_window_edges(edges_b, wm, carry1, carry_forward=True)
            if k == 1:
                act = lut.get_boundary(int(dets[0]))
            else:
                act = lut.get_pair(int(dets[0]), int(dets[1]))
                if act is None:
                    act = lut.fill_pair_from_edges(int(dets[0]), int(dets[1]), edges_b)
            assert act is not None
            obs_a, n_a = apply_commit_action(act, carry2)
            assert obs_a == obs_b
            assert n_a == n_b
            assert np.array_equal(carry1, carry2)


def test_sticky_blossom_behavior():
    d, p = 3, 1e-3
    bundle = build_surface_code_bundle(d=d, noise=p, rounds=10 * d)
    syn = np.zeros(bundle.n_detectors, dtype=np.uint8)
    wms = ensure_window_schedule(bundle, 3 * d, d)
    assert len(wms) >= 3
    wm1 = wms[1]
    n_fire = min(8, wm1.n_local)
    syn[wm1.det_lo : wm1.det_lo + n_fire] = 1
    wm2 = wms[2]
    syn[wm2.det_lo + 1] = 1
    obs = np.zeros(bundle.num_observables, dtype=np.uint8)
    state = CASCADEState()
    cfg = CASCADEConfig(sticky_blossom=True)
    stream_shot_cascade(bundle, syn, config=cfg, state=state, observable_flips=obs)
    assert state.n_escalate >= 1
    assert state.n_sticky >= 1
    syn0 = np.zeros(bundle.n_detectors, dtype=np.uint8)
    stream_shot_cascade(bundle, syn0, config=cfg, state=state, observable_flips=obs)
    assert state.sticky_this_shot is False


def test_extract_defects_matches_flatnonzero():
    n = 200
    buf = np.zeros(n + 5, dtype=np.uint8)
    buf[3] = 1
    buf[17] = 1
    buf[100] = 1
    out = np.empty(n, dtype=np.int64)
    k = extract_defects(buf, n, out, prefer_numba=True)
    assert k == 3
    assert list(out[:k]) == [3, 17, 100]


def test_flash_ler_agrees_batch_d3():
    d, p, shots = 3, 0.001, 60
    bundle = build_surface_code_bundle(d=d, noise=p, rounds=10 * d)
    syn, obs = sample_syndromes_with_observables(bundle, shots=shots, seed=11)
    batch_pred = batch_decode_observables(bundle, syn)
    state = CASCADEState()
    cfg = CASCADEConfig()
    prewarm_cascade_graphs(bundle, 3 * d, d, config=cfg)
    state.graphs_ready = True
    errors = 0
    batch_errors = 0
    for s in range(shots):
        out = stream_shot_cascade(bundle, syn[s], config=cfg, state=state, shot=s, observable_flips=obs[s])
        if out.logical_error:
            errors += 1
        if np.any(batch_pred[s] != obs[s]):
            batch_errors += 1
    assert errors == batch_errors
    assert CASCADEConfig().is_flash()


def test_numba_optional_flag():
    assert isinstance(has_numba(), bool)


def test_flash_records_true_syndrome_density():
    """syndrome_density must not inherit the cap=3 of the capped defect scan.

    The FLASH scan stops at three defects because that is all the route needs.
    syndrome_density is a reported record field, so it has to carry the real
    count, not the cap.
    """
    d = 3
    bundle = build_surface_code_bundle(d=d, noise=0.001, rounds=10 * d)
    wms = ensure_window_schedule(bundle, 3 * d, d)
    wm = wms[0]
    n_fire = min(12, wm.n_local)
    assert n_fire > 3, "need more than cap defects for this test to mean anything"
    syn = np.zeros(bundle.n_detectors, dtype=np.uint8)
    syn[wm.det_lo : wm.det_lo + n_fire] = 1
    state = CASCADEState()
    out = stream_shot_cascade(bundle, syn, config=CASCADEConfig(), state=state)
    rec = out.records[0]
    assert rec.syndrome_density == pytest.approx(n_fire / wm.n_local)
    assert rec.syndrome_density > 3.0 / wm.n_local


def test_flash_sticky_records_true_syndrome_density():
    """The sticky path never counts defects, so it must not report zero either."""
    d = 3
    bundle = build_surface_code_bundle(d=d, noise=0.001, rounds=10 * d)
    wms = ensure_window_schedule(bundle, 3 * d, d)
    syn = np.zeros(bundle.n_detectors, dtype=np.uint8)
    syn[wms[0].det_lo : wms[0].det_lo + 8] = 1
    n_fire_1 = 5
    syn[wms[1].det_lo : wms[1].det_lo + n_fire_1] = 1
    state = CASCADEState()
    out = stream_shot_cascade(
        bundle, syn, config=CASCADEConfig(sticky_blossom=True), state=state
    )
    assert state.n_sticky >= 1
    sticky_records = [
        r for r in out.records if getattr(r, "cascade_path", "") == "sticky"
    ]
    assert sticky_records, "expected at least one sticky window"
    assert all(r.syndrome_density > 0.0 for r in sticky_records)
