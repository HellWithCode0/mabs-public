"""Tests for MABS v4 CASCADE (iso-cache, clique MWPM, LER smoke)."""

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
from mabs.v3.local_decode import syndrome_cleared_by_edges
from mabs.v4.iso_cache import IsoCache, canonicalize_defects
from mabs.v4.clique_mwpm import clique_mwpm_edges
from mabs.v4.cascade import CASCADEConfig, CASCADEState, stream_shot_cascade


def test_canonicalize_relative():
    from mabs.v4.iso_cache import relative_key
    key, origin = canonicalize_defects([10, 12, 15])
    assert origin == 10
    assert key == (10, 12, 15)  # absolute sorted
    assert relative_key([10, 12, 15]) == (0, 2, 5)
    assert relative_key([100, 102, 105]) == (0, 2, 5)


def test_iso_cache_hit_miss_lru():
    cache = IsoCache(max_size=2)
    edges = np.asarray([[10, 11], [12, 13]], dtype=np.int64)
    cache.store([10, 11, 12, 13], edges, n_local=20)
    hit = cache.lookup([10, 11, 12, 13], n_local=20)
    assert hit is not None
    assert cache.hits == 1
    # different absolute pattern = miss (no unsafe translation)
    miss_t = cache.lookup([50, 51, 52, 53], n_local=60)
    assert miss_t is None
    assert cache.misses == 1
    miss = cache.lookup([1, 2, 3], n_local=10)
    assert miss is None
    assert cache.misses == 2
    cache.store([1, 2], np.asarray([[0, 1]], dtype=np.int64), n_local=5)
    cache.store([3, 4], np.asarray([[0, 1]], dtype=np.int64), n_local=5)
    assert cache.size <= 2


def test_clique_one_two_agree_blossom():
    d = 3
    bundle = build_surface_code_bundle(d=d, noise=0.001, rounds=10 * d)
    wms = ensure_window_schedule(bundle, 3 * d, d)
    wm = wms[0]
    g = _get_graph_cache(wm)
    defect = next(iter(g.bound_prev))
    local = clique_mwpm_edges(g, [defect], max_k=6)
    assert local is not None and local.size > 0
    buf = wm.buf
    buf[:] = 0
    buf[defect] = 1
    pairs = wm.matching.decode_to_edges_array(buf)
    arr = np.asarray(pairs, dtype=np.int64).reshape(-1, 2)

    def norm(edges, n):
        s = set()
        for u, v in np.asarray(edges, dtype=np.int64).reshape(-1, 2):
            uu = int(u)
            vv = n if int(v) < 0 or int(v) >= n else int(v)
            if int(v) < 0:
                vv = n
            s.add((min(uu, vv), max(uu, vv)))
        return s

    assert norm(local, wm.n_local) == norm(arr, wm.n_local)

    a = b = None
    for u, nbrs in enumerate(g.adj):
        for v, _w in nbrs:
            if v >= 0:
                a, b = u, v
                break
        if a is not None:
            break
    assert a is not None
    local2 = clique_mwpm_edges(g, [a, b], max_k=6)
    assert local2 is not None
    assert syndrome_cleared_by_edges(wm.n_local, np.array([a, b]), local2)


def test_clique_k3_clears_syndrome():
    d = 3
    bundle = build_surface_code_bundle(d=d, noise=0.001, rounds=10 * d)
    wms = ensure_window_schedule(bundle, 3 * d, d)
    wm = wms[0]
    g = _get_graph_cache(wm)
    candidates = [u for u in g.bound_prev if u in g.bound_dist][:8]
    if len(candidates) < 3:
        pytest.skip("not enough boundary-reachable detectors")
    dets = candidates[:3]
    edges = clique_mwpm_edges(g, dets, max_k=6)
    if edges is None:
        pytest.skip("clique fallthrough")
    assert syndrome_cleared_by_edges(wm.n_local, np.asarray(dets), edges)


def test_cascade_empty_path():
    d, p = 3, 1e-3
    bundle = build_surface_code_bundle(d=d, noise=p, rounds=10 * d)
    syn = np.zeros(bundle.n_detectors, dtype=np.uint8)
    obs = np.zeros(bundle.num_observables, dtype=np.uint8)
    state = CASCADEState()
    out = stream_shot_cascade(
        bundle, syn, config=CASCADEConfig(), state=state, observable_flips=obs
    )
    assert not out.logical_error
    assert state.n_empty == state.n_windows
    assert state.n_escalate == 0


def test_cascade_ler_smoke_d3():
    d, p, shots = 3, 0.001, 40
    bundle = build_surface_code_bundle(d=d, noise=p, rounds=10 * d)
    syn, obs = sample_syndromes_with_observables(bundle, shots=shots, seed=7)
    batch_pred = batch_decode_observables(bundle, syn)
    state = CASCADEState()
    cfg = CASCADEConfig(clique_cap=6, use_cache=True, defer_hard=False)
    errors = 0
    batch_errors = 0
    for s in range(shots):
        out = stream_shot_cascade(
            bundle, syn[s], config=cfg, state=state, shot=s, observable_flips=obs[s]
        )
        if out.logical_error:
            errors += 1
        if np.any(batch_pred[s] != obs[s]):
            batch_errors += 1
    assert errors == batch_errors
    assert state.n_empty + state.n_cache + state.n_clique + state.n_escalate == state.n_windows
