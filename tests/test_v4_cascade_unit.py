"""Tests for MABS v4.1 CASCADE priority list."""

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
from mabs.v3.local_decode import syndrome_cleared_by_edges, decode_two_defects
from mabs.v4.exact_pattern_cache import (
    ExactPatternCache,
    canonicalize_defects,
    relative_key,
    make_topo_id,
    topo_id_from_wm,
    build_pattern_key,
)
from mabs.v4.commit_action import edges_to_commit_action, apply_commit_action, CommitAction
from mabs.v4.clique_mwpm import clique_mwpm_edges
from mabs.v4.pair_lut import PairPathLUT, get_pair_lut, decode_pair_cached
from mabs.v4.cascade import CASCADEConfig, CASCADEState, stream_shot_cascade


def test_canonicalize_relative():
    key, origin = canonicalize_defects([10, 12, 15])
    assert origin == 10
    assert key == (10, 12, 15)
    assert relative_key([10, 12, 15]) == (0, 2, 5)
    assert relative_key([100, 102, 105]) == (0, 2, 5)


def test_topology_safe_keys_no_collision():
    """Different window configs must not share cache keys (priority 4)."""
    t1 = make_topo_id(d=5, window_depth=15, commit_stride=5, n_local=100, det_lo=0, graph_token=1)
    t2 = make_topo_id(d=5, window_depth=10, commit_stride=5, n_local=100, det_lo=0, graph_token=1)
    t3 = make_topo_id(d=7, window_depth=15, commit_stride=5, n_local=100, det_lo=0, graph_token=1)
    t4 = make_topo_id(d=5, window_depth=15, commit_stride=5, n_local=100, det_lo=50, graph_token=1)
    t5 = make_topo_id(d=5, window_depth=15, commit_stride=5, n_local=100, det_lo=0, graph_token=2)
    defects = (1, 2, 3)
    keys = {build_pattern_key(t, defects) for t in (t1, t2, t3, t4, t5)}
    assert len(keys) == 5


def test_exact_pattern_cache_commit_action_and_freq():
    """Cache stores CommitActions; frequency admission (priority 1, 6)."""
    d = 3
    bundle = build_surface_code_bundle(d=d, noise=0.001, rounds=10 * d)
    wms = ensure_window_schedule(bundle, 3 * d, d)
    wm = wms[0]
    topo = topo_id_from_wm(wm, d=d)
    cache = ExactPatternCache(max_size=8, admit_after=2)
    edges = np.asarray([[0, wm.n_local]], dtype=np.int64)
    # manufacture a commit action
    action = edges_to_commit_action(edges, wm)

    # first miss
    assert cache.lookup_action(topo, [0], wm=wm) is None
    assert cache.misses == 1
    assert not cache.offer_store_action(topo, [0], action, edges=edges, force=False)
    assert cache.size == 0
    # second miss → admit
    assert cache.lookup_action(topo, [0], wm=wm) is None
    assert cache.offer_store_action(topo, [0], action, edges=edges, force=False)
    hit = cache.lookup_action(topo, [0], wm=wm)
    assert hit is not None
    assert isinstance(hit, CommitAction)
    assert cache.hits >= 1


def test_relative_canonicalization_within_topo():
    """Opt-in relative keys collide only under same topo (priority 7)."""
    d = 3
    bundle = build_surface_code_bundle(d=d, noise=0.001, rounds=10 * d)
    wms = ensure_window_schedule(bundle, 3 * d, d)
    wm = wms[0]
    topo = topo_id_from_wm(wm, d=d)
    cache = ExactPatternCache(max_size=16, admit_after=1, canonicalize_relative=True)
    # Use two translated patterns with same relative shape if graph allows
    # Store with force using synthetic local edges
    e1 = np.asarray([[5, 6]], dtype=np.int64)
    a1 = edges_to_commit_action(e1, wm)
    assert cache.store_from_edges(topo, [5, 6], e1, wm, force=True)
    # Same relative offsets under same topo should hit
    hit = cache.lookup_action(topo, [15, 16], wm=wm)
    # May compile action; at least key must hit
    assert hit is not None
    # Different topo must miss
    topo2 = make_topo_id(
        d=d, window_depth=99, commit_stride=d, n_local=wm.n_local, det_lo=wm.det_lo, graph_token=id(wm.matching)
    )
    assert cache.lookup_action(topo2, [15, 16], wm=wm) is None


def test_pair_lut_lazy_reuse():
    """K=2 PairPathLUT populates once and reuses (priority 2)."""
    d = 3
    bundle = build_surface_code_bundle(d=d, noise=0.001, rounds=10 * d)
    wms = ensure_window_schedule(bundle, 3 * d, d)
    wm = wms[0]
    g = _get_graph_cache(wm)
    a = b = None
    for u, nbrs in enumerate(g.adj):
        for v, _w in nbrs:
            if v >= 0:
                a, b = u, v
                break
        if a is not None:
            break
    assert a is not None
    lut = get_pair_lut(g)
    e1 = lut.get_or_compute(a, b)
    assert e1 is not None
    assert lut.misses == 1 and lut.fills == 1
    e2 = lut.get_or_compute(a, b)
    assert e2 is not None
    assert lut.hits == 1
    assert np.array_equal(e1, e2)
    # agrees with Dijkstra path
    ref = decode_two_defects(g, a, b)
    assert ref is not None
    assert syndrome_cleared_by_edges(wm.n_local, np.array([a, b]), e2)


def test_defect_gate_escalates():
    """n_defects > gate_max → gate_escalate (priority 3)."""
    d, p = 3, 1e-3
    bundle = build_surface_code_bundle(d=d, noise=p, rounds=10 * d)
    syn = np.zeros(bundle.n_detectors, dtype=np.uint8)
    # fire many defects in first window
    wms = ensure_window_schedule(bundle, 3 * d, d)
    wm = wms[0]
    n_fire = min(20, wm.n_local)
    syn[wm.det_lo : wm.det_lo + n_fire] = 1
    obs = np.zeros(bundle.num_observables, dtype=np.uint8)
    state = CASCADEState()
    cfg = CASCADEConfig(gate_max=4, use_peel=False, use_cache=False, clique_cap=2)
    stream_shot_cascade(bundle, syn, config=cfg, state=state, observable_flips=obs)
    assert state.n_gate_escalate >= 1


def test_apply_commit_action_matches_edges():
    d = 3
    bundle = build_surface_code_bundle(d=d, noise=0.001, rounds=10 * d)
    wms = ensure_window_schedule(bundle, 3 * d, d)
    wm = wms[0]
    from mabs.streaming_windows import commit_window_edges

    g = _get_graph_cache(wm)
    defect = next(iter(g.bound_prev))
    edges = clique_mwpm_edges(g, [defect], max_k=2)
    assert edges is not None
    carry1 = np.zeros(bundle.n_detectors + 1, dtype=np.uint8)
    carry2 = np.zeros_like(carry1)
    obs1, n1 = commit_window_edges(edges, wm, carry1, carry_forward=True)
    action = edges_to_commit_action(edges, wm)
    obs2, n2 = apply_commit_action(action, carry2)
    assert obs1 == obs2
    assert n1 == n2
    assert np.array_equal(carry1, carry2)


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


def test_no_defer_hard_attribute_footgun():
    """defer_hard removed / unused — hard windows escalate immediately (priority 10)."""
    cfg = CASCADEConfig()
    assert not hasattr(cfg, "defer_hard") or getattr(cfg, "defer_hard", False) is False
