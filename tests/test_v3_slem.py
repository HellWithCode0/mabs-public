"""Tests for MABS v3.1 Sparse Local Escalation Matching."""

from __future__ import annotations

import numpy as np
import pytest

from mabs.streaming import (
    batch_decode_observables,
    build_surface_code_bundle,
    ensure_window_schedule,
    sample_syndromes_with_observables,
)
from mabs.v3.cluster import cluster_syndrome, build_detector_graph, _get_graph_cache
from mabs.v3.escalate import EscalationPolicy, should_escalate
from mabs.v3.local_decode import (
    decode_one_defect,
    decode_two_defects,
    local_decode_clusters,
    local_decode_induced,
)
from mabs.v3.slem import SLEMConfig, SLEMState, stream_shot_slem
from mabs.v3.union_find import fired_induced_components


def test_empty_path_noop():
    d, p = 3, 1e-3
    bundle = build_surface_code_bundle(d=d, noise=p, rounds=10 * d)
    syn = np.zeros(bundle.n_detectors, dtype=np.uint8)
    obs = np.zeros(bundle.num_observables, dtype=np.uint8)
    state = SLEMState()
    out = stream_shot_slem(
        bundle, syn, config=SLEMConfig(), state=state, observable_flips=obs
    )
    assert not out.logical_error
    assert state.n_empty == state.n_windows
    assert state.n_escalate == 0
    assert all(r.K == 0 for r in out.records)


def test_clustering_connects_neighbors():
    d = 3
    bundle = build_surface_code_bundle(d=d, noise=0.001, rounds=10 * d)
    wms = ensure_window_schedule(bundle, 3 * d, d)
    wm = wms[0]
    g = build_detector_graph(wm.matching, wm.n_local)
    a = b = None
    for u, nbrs in enumerate(g.adj):
        for v, _w in nbrs:
            if v >= 0:
                a, b = u, v
                break
        if a is not None:
            break
    assert a is not None
    syn = np.zeros(wm.n_local, dtype=np.uint8)
    syn[a] = 1
    syn[b] = 1
    clusters = cluster_syndrome(syn, g, radius=1)
    assert len(clusters) == 1
    assert clusters[0].size == 2


def test_local_one_defect_agrees_with_blossom():
    d = 3
    bundle = build_surface_code_bundle(d=d, noise=0.001, rounds=10 * d)
    wms = ensure_window_schedule(bundle, 3 * d, d)
    wm = wms[0]
    g = build_detector_graph(wm.matching, wm.n_local)
    defect = next(iter(g.bound_prev))
    local = decode_one_defect(g, defect)
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


def test_local_two_defect_exactness():
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
    local = decode_two_defects(g, a, b)
    assert local is not None and local.size > 0
    buf = wm.buf
    buf[:] = 0
    buf[a] = 1
    buf[b] = 1
    pairs = wm.matching.decode_to_edges_array(buf)
    arr = np.asarray(pairs, dtype=np.int64).reshape(-1, 2)

    def weight(edges, graph):
        syn = np.zeros(graph.n_local, dtype=np.uint8)
        syn[a] = 1
        syn[b] = 1
        for u, v in np.asarray(edges, dtype=np.int64).reshape(-1, 2):
            uu, vv = int(u), int(v)
            if 0 <= uu < graph.n_local:
                syn[uu] ^= 1
            if 0 <= vv < graph.n_local:
                syn[vv] ^= 1
        return not syn.any()

    assert weight(local, g)
    assert weight(arr, g)


def test_induced_components_adjacent():
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
    fired = np.asarray([a, b], dtype=np.int64)
    comps, ew = fired_induced_components(g, fired)
    assert len(comps) == 1
    assert len(comps[0]) == 2
    assert ((a, b) if a < b else (b, a)) in ew


def test_slem_ler_matches_batch_d3():
    d, p, shots = 3, 1e-3, 150
    bundle = build_surface_code_bundle(d=d, noise=p, rounds=10 * d)
    dets, obs = sample_syndromes_with_observables(bundle, shots=shots, seed=303)
    batch = batch_decode_observables(bundle, dets)
    state = SLEMState()
    cfg = SLEMConfig()
    agree = 0
    slem_err = batch_err = 0
    for i in range(shots):
        out = stream_shot_slem(
            bundle, dets[i], config=cfg, state=state, observable_flips=obs[i]
        )
        if np.array_equal(out.predicted_observables, batch[i]):
            agree += 1
        if out.logical_error:
            slem_err += 1
        if np.any(batch[i] != obs[i]):
            batch_err += 1
    assert agree / shots >= 0.95, f"agreement {agree}/{shots}"
    assert slem_err == batch_err


def test_slem_ler_matches_batch_d5_smoke():
    d, p, shots = 5, 1e-3, 60
    bundle = build_surface_code_bundle(d=d, noise=p, rounds=10 * d)
    dets, obs = sample_syndromes_with_observables(bundle, shots=shots, seed=404)
    batch = batch_decode_observables(bundle, dets)
    batch_err = int(np.sum(np.any(batch != obs, axis=1)))
    state = SLEMState()
    slem_err = 0
    for i in range(shots):
        out = stream_shot_slem(
            bundle, dets[i], config=SLEMConfig(), state=state, observable_flips=obs[i]
        )
        if out.logical_error:
            slem_err += 1
    assert slem_err <= batch_err + 2
    assert state.n_windows > 0


def test_escalate_rate_smoke_d5():
    """Default (cap=2) improves on v3.0; cluster-local mode reaches ≪1 escalate."""
    d, p, shots = 5, 1e-3, 40
    bundle = build_surface_code_bundle(d=d, noise=p, rounds=10 * d)
    dets, obs = sample_syndromes_with_observables(bundle, shots=shots, seed=505)

    state = SLEMState()
    cfg = SLEMConfig()
    for i in range(shots):
        stream_shot_slem(
            bundle, dets[i], config=cfg, state=state, observable_flips=obs[i]
        )
    assert state.n_windows > 0
    assert state.escalate_rate <= 0.85, f"default escalate_rate={state.escalate_rate}"
    assert state.local_rate > 0.05

    from mabs.v3.escalate import EscalationPolicy
    state2 = SLEMState()
    cfg2 = SLEMConfig(
        use_cluster_local=True,
        policy=EscalationPolicy(
            max_local_cluster=2, max_local_defects=12, tune=False
        ),
    )
    for i in range(shots):
        stream_shot_slem(
            bundle, dets[i], config=cfg2, state=state2, observable_flips=obs[i]
        )
    assert state2.escalate_rate < 0.35, f"cluster escalate_rate={state2.escalate_rate}"


def test_escalation_policy_empty_and_large():
    pol = EscalationPolicy(max_local_defects=4, max_local_cluster=2)
    d0 = should_escalate(n_defects=0, clusters=[], syndrome_density=0.0, policy=pol)
    assert not d0.escalate and d0.reason == "empty"
    from mabs.v3.cluster import Cluster

    big = [Cluster(defects=[0, 1, 2, 3, 4], nodes=[0, 1, 2, 3, 4])]
    d1 = should_escalate(n_defects=5, clusters=big, syndrome_density=0.01, policy=pol)
    assert d1.escalate
