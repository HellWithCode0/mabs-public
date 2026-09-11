"""CLUSTER route: shape test, all-pairs table, dual certificate, end to end."""

from __future__ import annotations

import heapq

import numpy as np
import pytest

from mabs.streaming import (
    batch_decode_observables,
    build_surface_code_bundle,
    ensure_window_schedule,
    sample_syndromes_with_observables,
)
from mabs.streaming_windows import commit_window_edges
from mabs.v3.cluster import DetectorGraph, _get_graph_cache
from mabs.v3.local_decode import blossom_edges_only
from mabs.v3.union_find import fired_induced_components
from mabs.v4.cascade import (
    CASCADEConfig,
    CASCADEState,
    prewarm_cascade_graphs,
    stream_shot_cascade,
)
from mabs.v4.cluster_route import (
    ClusterStats,
    ClusterTable,
    certify,
    get_cluster_table,
    has_numba,
    pair_up,
    route_clusters_buf,
)
from mabs.v4.commit_action import apply_commit_action
from mabs.v4.flash_lut import prewarm_flash_lut


def _line_graph(n, boundary):
    """Path 0-1-...-(n-1), unit weights, boundary edges given as {node: weight}."""
    adj = [[] for _ in range(n)]
    for u in range(n - 1):
        adj[u].append((u + 1, 1.0))
        adj[u + 1].append((u, 1.0))
    for u, w in boundary.items():
        adj[u].append((-1, float(w)))
    g = DetectorGraph(n_local=n, adj=adj, boundary=set(boundary))
    g._compute_boundary_forest()
    return g


def _dijkstra(g, src):
    dist = {src: 0.0}
    pq = [(0.0, src)]
    while pq:
        du, u = heapq.heappop(pq)
        if du != dist.get(u):
            continue
        for v, w in g.adj[u]:
            if v < 0:
                continue
            nd = du + w
            if nd < dist.get(v, np.inf):
                dist[v] = nd
                heapq.heappush(pq, (nd, v))
    return dist


def _windows(d=5, p=1e-3):
    bundle = build_surface_code_bundle(d=d, noise=p, rounds=10 * d)
    return bundle, ensure_window_schedule(bundle, 3 * d, d)


def test_shape_test_matches_union_find():
    """Max fired degree <= 1 must agree with 'every induced component <= 2'."""
    bundle, wms = _windows()
    wm = wms[1]
    tab = get_cluster_table(wm)
    g = _get_graph_cache(wm)
    rng = np.random.default_rng(3)
    n = wm.n_local
    checked = 0
    for k in (3, 4, 6, 10, 20):
        for _ in range(40):
            fired = np.sort(rng.choice(n, size=k, replace=False)).astype(np.int64)
            comps, _ = fired_induced_components(g, fired)
            small = max(len(c) for c in comps) <= 2
            split = pair_up(fired, tab)
            assert (split is not None) == small
            if split is not None:
                got = sorted(
                    [tuple(sorted(x)) for x in split[0].tolist()]
                    + [(int(x),) for x in split[1].tolist()]
                )
                assert got == sorted(tuple(sorted(c)) for c in comps)
            checked += 1
    assert checked == 200


def test_all_pairs_table_matches_dijkstra():
    bundle, wms = _windows(d=3)
    wm = wms[0]
    g = _get_graph_cache(wm)
    tab = get_cluster_table(wm)
    for src in (0, 5, wm.n_local // 2, wm.n_local - 1):
        dist = _dijkstra(g, src)
        for v, dv in dist.items():
            assert tab.D[src, v] == pytest.approx(dv, rel=1e-5, abs=1e-5)


def test_certificate_accepts_separated_pairs():
    g = _line_graph(10, {0: 1.0, 9: 1.0})
    tab = ClusterTable(g)
    pairs = np.array([[2, 3], [7, 8]], dtype=np.int64)
    assert certify(pairs, np.zeros(0, dtype=np.int64), tab)


def test_certificate_rejects_suboptimal_decomposition():
    """Pair {0,1} plus singleton {3}: decomposed cost 1 + 21 = 22, but
    0->boundary (10) plus 1-3 (2) costs 12. The certificate must refuse."""
    g = _line_graph(5, {0: 10.0, 4: 20.0})
    tab = ClusterTable(g)
    pairs = np.array([[0, 1]], dtype=np.int64)
    singles = np.array([3], dtype=np.int64)
    assert not certify(pairs, singles, tab)


def test_certificate_refuses_a_pair_on_its_tie():
    """w_ab == B(a) + B(b): the matcher may break the tie either way."""
    g = _line_graph(4, {0: 0.5, 3: 0.5})
    tab = ClusterTable(g)
    # D(1,2) = 1, B(1) = B(2) = 1.5, so no tie here, but move the boundary in:
    g2 = _line_graph(2, {0: 0.5, 1: 0.5})
    tab2 = ClusterTable(g2)
    assert certify(np.array([[1, 2]], dtype=np.int64), np.zeros(0, dtype=np.int64), tab)
    assert not certify(np.array([[0, 1]], dtype=np.int64), np.zeros(0, dtype=np.int64), tab2)


def _differential(prefer_numba, d=5, p=1e-3, shots=60, seed=91):
    bundle, wms = _windows(d=d, p=p)
    syn, _obs = sample_syndromes_with_observables(bundle, shots=shots, seed=seed)
    luts = [prewarm_flash_lut(w, pair_radius=4) for w in wms]
    tabs = [get_cluster_table(w) for w in wms]
    stats = ClusterStats()
    routed = 0
    for s in range(shots):
        det = syn[s]
        carry = np.zeros(bundle.n_detectors + 1, dtype=np.uint8)
        for wi, wm in enumerate(wms):
            lo, hi, n = wm.det_lo, wm.det_hi, wm.n_local
            buf = wm.buf
            np.bitwise_xor(det[lo:hi], carry[lo:hi], out=buf[:n])
            carry[lo:hi] = 0
            edges, _ = blossom_edges_only(wm, buf)
            if np.count_nonzero(buf[:n]) >= 3:
                act = route_clusters_buf(
                    buf, n, tabs[wi], luts[wi], stats=stats, prefer_numba=prefer_numba
                )
                if act is not None:
                    routed += 1
                    c1 = np.zeros_like(carry)
                    c2 = np.zeros_like(carry)
                    o1, n1 = commit_window_edges(edges, wm, c1, carry_forward=True)
                    o2, n2 = apply_commit_action(act, c2)
                    assert o1 == o2 and n1 == n2 and np.array_equal(c1, c2), (s, wi)
            commit_window_edges(edges, wm, carry, carry_forward=True)
    return routed, stats


def test_routed_windows_commit_exactly_what_blossom_commits_numpy():
    routed, stats = _differential(prefer_numba=False)
    assert routed > 50, stats.as_dict()


@pytest.mark.skipif(not has_numba(), reason="numba not installed")
def test_routed_windows_commit_exactly_what_blossom_commits_numba():
    routed, stats = _differential(prefer_numba=True)
    assert routed > 50, stats.as_dict()


@pytest.mark.skipif(not has_numba(), reason="numba not installed")
def test_numba_and_numpy_paths_route_identically():
    r_nb, s_nb = _differential(prefer_numba=True, seed=17)
    r_np, s_np = _differential(prefer_numba=False, seed=17)
    assert r_nb == r_np
    assert s_nb.as_dict() == s_np.as_dict()


@pytest.mark.parametrize("prefer_numba", [False, True])
@pytest.mark.parametrize("d,p,shots", [(3, 1e-3, 300), (5, 2e-3, 120)])
def test_cluster_route_ler_equals_batch(prefer_numba, d, p, shots):
    if prefer_numba and not has_numba():
        pytest.skip("numba not installed")
    bundle = build_surface_code_bundle(d=d, noise=p, rounds=10 * d)
    syn, obs = sample_syndromes_with_observables(bundle, shots=shots, seed=500 + d)
    ref = batch_decode_observables(bundle, syn)
    cfg = CASCADEConfig(cluster_route=True, prefer_numba=prefer_numba)
    state = CASCADEState()
    prewarm_cascade_graphs(bundle, 3 * d, d, config=cfg)
    state.graphs_ready = True
    disagree = 0
    for s in range(shots):
        out = stream_shot_cascade(bundle, syn[s], config=cfg, state=state, shot=s,
                                  observable_flips=obs[s])
        disagree += int(not np.array_equal(out.predicted_observables, ref[s]))
    assert disagree == 0
    assert state.n_cluster > 0
    assert state.route_counts.get("cluster", 0) == state.n_cluster


def _square_graph():
    """0 at the corner of a unit square 0-1, 0-2, 1-3, 2-3, plus a chord 0-3 of
    weight 2. Boundary edges of weight 1 hang off 1 and 2 only."""
    n = 4
    adj = [[] for _ in range(n)]
    for u, v, w in ((0, 1, 1.0), (0, 2, 1.0), (1, 3, 1.0), (2, 3, 1.0), (0, 3, 2.0)):
        adj[u].append((v, w))
        adj[v].append((u, w))
    adj[1].append((-1, 1.0))
    adj[2].append((-1, 1.0))
    g = DetectorGraph(n_local=n, adj=adj, boundary={1, 2})
    g._compute_boundary_forest()
    return g


def test_degeneracy_guard_fires_on_equal_paths_with_different_effects():
    """Node 0 reaches the boundary through 1 or through 2 at equal weight. If the
    two routes commit different observable parity, the answer is not unique and
    must be refused; the same for the pair (0, 3), joined by three equal paths."""
    from mabs.v4.cluster_route import degenerate_flags

    g = _square_graph()
    tab = ClusterTable(g)

    def effect(a, b):
        key = (min(a, b), max(a, b))
        # Only the boundary edge at node 1 and the edge 1-3 flip the logical.
        obs = 1 if key in ((1, 4), (1, 3)) else 0
        return (obs, frozenset())

    bad_node, bad_edge = degenerate_flags(
        g.n_local, g.adj, tab.indptr, tab.indices, tab.B, tab.D, effect
    )
    assert bad_node[0], "two equal boundary routes with different parity"
    assert not bad_node[1] and not bad_node[2], "single boundary edge, unique"
    e03 = [e for e in range(tab.indptr[0], tab.indptr[1]) if tab.indices[e] == 3]
    assert bad_edge[e03[0]], "0-3 direct vs via 1 differ in parity"

    def same_effect(a, b):
        return (0, frozenset())

    bad_node2, bad_edge2 = degenerate_flags(
        g.n_local, g.adj, tab.indptr, tab.indices, tab.B, tab.D, same_effect
    )
    assert not bad_node2.any() and not bad_edge2.any(), "ties with equal effect are harmless"


def test_real_windows_have_no_degenerate_answers():
    """Measured: zero degenerate singletons and adjacent pairs at d=3,5,7. If this
    ever changes, the route still stays exact because those answers are refused,
    but the coverage numbers in the changelog would no longer hold."""
    bundle, wms = _windows(d=5)
    for wm in wms:
        tab = get_cluster_table(wm)
        assert tab.n_degenerate == (0, 0), tab.n_degenerate


def test_cluster_route_default_follows_numba():
    """Auto by default: on exactly when the compiled route exists."""
    assert CASCADEConfig().cluster_route is None
    assert CASCADEConfig().resolved_cluster_route() is has_numba()
    assert CASCADEConfig(cluster_route=True).resolved_cluster_route() is True
    assert CASCADEConfig(cluster_route=False).resolved_cluster_route() is False


def test_oversized_window_has_no_table():
    bundle, wms = _windows(d=3)
    assert get_cluster_table(wms[0], max_nodes=10) is None
    stats = ClusterStats()
    assert route_clusters_buf(wms[0].buf, wms[0].n_local, None, None, stats=stats) is None
    assert stats.no_table == 1
