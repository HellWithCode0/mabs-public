"""Tests for naive / loop / vector / auto commit kernels."""

from __future__ import annotations

import numpy as np
import pytest

from mabs.kernels import get_kernel, kernel_auto, kernel_loop, kernel_naive, kernel_vector
from mabs.queue_replay import batch_K_and_empty, replay_commits


def _edges(*pairs):
    if not pairs:
        return np.zeros((0, 2), dtype=np.int64)
    return np.asarray(pairs, dtype=np.int64)


@pytest.mark.parametrize("fn", [kernel_naive, kernel_loop, kernel_vector])
def test_empty_K_early_path(fn):
    active = np.array([0, 1, 2, 10, 11], dtype=np.int64)
    r = fn(_edges(), active, commit_lo=0, commit_hi=3)
    assert r.K == 0
    assert r.empty_output is True
    assert r.committed_edges.shape == (0, 2)
    # Carry is detectors >= commit_hi
    assert np.all(r.carry_detectors >= 3)


@pytest.mark.parametrize("fn", [kernel_naive, kernel_loop, kernel_vector])
def test_commit_earliest_in_region(fn):
    edges = _edges((1, 5), (4, 8), (0, 2), (-1, 7))
    active = np.arange(10, dtype=np.int64)
    r = fn(edges, active, commit_lo=0, commit_hi=3)
    assert r.K == 4
    # (1,5) earliest=1 in [0,3); (0,2) earliest=0 in; (-1,7) earliest=7 not in
    # (4,8) earliest=4 not in
    committed = {tuple(sorted(map(int, row))) for row in r.committed_edges}
    assert (1, 5) in committed or (5, 1) in committed
    assert (0, 2) in committed or (2, 0) in committed
    assert len(r.committed_edges) == 2


def test_kernels_agree_on_commit_set():
    rng = np.random.default_rng(0)
    edges = rng.integers(0, 20, size=(30, 2))
    # sprinkle some boundary endpoints
    edges[0, 0] = -1
    edges[5, 1] = -1
    active = np.arange(25, dtype=np.int64)
    lo, hi = 2, 8
    results = [fn(edges, active, lo, hi) for fn in (kernel_naive, kernel_loop, kernel_vector)]
    sets = []
    for r in results:
        s = {tuple(sorted(map(int, row))) for row in r.committed_edges}
        sets.append(s)
    assert sets[0] == sets[1] == sets[2]


def test_auto_threshold_dispatch():
    active = np.arange(10, dtype=np.int64)
    small = _edges((1, 2), (3, 4))
    r_small = kernel_auto(small, active, 0, 5, threshold=64)
    assert r_small.K == 2
    big = np.arange(80 * 2, dtype=np.int64).reshape(80, 2) % 20
    r_big = kernel_auto(big, active, 0, 5, threshold=64)
    assert r_big.K == 80
    # get_kernel wiring
    k = get_kernel("auto", auto_threshold=1)
    r = k(small, active, 0, 5)
    assert r.K == 2


def test_replay_batch():
    edges = [_edges(), _edges((1, 2), (5, 6))]
    actives = [np.arange(10), np.arange(10)]
    regions = [(0, 3), (0, 3)]
    results = replay_commits(edges, actives, regions, kernel="vector")
    K, empty = batch_K_and_empty(results)
    assert list(K) == [0, 2]
    assert list(empty) == [True, False]
