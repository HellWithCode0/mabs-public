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
    cfg = CASCADEConfig(clique_cap=6, use_cache=True, use_peel=True)
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
    routed = state.n_empty + state.n_cache + state.n_pair + state.n_clique + state.n_peel + state.n_escalate
    assert routed == state.n_windows
