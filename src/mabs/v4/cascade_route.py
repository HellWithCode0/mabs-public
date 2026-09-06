"""CASCADE per-window route decisions (v4.1)."""
from __future__ import annotations
from typing import Any, List, Optional, Tuple
import numpy as np
from mabs.v3.local_decode import blossom_edges_only, syndrome_cleared_by_edges
from mabs.v4.clique_mwpm import clique_mwpm_edges
from mabs.v4.commit_action import CommitAction, edges_to_commit_action
from mabs.v4.exact_pattern_cache import ExactPatternCache
from mabs.v4.cascade_core import (
    CASCADEConfig, CASCADEState,
    _COST_CLIQUE_BASE, _COST_CLIQUE_K, _COST_PAIR,
    _estimate_local_cost, _peel_easy_residual, _try_clique, _try_pair_lut,
)

def route_window(
    wm, buf, fired, n_defects, *, config: CASCADEConfig, state: CASCADEState, topo, gate_max: int,
) -> Tuple[str, np.ndarray, Optional[CommitAction], bool, float]:
    """Return (path, edges, action, escalated, weight)."""
    edges = np.zeros((0, 2), dtype=np.int64)
    action: Optional[CommitAction] = None
    weight = 0.0
    escalated = False
    n = wm.n_local
    if n_defects == 0:
        state.n_empty += 1
        state._bump_route("empty")
        return "empty", edges, CommitAction.empty(), False, 0.0
    if n_defects > gate_max:
        edges, weight = blossom_edges_only(wm, buf)
        action = edges_to_commit_action(edges, wm)
        state.n_escalate += 1
        state.n_gate_escalate += 1
        state._bump_route("gate_escalate")
        return "gate_escalate", edges, action, True, weight
    handled = False
    path = "escalate"
    est = _estimate_local_cost(n_defects, clique_cap=config.clique_cap, use_peel=config.use_peel)
    if config.cost_aware and est >= config.blossom_cost_budget and n_defects > config.clique_cap:
        edges, weight = blossom_edges_only(wm, buf)
        action = edges_to_commit_action(edges, wm)
        state.n_escalate += 1
        state.n_cost_escalate += 1
        state._bump_route("cost_escalate")
        handled = True
        path = "cost_escalate"
        escalated = True
    if not handled and config.use_cache and n_defects <= config.cache_k_max:
        cached = state.cache.lookup_action(topo, fired.tolist(), wm=wm)
        if cached is not None:
            action = cached
            state.n_cache += 1
            state._bump_route("cache")
            handled = True
            path = "cache"
    if not handled and config.use_pair_lut and n_defects <= 2:
        if (not config.cost_aware) or (_COST_PAIR < config.blossom_cost_budget):
            cl = _try_pair_lut(wm, fired, check=config.check_syndrome)
            if cl is not None:
                edges = cl
                action = edges_to_commit_action(edges, wm)
                weight = float(edges.shape[0])
                state.n_pair += 1
                state._bump_route("pair")
                handled = True
                path = "pair"
    if (not handled and config.use_clique and n_defects <= config.clique_cap and n_defects > 2):
        cl_cost = _COST_CLIQUE_BASE + _COST_CLIQUE_K * (n_defects ** 2)
        if (not config.cost_aware) or (cl_cost < config.blossom_cost_budget):
            cl = _try_clique(wm, fired, clique_cap=config.clique_cap, check=config.check_syndrome)
            if cl is not None:
                edges = cl
                action = edges_to_commit_action(edges, wm)
                weight = float(edges.shape[0])
                state.n_clique += 1
                state._bump_route("clique")
                handled = True
                path = "clique"
    if not handled and config.use_peel and n_defects > config.clique_cap:
        easy, residual, tag = _peel_easy_residual(
            wm, buf, fired, peel_cap=config.peel_cap, check=config.check_syndrome,
            prefer_correctness=config.prefer_correctness,
        )
        if tag == "peel" and easy is not None:
            edges = easy
            action = edges_to_commit_action(edges, wm)
            weight = float(edges.shape[0])
            state.n_peel += 1
            state._bump_route("peel")
            handled = True
            path = "peel"
        elif tag == "residual":
            state.n_residual += 1
            if config.residual_only_blossom and residual:
                buf2 = buf.copy()
                if easy is not None and easy.size:
                    for u, v in np.asarray(easy, dtype=np.int64).reshape(-1, 2):
                        uu, vv = int(u), int(v)
                        if 0 <= uu < n: buf2[uu] ^= 1
                        if 0 <= vv < n: buf2[vv] ^= 1
                edges_r, _ = blossom_edges_only(wm, buf2)
                if easy is not None and easy.size and edges_r.size:
                    edges = np.vstack([easy, edges_r])
                elif easy is not None and easy.size:
                    edges = easy
                else:
                    edges = edges_r
                if config.prefer_correctness and not syndrome_cleared_by_edges(n, fired, edges):
                    edges, weight = blossom_edges_only(wm, buf)
                else:
                    weight = float(edges.shape[0])
            else:
                edges, weight = blossom_edges_only(wm, buf)
            action = edges_to_commit_action(edges, wm)
            escalated = True
            path = "residual_escalate"
            state.n_escalate += 1
            state._bump_route("residual_escalate")
            handled = True
    if not handled:
        edges, weight = blossom_edges_only(wm, buf)
        action = edges_to_commit_action(edges, wm)
        escalated = True
        path = "escalate"
        state.n_escalate += 1
        state._bump_route("escalate")
    if (
        config.use_cache and config.cache_escalations and action is not None
        and n_defects <= config.cache_k_max and path not in ("empty", "cache")
    ):
        state.cache.store_from_edges(
            topo, fired.tolist(), edges if edges is not None else np.zeros((0, 2), dtype=np.int64), wm
        )
    return path, edges, action, escalated, weight
