"""Baseline runner for MABS v4 / CASCADE."""

from __future__ import annotations

import time
import numpy as np

from mabs.baselines import BaselineResult, _agg_records


def run_cascade_baseline(
    bundle,
    syndromes: np.ndarray,
    observables: np.ndarray,
    *,
    config=None,
    method_name: str = "mabs_v4",
    warmup: int = 0,
) -> BaselineResult:
    """MABS v4 CASCADE streaming decoder."""
    from mabs.v4.cascade import (
        CASCADEConfig,
        CASCADEState,
        stream_shot_cascade_timed,
        prewarm_cascade_graphs,
    )
    from mabs.v4.exact_pattern_cache import ExactPatternCache

    if config is None:
        config = CASCADEConfig(
            clique_cap=2,  # stage Pareto default; raise for lower escalate
            use_cache=True,
            use_clique=True,
            use_component_clique=False,  # LER-safe
            check_syndrome=True,
            prefer_correctness=True,
            defer_hard=False,
            cache_escalations=True,
            cache_k_max=6,
            prewarm_graphs=True,
        )
    state = CASCADEState(cache=ExactPatternCache(max_size=config.cache_size))
    d = bundle.d
    w = max(1, int(round(config.window_factor * d)))
    C = max(1, int(round(config.commit_factor * d)))
    prewarm_cascade_graphs(bundle, w, C)
    state.graphs_ready = True
    shots = int(syndromes.shape[0])
    warm = max(0, min(int(warmup), max(shots - 1, 0)))
    all_records = []
    errors = 0
    shot_times = []
    preds = []
    for s in range(shots):
        t0 = time.perf_counter_ns()
        out = stream_shot_cascade_timed(
            bundle,
            syndromes[s],
            config=config,
            state=state,
            shot=s,
            observable_flips=observables[s],
        )
        t1 = time.perf_counter_ns()
        preds.append(np.asarray(out.predicted_observables, dtype=np.uint8).ravel())
        if out.logical_error:
            errors += 1
        if s >= warm:
            shot_times.append(t1 - t0)
            all_records.extend(out.records)
    mean_stage, mean_match = _agg_records(all_records)
    pred_arr = np.stack(preds, axis=0) if preds else np.zeros((0, 1), dtype=np.uint8)
    return BaselineResult(
        method=method_name,
        d=bundle.d,
        noise=bundle.noise,
        shots=shots,
        logical_errors=errors,
        ler=errors / shots if shots else float("nan"),
        mean_stage_ns=mean_stage,
        mean_match_ns=mean_match,
        mean_shot_ns=float(np.mean(shot_times)) if shot_times else float("nan"),
        retry_rate=state.escalate_rate,
        extra={
            "escalate_rate": state.escalate_rate,
            "empty_rate": state.empty_rate,
            "cache_rate": state.cache_rate,
            "clique_rate": state.clique_rate,
            "cache_hit_rate": state.cache_hit_rate,
            "n_windows": state.n_windows,
            "n_escalate": state.n_escalate,
            "n_empty": state.n_empty,
            "n_cache": state.n_cache,
            "n_clique": state.n_clique,
            "n_defer_flush": state.n_defer_flush,
            "clique_cap": config.clique_cap,
            "_preds": pred_arr,
            "warmup_shots": warm,
        },
    )
