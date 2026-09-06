"""Baseline runner for MABS v3 / SLEM."""

from __future__ import annotations

from typing import Any, Optional
import time
import numpy as np

from mabs.baselines import BaselineResult, _agg_records


def run_slem_baseline(
    bundle,
    syndromes: np.ndarray,
    observables: np.ndarray,
    *,
    config=None,
    method_name: str = "mabs_v3",
) -> BaselineResult:
    """MABS v3 Sparse Local Escalation Matching (SLEM)."""
    from mabs.v3.slem import SLEMConfig, SLEMState, stream_shot_slem_timed, prewarm_slem_graphs
    from mabs.v3.escalate import EscalationPolicy

    if config is None:
        config = SLEMConfig(
            local_defect_cap=1,
            policy=EscalationPolicy(tune=False),
            prewarm_graphs=True,
        )
    state = SLEMState()
    state.policy = EscalationPolicy(
        max_local_cluster=config.policy.max_local_cluster,
        max_local_defects=config.policy.max_local_defects,
        max_local_density=config.policy.max_local_density,
        cluster_radius=config.policy.cluster_radius,
        target_escalate_rate=config.policy.target_escalate_rate,
        tune=config.policy.tune,
        tune_every=config.policy.tune_every,
        tune_step=config.policy.tune_step,
    )
    d = bundle.d
    w = max(1, int(round(config.window_factor * d)))
    C = max(1, int(round(config.commit_factor * d)))
    prewarm_slem_graphs(bundle, w, C)
    state.graphs_ready = True
    shots = int(syndromes.shape[0])
    all_records = []
    errors = 0
    shot_times = []
    for s in range(shots):
        t0 = time.perf_counter_ns()
        out = stream_shot_slem_timed(
            bundle,
            syndromes[s],
            config=config,
            state=state,
            shot=s,
            observable_flips=observables[s],
        )
        t1 = time.perf_counter_ns()
        shot_times.append(t1 - t0)
        all_records.extend(out.records)
        if out.logical_error:
            errors += 1
    mean_stage, mean_match = _agg_records(all_records)
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
            "local_rate": state.local_rate,
            "n_windows": state.n_windows,
            "n_escalate": state.n_escalate,
            "n_empty": state.n_empty,
            "n_local": state.n_local,
            "max_local_defects": state.policy.max_local_defects,
            "max_local_cluster": state.policy.max_local_cluster,
        },
    )
