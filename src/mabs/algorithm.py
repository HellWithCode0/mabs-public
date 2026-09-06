"""Top-level MABS campaign runner: run_mabs(cfg) -> MABSResult."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Optional
import numpy as np

from mabs.adaptive import AdaptiveConfig, AdaptiveState, stream_shot_adaptive_timed
from mabs.confidence import ConfidenceConfig, calibrate_auto_threshold
from mabs.config import MABSConfig
from mabs.reporting import summarize
from mabs.streaming import (
    CircuitBundle,
    ShotDecodeResult,
    WindowRecord,
    build_surface_code_bundle,
    sample_syndromes,
    sample_syndromes_with_observables,
    stream_shot,
)


@dataclass
class MABSResult:
    """Records plus summary statistics from a MABS run."""

    records: list[dict[str, Any]]
    summary: dict[str, Any]
    config: MABSConfig
    bundles: dict[int, CircuitBundle] = field(default_factory=dict, repr=False)
    ler_by_distance: dict[int, float] = field(default_factory=dict)

    @property
    def by_distance(self) -> dict[int, Any]:
        return self.summary.get("by_distance", {})

    @property
    def S_meas(self) -> dict[str, Any]:
        return self.summary.get("S_meas", {})


def run_mabs(cfg: Optional[MABSConfig] = None) -> MABSResult:
    """Run MABS campaigns/shots and return window records + summary stats.

    For each distance d in ``cfg.distances``:
      - build Stim rotated surface-code memory (R=10*d) + DEM + PyMatching
      - sample ``cfg.shots`` syndromes (+ observables if track_ler)
      - stream with fixed w=3d or MABS-Adaptive policy
      - apply the configured post-matching kernel
    """
    if cfg is None:
        cfg = MABSConfig()

    all_records: list[WindowRecord] = []
    bundles: dict[int, CircuitBundle] = {}
    ler_by_distance: dict[int, float] = {}
    K_all: list[int] = []

    adaptive_cfg = AdaptiveConfig(
        w_large_factor=float(cfg.window_factor),
        w_small_factor=float(cfg.w_small_factor),
        commit_factor=float(cfg.commit_stride_factor),
        confidence=ConfidenceConfig(threshold=cfg.confidence_threshold),
        max_retries_per_window=cfg.max_retries_per_window,
    )
    adaptive_state = AdaptiveState(threshold=cfg.confidence_threshold)

    for d in cfg.distances:
        d = int(d)
        rounds = cfg.rounds(d)
        bundle = build_surface_code_bundle(d=d, noise=cfg.noise, rounds=rounds)
        bundles[d] = bundle
        w = cfg.window_depth(d)
        C = cfg.commit_stride(d)
        errors = 0
        n_shots_total = 0

        for camp in range(cfg.campaigns):
            seed = int(cfg.seed + 10007 * d + 97 * camp)
            if cfg.track_ler:
                syndromes, observables = sample_syndromes_with_observables(
                    bundle, shots=cfg.shots, seed=seed
                )
            else:
                syndromes = sample_syndromes(bundle, shots=cfg.shots, seed=seed)
                observables = None

            for s in range(cfg.shots):
                syn = syndromes[s]
                obs = observables[s] if observables is not None else None
                if cfg.adaptive:
                    out = stream_shot_adaptive_timed(
                        bundle,
                        syn,
                        adaptive=adaptive_cfg,
                        state=adaptive_state,
                        kernel_name=cfg.kernel,
                        auto_threshold=cfg.auto_threshold,
                        campaign=camp,
                        shot=s,
                        observable_flips=obs,
                    )
                    recs = out.records
                    if obs is not None and out.logical_error:
                        errors += 1
                else:
                    out2 = stream_shot(
                        bundle,
                        syn,
                        window_depth=w,
                        commit_stride=C,
                        kernel_name=cfg.kernel,
                        auto_threshold=cfg.auto_threshold,
                        campaign=camp,
                        shot=s,
                        observable_flips=obs,
                    )
                    if isinstance(out2, ShotDecodeResult):
                        recs = out2.records
                        if obs is not None and out2.logical_error:
                            errors += 1
                    else:
                        recs = out2
                all_records.extend(recs)
                K_all.extend(r.K for r in recs)
                n_shots_total += 1

        if cfg.track_ler and n_shots_total:
            ler_by_distance[d] = errors / n_shots_total

    # Optional auto-threshold calibration suggestion from K distribution
    suggested_thr = calibrate_auto_threshold(np.asarray(K_all, dtype=np.int64))

    record_dicts = [r.as_dict() for r in all_records]
    summary = summarize(
        record_dicts,
        distances=list(cfg.distances),
        noise=cfg.noise,
        shots=cfg.shots,
        campaigns=cfg.campaigns,
        kernel=cfg.kernel,
        auto_threshold=cfg.auto_threshold,
        rounds_factor=cfg.rounds_factor,
        window_factor=cfg.window_factor,
        commit_stride_factor=cfg.commit_stride_factor,
        seed=cfg.seed,
        fit_loglog_slope=cfg.fit_loglog_slope,
        k_workers=cfg.k_workers,
        tau_c_ns=cfg.tau_c_ns,
    )
    summary["ler_by_distance"] = ler_by_distance
    summary["adaptive"] = {
        "enabled": cfg.adaptive,
        "retry_rate": adaptive_state.retry_rate,
        "n_windows": adaptive_state.n_windows,
        "n_retries": adaptive_state.n_retries,
        "threshold": adaptive_state.threshold,
        "suggested_auto_threshold": suggested_thr,
    }
    # Attach LER into by_distance
    for d, ler in ler_by_distance.items():
        if d in summary.get("by_distance", {}):
            summary["by_distance"][d]["ler"] = ler

    return MABSResult(
        records=record_dicts,
        summary=summary,
        config=cfg,
        bundles=bundles,
        ler_by_distance=ler_by_distance,
    )
