"""Decode baselines for competitive comparison.

- Batch PyMatching (full shot)
- Fixed streaming w=3d, C=d
- Fixed streaming smaller window (w=2d, C=d)
- MABS-Adaptive (see ``mabs.adaptive``)
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Optional
import time
import numpy as np

from mabs.adaptive import (
    AdaptiveConfig,
    AdaptiveState,
    stream_shot_adaptive_timed,
)
from mabs.streaming import (
    CircuitBundle,
    ShotDecodeResult,
    batch_decode_observables,
    sample_syndromes_with_observables,
    stream_shot,
)


@dataclass
class BaselineResult:
    """Aggregate LER + timing for one (method, d, p) cell."""

    method: str
    d: int
    noise: float
    shots: int
    logical_errors: int
    ler: float
    mean_stage_ns: float
    mean_match_ns: float
    mean_shot_ns: float
    retry_rate: float = 0.0
    extra: dict[str, Any] | None = None

    def as_dict(self) -> dict[str, Any]:
        d = {
            "method": self.method,
            "d": self.d,
            "noise": self.noise,
            "shots": self.shots,
            "logical_errors": self.logical_errors,
            "ler": self.ler,
            "mean_stage_ns": self.mean_stage_ns,
            "mean_match_ns": self.mean_match_ns,
            "mean_shot_ns": self.mean_shot_ns,
            "retry_rate": self.retry_rate,
        }
        if self.extra:
            d.update(self.extra)
        return d


def _agg_records(records: list) -> tuple[float, float]:
    if not records:
        return float("nan"), float("nan")
    stage = np.mean([r.tau_stage_ns for r in records])
    match = np.mean([r.tau_match_ns for r in records])
    return float(stage), float(match)


def run_batch_baseline(
    bundle: CircuitBundle,
    syndromes: np.ndarray,
    observables: np.ndarray,
) -> BaselineResult:
    """Full-shot batch MWPM."""
    t0 = time.perf_counter_ns()
    pred = batch_decode_observables(bundle, syndromes)
    t1 = time.perf_counter_ns()
    shots = int(syndromes.shape[0])
    err = int(np.sum(np.any(pred != observables, axis=1)))
    total_ns = t1 - t0
    return BaselineResult(
        method="batch",
        d=bundle.d,
        noise=bundle.noise,
        shots=shots,
        logical_errors=err,
        ler=err / shots if shots else float("nan"),
        mean_stage_ns=float(total_ns / shots) if shots else float("nan"),
        mean_match_ns=float(total_ns / shots) if shots else float("nan"),
        mean_shot_ns=float(total_ns / shots) if shots else float("nan"),
        retry_rate=0.0,
    )


def run_fixed_streaming_baseline(
    bundle: CircuitBundle,
    syndromes: np.ndarray,
    observables: np.ndarray,
    *,
    window_factor: float,
    commit_factor: float = 1.0,
    method_name: Optional[str] = None,
    kernel_name: str = "auto",
    auto_threshold: int = 64,
) -> BaselineResult:
    """Fixed-window streaming MWPM."""
    d = bundle.d
    w = max(1, int(round(window_factor * d)))
    C = max(1, int(round(commit_factor * d)))
    name = method_name or f"stream_w{window_factor:g}d"
    shots = int(syndromes.shape[0])
    all_records = []
    errors = 0
    shot_times = []
    for s in range(shots):
        t0 = time.perf_counter_ns()
        out = stream_shot(
            bundle,
            syndromes[s],
            window_depth=w,
            commit_stride=C,
            kernel_name=kernel_name,
            auto_threshold=auto_threshold,
            shot=s,
            observable_flips=observables[s],
        )
        assert isinstance(out, ShotDecodeResult)
        t1 = time.perf_counter_ns()
        shot_times.append(t1 - t0)
        all_records.extend(out.records)
        if out.logical_error:
            errors += 1
    mean_stage, mean_match = _agg_records(all_records)
    return BaselineResult(
        method=name,
        d=d,
        noise=bundle.noise,
        shots=shots,
        logical_errors=errors,
        ler=errors / shots if shots else float("nan"),
        mean_stage_ns=mean_stage,
        mean_match_ns=mean_match,
        mean_shot_ns=float(np.mean(shot_times)) if shot_times else float("nan"),
        retry_rate=0.0,
        extra={"window_depth": w, "commit_stride": C},
    )


def run_adaptive_baseline(
    bundle: CircuitBundle,
    syndromes: np.ndarray,
    observables: np.ndarray,
    *,
    adaptive: Optional[AdaptiveConfig] = None,
    kernel_name: str = "auto",
    auto_threshold: int = 64,
) -> BaselineResult:
    """MABS-Adaptive streaming."""
    if adaptive is None:
        adaptive = AdaptiveConfig()
    state = AdaptiveState(threshold=adaptive.confidence.threshold)
    shots = int(syndromes.shape[0])
    all_records = []
    errors = 0
    shot_times = []
    for s in range(shots):
        t0 = time.perf_counter_ns()
        out = stream_shot_adaptive_timed(
            bundle,
            syndromes[s],
            adaptive=adaptive,
            state=state,
            kernel_name=kernel_name,
            auto_threshold=auto_threshold,
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
        method="mabs_adaptive",
        d=bundle.d,
        noise=bundle.noise,
        shots=shots,
        logical_errors=errors,
        ler=errors / shots if shots else float("nan"),
        mean_stage_ns=mean_stage,
        mean_match_ns=mean_match,
        mean_shot_ns=float(np.mean(shot_times)) if shot_times else float("nan"),
        retry_rate=state.retry_rate,
        extra={
            "final_threshold": state.threshold,
            "n_windows": state.n_windows,
            "n_retries": state.n_retries,
        },
    )


def prepare_samples(
    bundle: CircuitBundle,
    shots: int,
    seed: int,
) -> tuple[np.ndarray, np.ndarray]:
    return sample_syndromes_with_observables(bundle, shots=shots, seed=seed)
