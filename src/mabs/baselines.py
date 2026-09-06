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
            for k, v in self.extra.items():
                if k.startswith("_"):
                    continue  # strip internal fields (e.g. _preds) from CSV
                d[k] = v
        return d


def _agg_records(records: list, *, skip_shots: int = 0) -> tuple[float, float]:
    if not records:
        return float("nan"), float("nan")
    if skip_shots > 0:
        # Drop records belonging to the first skip_shots shot indices
        kept = [r for r in records if getattr(r, "shot", 0) >= skip_shots]
        if not kept:
            kept = records
        records = kept
    stage = np.mean([r.tau_stage_ns for r in records])
    match = np.mean([r.tau_match_ns for r in records])
    return float(stage), float(match)


def n_disagree_vs_ref(preds: np.ndarray, ref: np.ndarray) -> int:
    """Count shots where method logical prediction differs from reference (batch)."""
    if preds is None or ref is None:
        return -1
    a = np.asarray(preds)
    b = np.asarray(ref)
    if a.shape != b.shape:
        return -1
    if a.ndim == 1:
        return int(np.sum(a != b))
    return int(np.sum(np.any(a != b, axis=1)))


def disagree_rate(n_disagree: int, shots: int) -> float:
    if n_disagree < 0 or shots <= 0:
        return float("nan")
    return float(n_disagree) / float(shots)


def attach_disagree(result: BaselineResult, ref_preds: Optional[np.ndarray]) -> BaselineResult:
    """Fill n_disagree / disagree_rate from stored _preds vs batch reference."""
    if result.extra is None:
        result.extra = {}
    preds = result.extra.get("_preds")
    if ref_preds is None or preds is None:
        result.extra.setdefault("n_disagree", -1 if result.method != "batch" else 0)
        result.extra.setdefault("disagree_rate", 0.0 if result.method == "batch" else float("nan"))
        return result
    nd = n_disagree_vs_ref(preds, ref_preds)
    result.extra["n_disagree"] = nd
    result.extra["disagree_rate"] = disagree_rate(nd, result.shots)
    return result


def run_batch_baseline(
    bundle: CircuitBundle,
    syndromes: np.ndarray,
    observables: np.ndarray,
    *,
    warmup: int = 0,
) -> BaselineResult:
    """Full-shot batch MWPM."""
    shots = int(syndromes.shape[0])
    w = max(0, min(int(warmup), max(shots - 1, 0)))
    if w > 0:
        _ = batch_decode_observables(bundle, syndromes[:w])
    timed_syn = syndromes[w:]
    n_timed = int(timed_syn.shape[0])
    t0 = time.perf_counter_ns()
    pred_timed = batch_decode_observables(bundle, timed_syn) if n_timed else np.zeros(
        (0, observables.shape[1] if observables.ndim > 1 else 1), dtype=np.uint8
    )
    t1 = time.perf_counter_ns()
    if w > 0:
        pred_full = np.concatenate(
            [batch_decode_observables(bundle, syndromes[:w]), pred_timed], axis=0
        )
    else:
        pred_full = pred_timed
    err = int(np.sum(np.any(pred_full != observables, axis=1))) if shots else 0
    total_ns = t1 - t0
    denom = n_timed if n_timed else 1
    return BaselineResult(
        method="batch",
        d=bundle.d,
        noise=bundle.noise,
        shots=shots,
        logical_errors=err,
        ler=err / shots if shots else float("nan"),
        mean_stage_ns=float(total_ns / denom),
        mean_match_ns=float(total_ns / denom),
        mean_shot_ns=float(total_ns / denom),
        retry_rate=0.0,
        extra={
            "n_disagree": 0,
            "disagree_rate": 0.0,
            "_preds": pred_full,
            "warmup_shots": w,
        },
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
    warmup: int = 0,
) -> BaselineResult:
    """Fixed-window streaming MWPM (single blossom per window after fair-baseline fix)."""
    d = bundle.d
    w = max(1, int(round(window_factor * d)))
    C = max(1, int(round(commit_factor * d)))
    name = method_name or f"stream_w{window_factor:g}d"
    shots = int(syndromes.shape[0])
    warm = max(0, min(int(warmup), max(shots - 1, 0)))
    all_records = []
    errors = 0
    shot_times = []
    preds = []
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
        preds.append(np.asarray(out.predicted_observables, dtype=np.uint8).ravel())
        if out.logical_error:
            errors += 1
        if s >= warm:
            shot_times.append(t1 - t0)
            all_records.extend(out.records)
    mean_stage, mean_match = _agg_records(all_records)
    pred_arr = np.stack(preds, axis=0) if preds else np.zeros((0, 1), dtype=np.uint8)
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
        extra={
            "window_depth": w,
            "commit_stride": C,
            "_preds": pred_arr,
            "warmup_shots": warm,
        },
    )


def run_adaptive_baseline(
    bundle: CircuitBundle,
    syndromes: np.ndarray,
    observables: np.ndarray,
    *,
    adaptive: Optional[AdaptiveConfig] = None,
    kernel_name: str = "auto",
    auto_threshold: int = 64,
    warmup: int = 0,
) -> BaselineResult:
    """MABS-Adaptive streaming."""
    if adaptive is None:
        adaptive = AdaptiveConfig()
    state = AdaptiveState(threshold=adaptive.confidence.threshold)
    shots = int(syndromes.shape[0])
    warm = max(0, min(int(warmup), max(shots - 1, 0)))
    all_records = []
    errors = 0
    shot_times = []
    preds = []
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
        preds.append(np.asarray(out.predicted_observables, dtype=np.uint8).ravel())
        if out.logical_error:
            errors += 1
        if s >= warm:
            shot_times.append(t1 - t0)
            all_records.extend(out.records)
    mean_stage, mean_match = _agg_records(all_records)
    pred_arr = np.stack(preds, axis=0) if preds else np.zeros((0, 1), dtype=np.uint8)
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
            "_preds": pred_arr,
            "warmup_shots": warm,
        },
    )


def prepare_samples(
    bundle: CircuitBundle,
    shots: int,
    seed: int,
) -> tuple[np.ndarray, np.ndarray]:
    return sample_syndromes_with_observables(bundle, shots=shots, seed=seed)


from mabs.v3.baseline_runner import run_slem_baseline  # noqa: E402,F401
