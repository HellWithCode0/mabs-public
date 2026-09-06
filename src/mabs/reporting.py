"""Boundary ratios, offered load, and S_meas reporting metadata."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Mapping, Optional, Sequence
import math
import numpy as np

from mabs.mixture import MixtureStats, compute_mixture


@dataclass
class BoundaryRatios:
    """R_q(d) = q(τ_stage) / q(τ_match) for mean / median / q95 / q99."""

    d: int
    R_mean: float
    R_median: float
    R_q95: float
    R_q99: float
    loglog_slope: Optional[float] = None

    def as_dict(self) -> dict:
        return {
            "d": self.d,
            "R_mean": self.R_mean,
            "R_median": self.R_median,
            "R_q95": self.R_q95,
            "R_q99": self.R_q99,
            "loglog_slope": self.loglog_slope,
        }


@dataclass
class SMeas:
    """Measurement / campaign metadata fields."""

    distances: list[int]
    noise: float
    shots: int
    campaigns: int
    kernel: str
    auto_threshold: int
    rounds_factor: int
    window_factor: int
    commit_stride_factor: int
    seed: int
    empty_output_fraction: float
    n_windows: int
    extra: dict[str, Any] = field(default_factory=dict)

    def as_dict(self) -> dict:
        d = {
            "distances": list(self.distances),
            "noise": self.noise,
            "shots": self.shots,
            "campaigns": self.campaigns,
            "kernel": self.kernel,
            "auto_threshold": self.auto_threshold,
            "rounds_factor": self.rounds_factor,
            "window_factor": self.window_factor,
            "commit_stride_factor": self.commit_stride_factor,
            "seed": self.seed,
            "empty_output_fraction": self.empty_output_fraction,
            "n_windows": self.n_windows,
        }
        d.update(self.extra)
        return d


def _safe_ratio(num: float, den: float) -> float:
    if den == 0 or math.isnan(den) or math.isnan(num):
        return float("nan")
    return float(num / den)


def _quantile(x: np.ndarray, q: float) -> float:
    if x.size == 0:
        return float("nan")
    return float(np.quantile(x, q))


def boundary_ratios(
    tau_stage: np.ndarray,
    tau_match: np.ndarray,
    d: int,
    loglog_slope: Optional[float] = None,
) -> BoundaryRatios:
    """Compute R_q(d) for mean, median, q95, q99."""
    tau_stage = np.asarray(tau_stage, dtype=np.float64)
    tau_match = np.asarray(tau_match, dtype=np.float64)
    return BoundaryRatios(
        d=d,
        R_mean=_safe_ratio(float(tau_stage.mean()) if tau_stage.size else float("nan"),
                           float(tau_match.mean()) if tau_match.size else float("nan")),
        R_median=_safe_ratio(_quantile(tau_stage, 0.5), _quantile(tau_match, 0.5)),
        R_q95=_safe_ratio(_quantile(tau_stage, 0.95), _quantile(tau_match, 0.95)),
        R_q99=_safe_ratio(_quantile(tau_stage, 0.99), _quantile(tau_match, 0.99)),
        loglog_slope=loglog_slope,
    )


def loglog_ols_slope(
    distances: Sequence[int],
    values: Sequence[float],
) -> Optional[float]:
    """Optional log-log OLS slope of values vs distance."""
    ds = np.asarray(distances, dtype=np.float64)
    vs = np.asarray(values, dtype=np.float64)
    mask = (ds > 0) & (vs > 0) & np.isfinite(vs)
    if mask.sum() < 2:
        return None
    x = np.log(ds[mask])
    y = np.log(vs[mask])
    x_mean = x.mean()
    y_mean = y.mean()
    var = ((x - x_mean) ** 2).sum()
    if var == 0:
        return None
    return float(((x - x_mean) * (y - y_mean)).sum() / var)


def offered_load(
    E_tau_B: float,
    k_workers: int,
    C: int,
    tau_c: float,
) -> float:
    """Offered load ρ = E[τ_B] / (k * C * τ_c)."""
    den = k_workers * C * tau_c
    if den == 0:
        return float("nan")
    return float(E_tau_B / den)


def summarize(
    records: Sequence[Mapping[str, Any]],
    *,
    distances: Sequence[int],
    noise: float,
    shots: int,
    campaigns: int,
    kernel: str,
    auto_threshold: int,
    rounds_factor: int,
    window_factor: int,
    commit_stride_factor: int,
    seed: int,
    fit_loglog_slope: bool = True,
    k_workers: int = 1,
    tau_c_ns: Optional[float] = None,
) -> dict[str, Any]:
    """Aggregate WindowRecords into mixture stats, ratios, and S_meas."""
    if not records:
        s_meas = SMeas(
            distances=list(distances),
            noise=noise,
            shots=shots,
            campaigns=campaigns,
            kernel=kernel,
            auto_threshold=auto_threshold,
            rounds_factor=rounds_factor,
            window_factor=window_factor,
            commit_stride_factor=commit_stride_factor,
            seed=seed,
            empty_output_fraction=0.0,
            n_windows=0,
        )
        return {"by_distance": {}, "S_meas": s_meas.as_dict(), "deltas": []}

    empty_flags = [bool(r.get("empty_output", False)) for r in records]
    empty_frac = float(sum(empty_flags) / len(empty_flags))

    by_d: dict[int, dict[str, Any]] = {}
    R_means: list[float] = []
    ds_for_slope: list[int] = []

    for d in distances:
        subset = [r for r in records if int(r["d"]) == int(d)]
        if not subset:
            continue
        K = np.array([r["K"] for r in subset])
        T_post = np.array([r["tau_post_ns"] for r in subset], dtype=np.float64)
        tau_stage = np.array([r["tau_stage_ns"] for r in subset], dtype=np.float64)
        tau_match = np.array([r["tau_match_ns"] for r in subset], dtype=np.float64)
        mix = compute_mixture(K, T_post)
        ratios = boundary_ratios(tau_stage, tau_match, d=int(d))
        C = commit_stride_factor * int(d)
        e_tau_b = float(tau_stage.mean()) if tau_stage.size else float("nan")
        tc = tau_c_ns if tau_c_ns is not None else (
            float(np.median(tau_match) / max(C, 1)) if tau_match.size else float("nan")
        )
        rho = offered_load(e_tau_b, k_workers, C, tc) if np.isfinite(tc) else float("nan")
        by_d[int(d)] = {
            "mixture": mix.as_dict(),
            "ratios": ratios.as_dict(),
            "offered_load": rho,
            "n_windows": len(subset),
            "E_tau_stage": e_tau_b,
            "E_tau_match": float(tau_match.mean()) if tau_match.size else float("nan"),
            "E_tau_input": float(np.mean([r["tau_input_ns"] for r in subset])),
            "E_tau_post": mix.expectation,
            "pi": mix.pi,
        }
        if np.isfinite(ratios.R_mean):
            R_means.append(ratios.R_mean)
            ds_for_slope.append(int(d))

    slope = loglog_ols_slope(ds_for_slope, R_means) if fit_loglog_slope else None
    if slope is not None:
        for d, info in by_d.items():
            info["ratios"]["loglog_slope"] = slope

    from mabs.mixture import MixtureStats as MS, delta_decomposition

    deltas = []
    d_list = [int(d) for d in distances if int(d) in by_d]
    for i in range(len(d_list) - 1):
        da, db = d_list[i], d_list[i + 1]
        ma = by_d[da]["mixture"]
        mb = by_d[db]["mixture"]
        sa = MS(ma["pi"], ma["mu_plus"], ma["mu0"], ma["E_T_post"], ma["n"], ma["n_positive"], ma["n_zero"])
        sb = MS(mb["pi"], mb["mu_plus"], mb["mu0"], mb["E_T_post"], mb["n"], mb["n_positive"], mb["n_zero"])
        deltas.append(delta_decomposition(sa, sb, da, db).as_dict())

    s_meas = SMeas(
        distances=list(distances),
        noise=noise,
        shots=shots,
        campaigns=campaigns,
        kernel=kernel,
        auto_threshold=auto_threshold,
        rounds_factor=rounds_factor,
        window_factor=window_factor,
        commit_stride_factor=commit_stride_factor,
        seed=seed,
        empty_output_fraction=empty_frac,
        n_windows=len(records),
        extra={"loglog_slope_R_mean": slope},
    )
    return {"by_distance": by_d, "S_meas": s_meas.as_dict(), "deltas": deltas}


# Back-compat alias for older imports / mirrors
boundary_factor = boundary_ratios
