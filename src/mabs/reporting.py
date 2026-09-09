"""Boundary ratios, offered load, and S_meas / S_arch reporting metadata.

Field names follow the reporting checklist of the manuscript this decoder
accompanies: S_meas accompanies every timing claim, S_arch is conditional on an
accounting or deadline reading and is never required to reproduce a CPU timing
measurement.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Mapping, Optional, Sequence
import math
import os
import platform
import sys
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
class OverheadStats:
    """Paired absolute overhead on one set of per-window rows.

    R_q compares two marginal summaries, so R_99 is not a tail-overhead factor
    for any one window. These paired statistics answer the other question: what
    a single window actually pays for the wider boundary. Both are reported.
    """

    d: int
    n: int
    mean_ns: float
    median_ns: float
    q95_ns: float
    q99_ns: float
    ratio_mean: float
    ratio_median: float
    ratio_q95: float
    ratio_q99: float

    def as_dict(self) -> dict:
        return {
            "d": self.d,
            "n": self.n,
            "mean_ns": self.mean_ns,
            "median_ns": self.median_ns,
            "q95_ns": self.q95_ns,
            "q99_ns": self.q99_ns,
            "ratio_mean": self.ratio_mean,
            "ratio_median": self.ratio_median,
            "ratio_q95": self.ratio_q95,
            "ratio_q99": self.ratio_q99,
        }


@dataclass
class BoundaryContrast:
    """Boundary slope contrast over a declared interval I and statistic q.

    ``delta_alpha`` is the contrast of the two fitted log-log slopes.
    ``alpha_of_ratio`` is the log-log slope of R_q itself over the same
    interval. They are the same number, not two independent results, so
    ``identity_residual`` is carried alongside as the check rather than as
    corroboration.
    """

    statistic: str
    interval: list[int]
    alpha_stage: Optional[float]
    alpha_match: Optional[float]
    delta_alpha: Optional[float]
    alpha_of_ratio: Optional[float]
    identity_residual: Optional[float]

    def as_dict(self) -> dict:
        return {
            "statistic": self.statistic,
            "interval": list(self.interval),
            "alpha_stage": self.alpha_stage,
            "alpha_match": self.alpha_match,
            "delta_alpha": self.delta_alpha,
            "alpha_of_ratio": self.alpha_of_ratio,
            "identity_residual": self.identity_residual,
        }


def host_metadata() -> dict:
    """Hardware / software / threading fields required beside a timing claim."""
    versions = {}
    for name in ("numpy", "stim", "pymatching", "numba"):
        mod = sys.modules.get(name)
        if mod is None:
            try:
                mod = __import__(name)
            except ImportError:
                versions[name] = None
                continue
        versions[name] = getattr(mod, "__version__", "unknown")
    thread_env = {
        key: os.environ.get(key)
        for key in (
            "OMP_NUM_THREADS",
            "MKL_NUM_THREADS",
            "OPENBLAS_NUM_THREADS",
            "NUMEXPR_NUM_THREADS",
        )
    }
    return {
        "platform": platform.platform(),
        "machine": platform.machine(),
        "processor": platform.processor(),
        "python": platform.python_version(),
        "cpu_count": os.cpu_count(),
        "versions": versions,
        "thread_env": thread_env,
        # Affinity and CPU governor are not exposed portably; declare that
        # rather than reporting a value the host cannot support.
        "affinity_pinned": None,
        "cpu_governor": None,
    }


@dataclass
class SMeas:
    """Measurement / campaign metadata fields.

    ``boundary``, ``execution``, ``statistic``, ``fit_interval``, ``fit_model``
    and ``host`` complete the S_meas checklist: which timestamp endpoints were
    used, which timed implementation ran, which summary statistic is reported,
    and, when a fit is reported, over which interval and model.
    """

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
    boundary: str = "tau_stage = tau_input + tau_match + tau_post"
    execution: str = "python-streaming/pymatching-sparse-blossom"
    statistic: str = "mean"
    fit_interval: Optional[list[int]] = None
    fit_model: Optional[str] = None
    host: dict[str, Any] = field(default_factory=host_metadata)
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
            "boundary": self.boundary,
            "execution": self.execution,
            "statistic": self.statistic,
            "fit_interval": None if self.fit_interval is None else list(self.fit_interval),
            "fit_model": self.fit_model,
            "host": dict(self.host),
        }
        d.update(self.extra)
        return d


class DeadlineInterpretationError(ValueError):
    """Raised when a deadline reading is asked for without its S_arch fields."""


@dataclass
class SArch:
    """Architectural fields, conditional on an accounting or deadline reading.

    None of these are measured by a CPU timing run. A deadline reading also
    needs a residual, measured or modelled, so ``deadline_burden`` refuses
    rather than substituting a default.
    """

    c_of_d: Optional[float] = None       # architectural accounting unit c(d)
    slack_ns: Optional[float] = None     # reaction slack Lambda(d)
    tau_c_ns: Optional[float] = None     # round duration tau_c
    residual_ns: Optional[float] = None  # tau_res, measured or modelled
    residual_source: Optional[str] = None

    def missing_for_deadline(self) -> list[str]:
        missing = []
        if self.slack_ns is None:
            missing.append("slack_ns")
        if self.tau_c_ns is None:
            missing.append("tau_c_ns")
        if self.residual_ns is None:
            missing.append("residual_ns")
        elif not self.residual_source:
            missing.append("residual_source")
        return missing

    def as_dict(self) -> dict:
        return {
            "c_of_d": self.c_of_d,
            "slack_ns": self.slack_ns,
            "tau_c_ns": self.tau_c_ns,
            "residual_ns": self.residual_ns,
            "residual_source": self.residual_source,
        }


def deadline_burden(tau_ns: float, arch: SArch) -> float:
    """Fraction of the reaction slack a measured interval consumes.

    Refuses on an incomplete S_arch. A CPU timing measurement alone does not
    license a deadline claim, and silently defaulting the slack or the residual
    would turn a measurement into an architectural claim it cannot support.
    """
    missing = arch.missing_for_deadline()
    if missing:
        raise DeadlineInterpretationError(
            "deadline reading needs S_arch fields: " + ", ".join(missing)
        )
    slack = float(arch.slack_ns)
    if slack <= 0:
        raise DeadlineInterpretationError("slack_ns must be positive")
    return float((float(tau_ns) + float(arch.residual_ns)) / slack)


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


def paired_overhead(
    tau_stage: np.ndarray,
    tau_match: np.ndarray,
    d: int,
) -> OverheadStats:
    """Paired tau_stage - tau_match and tau_stage / tau_match on the same rows.

    The two arrays must be the identical per-window rows in the same order, so
    every pairing is exact. Rows with tau_match == 0 are dropped from the ratio
    statistics only; they still count in the difference statistics.
    """
    stage = np.asarray(tau_stage, dtype=np.float64).ravel()
    match = np.asarray(tau_match, dtype=np.float64).ravel()
    if stage.size != match.size:
        raise ValueError(
            f"paired_overhead needs the same rows: {stage.size} vs {match.size}"
        )
    if stage.size == 0:
        nan = float("nan")
        return OverheadStats(d=int(d), n=0, mean_ns=nan, median_ns=nan, q95_ns=nan,
                             q99_ns=nan, ratio_mean=nan, ratio_median=nan,
                             ratio_q95=nan, ratio_q99=nan)
    diff = stage - match
    ok = match > 0
    ratio = stage[ok] / match[ok] if np.any(ok) else np.zeros(0, dtype=np.float64)
    return OverheadStats(
        d=int(d),
        n=int(stage.size),
        mean_ns=float(diff.mean()),
        median_ns=_quantile(diff, 0.5),
        q95_ns=_quantile(diff, 0.95),
        q99_ns=_quantile(diff, 0.99),
        ratio_mean=float(ratio.mean()) if ratio.size else float("nan"),
        ratio_median=_quantile(ratio, 0.5),
        ratio_q95=_quantile(ratio, 0.95),
        ratio_q99=_quantile(ratio, 0.99),
    )


def boundary_slope_contrast(
    distances: Sequence[int],
    q_stage: Sequence[float],
    q_match: Sequence[float],
    *,
    statistic: str = "mean",
) -> BoundaryContrast:
    """Delta alpha_I(q) = alpha_I(q(tau_stage)) - alpha_I(q(tau_match)).

    Also returns alpha_I(R_q), the log-log slope of the ratio curve. The OLS
    slope is linear in the log response, so the two are identically equal;
    ``identity_residual`` records the numerical gap and exists as a check on
    the fit range, not as a second piece of evidence.
    """
    ds = np.asarray(distances, dtype=np.float64)
    st = np.asarray(q_stage, dtype=np.float64)
    mt = np.asarray(q_match, dtype=np.float64)
    if not (ds.size == st.size == mt.size):
        raise ValueError("distances, q_stage and q_match must align")
    mask = (ds > 0) & (st > 0) & (mt > 0) & np.isfinite(st) & np.isfinite(mt)
    interval = [int(x) for x in ds[mask]]
    a_stage = loglog_ols_slope(ds[mask], st[mask])
    a_match = loglog_ols_slope(ds[mask], mt[mask])
    a_ratio = loglog_ols_slope(ds[mask], st[mask] / mt[mask])
    delta = None if (a_stage is None or a_match is None) else float(a_stage - a_match)
    resid = None if (delta is None or a_ratio is None) else float(abs(delta - a_ratio))
    return BoundaryContrast(
        statistic=statistic,
        interval=interval,
        alpha_stage=a_stage,
        alpha_match=a_match,
        delta_alpha=delta,
        alpha_of_ratio=a_ratio,
        identity_residual=resid,
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
            fit_model="power law q = A d^alpha_eff" if fit_loglog_slope else None,
        )
        return {
            "by_distance": {},
            "S_meas": s_meas.as_dict(),
            "deltas": [],
            "boundary_contrast": None,
        }

    empty_flags = [bool(r.get("empty_output", False)) for r in records]
    empty_frac = float(sum(empty_flags) / len(empty_flags))

    by_d: dict[int, dict[str, Any]] = {}
    R_means: list[float] = []
    ds_for_slope: list[int] = []
    stage_means: list[float] = []
    match_means: list[float] = []

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
        e_tau_match = float(tau_match.mean()) if tau_match.size else float("nan")
        by_d[int(d)] = {
            "mixture": mix.as_dict(),
            "ratios": ratios.as_dict(),
            "overhead": paired_overhead(tau_stage, tau_match, d=int(d)).as_dict(),
            "offered_load": rho,
            "n_windows": len(subset),
            "E_tau_stage": e_tau_b,
            "E_tau_match": e_tau_match,
            "E_tau_input": float(np.mean([r["tau_input_ns"] for r in subset])),
            "E_tau_post": mix.expectation,
            "pi": mix.pi,
        }
        if np.isfinite(ratios.R_mean):
            R_means.append(ratios.R_mean)
            ds_for_slope.append(int(d))
            stage_means.append(e_tau_b)
            match_means.append(e_tau_match)

    slope = loglog_ols_slope(ds_for_slope, R_means) if fit_loglog_slope else None
    if slope is not None:
        for d, info in by_d.items():
            info["ratios"]["loglog_slope"] = slope

    contrast = (
        boundary_slope_contrast(ds_for_slope, stage_means, match_means, statistic="mean")
        if (fit_loglog_slope and len(ds_for_slope) >= 2)
        else None
    )

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
        fit_interval=list(ds_for_slope) if slope is not None else None,
        fit_model="power law q = A d^alpha_eff" if slope is not None else None,
        extra={"loglog_slope_R_mean": slope},
    )
    return {
        "by_distance": by_d,
        "S_meas": s_meas.as_dict(),
        "deltas": deltas,
        "boundary_contrast": None if contrast is None else contrast.as_dict(),
    }


# Back-compat alias for older imports / mirrors
boundary_factor = boundary_ratios
