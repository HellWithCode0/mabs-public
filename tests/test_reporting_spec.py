"""S_meas / S_arch checklist, paired overhead, and the boundary-slope identity."""

from __future__ import annotations

import numpy as np
import pytest

from mabs.reporting import (
    DeadlineInterpretationError,
    SArch,
    boundary_ratios,
    boundary_slope_contrast,
    deadline_burden,
    host_metadata,
    loglog_ols_slope,
    paired_overhead,
    summarize,
)


def test_boundary_slope_contrast_equals_slope_of_ratio():
    """Delta alpha_I(q) is identically the log-log slope of R_q over I."""
    ds = [3, 5, 7, 9, 11, 13]
    rng = np.random.default_rng(0)
    match = np.array([120.0 * d**1.7 for d in ds]) * (1 + 0.02 * rng.standard_normal(6))
    stage = match * np.array([1.30, 1.18, 1.11, 1.07, 1.05, 1.04])
    c = boundary_slope_contrast(ds, stage, match)
    assert c.interval == ds
    assert c.delta_alpha is not None
    assert c.alpha_of_ratio is not None
    # Identity, not corroboration: the residual is float noise only.
    assert c.identity_residual < 1e-12
    assert c.delta_alpha == pytest.approx(c.alpha_of_ratio, abs=1e-12)
    # Ratio falls with d here, so the contrast must be negative.
    assert c.delta_alpha < 0


def test_boundary_slope_contrast_matches_manual_slopes():
    ds = [3, 5, 7]
    match = [100.0, 300.0, 600.0]
    stage = [150.0, 400.0, 720.0]
    c = boundary_slope_contrast(ds, stage, match, statistic="q99")
    assert c.statistic == "q99"
    assert c.alpha_stage == pytest.approx(loglog_ols_slope(ds, stage))
    assert c.alpha_match == pytest.approx(loglog_ols_slope(ds, match))
    assert c.delta_alpha == pytest.approx(c.alpha_stage - c.alpha_match)


def test_boundary_slope_contrast_rejects_misaligned_input():
    with pytest.raises(ValueError):
        boundary_slope_contrast([3, 5], [1.0, 2.0], [1.0])


def test_paired_overhead_is_on_the_same_rows():
    match = np.array([100.0, 200.0, 400.0, 800.0])
    stage = np.array([150.0, 260.0, 480.0, 1600.0])
    o = paired_overhead(stage, match, d=5)
    assert o.n == 4
    assert o.mean_ns == pytest.approx(np.mean(stage - match))
    assert o.median_ns == pytest.approx(np.median(stage - match))
    assert o.ratio_median == pytest.approx(np.median(stage / match))
    # The paired ratio and the ratio of summaries are different objects.
    r = boundary_ratios(stage, match, d=5)
    assert o.ratio_q99 != pytest.approx(r.R_q99)


def test_paired_overhead_rejects_unpaired_rows():
    with pytest.raises(ValueError):
        paired_overhead(np.zeros(4), np.zeros(3), d=3)


def test_paired_overhead_skips_zero_match_in_ratio_only():
    stage = np.array([10.0, 20.0])
    match = np.array([0.0, 10.0])
    o = paired_overhead(stage, match, d=3)
    assert o.n == 2
    assert o.mean_ns == pytest.approx(10.0)
    assert o.ratio_mean == pytest.approx(2.0)


def test_deadline_burden_refuses_incomplete_s_arch():
    with pytest.raises(DeadlineInterpretationError):
        deadline_burden(1000.0, SArch())
    with pytest.raises(DeadlineInterpretationError):
        deadline_burden(1000.0, SArch(slack_ns=5000.0, tau_c_ns=1000.0))
    # Residual present but unattributed is still refused.
    with pytest.raises(DeadlineInterpretationError):
        deadline_burden(1000.0, SArch(slack_ns=5000.0, tau_c_ns=1000.0, residual_ns=0.0))


def test_deadline_burden_with_complete_s_arch():
    arch = SArch(
        c_of_d=1.0,
        slack_ns=5000.0,
        tau_c_ns=1000.0,
        residual_ns=500.0,
        residual_source="modelled: FCFS replay",
    )
    assert arch.missing_for_deadline() == []
    assert deadline_burden(2000.0, arch) == pytest.approx(0.5)


def test_host_metadata_declares_what_it_cannot_measure():
    h = host_metadata()
    assert h["python"]
    assert "numpy" in h["versions"]
    assert "OMP_NUM_THREADS" in h["thread_env"]
    # Affinity and governor are not portably readable; they are declared None
    # rather than filled with a value the host cannot support.
    assert h["affinity_pinned"] is None
    assert h["cpu_governor"] is None


def _record(d, stage, match, post, K):
    return {
        "d": d,
        "K": K,
        "tau_input_ns": stage - match - post,
        "tau_match_ns": match,
        "tau_post_ns": post,
        "tau_stage_ns": stage,
        "empty_output": K == 0,
    }


def test_summarize_carries_the_spec_fields():
    records = []
    for d, base in ((3, 1000.0), (5, 3000.0)):
        for i in range(20):
            match = base + 10.0 * i
            records.append(_record(d, match * 1.4 + 50.0, match, 30.0 + i, i % 3))
    out = summarize(
        records,
        distances=[3, 5],
        noise=1e-3,
        shots=20,
        campaigns=1,
        kernel="auto",
        auto_threshold=64,
        rounds_factor=10,
        window_factor=3,
        commit_stride_factor=1,
        seed=42,
    )
    s = out["S_meas"]
    for key in ("boundary", "execution", "statistic", "fit_interval", "fit_model", "host"):
        assert key in s
    assert s["fit_interval"] == [3, 5]
    assert "tau_stage" in s["boundary"]
    c = out["boundary_contrast"]
    assert c is not None
    assert c["identity_residual"] < 1e-9
    assert "overhead" in out["by_distance"][3]
    assert out["by_distance"][3]["overhead"]["n"] == 20
