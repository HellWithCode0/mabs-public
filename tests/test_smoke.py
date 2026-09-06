"""End-to-end smoke: run_mabs for d=3, CLI-shaped summary fields."""

from __future__ import annotations

from mabs.algorithm import run_mabs
from mabs.config import MABSConfig
from mabs.reporting import offered_load


def test_run_mabs_d3_smoke():
    cfg = MABSConfig(
        distances=(3,),
        noise=0.001,
        shots=2,
        campaigns=1,
        kernel="vector",
        seed=1,
        adaptive=False,
        track_ler=True,
    )
    result = run_mabs(cfg)
    assert len(result.records) > 0
    assert 3 in result.by_distance
    info = result.by_distance[3]
    assert "pi" in info
    assert "ratios" in info
    assert "R_mean" in info["ratios"]
    assert info["ratios"]["R_mean"] >= 1.0
    assert 0.0 <= info["pi"] <= 1.0
    s = result.S_meas
    assert "empty_output_fraction" in s
    assert s["n_windows"] == len(result.records)
    for r in result.records:
        assert r["tau_stage_ns"] == r["tau_input_ns"] + r["tau_match_ns"] + r["tau_post_ns"]
    assert 3 in result.ler_by_distance


def test_run_mabs_adaptive_d3_d5():
    cfg = MABSConfig(
        distances=(3, 5),
        noise=0.001,
        shots=2,
        campaigns=1,
        kernel="auto",
        seed=2,
        adaptive=True,
        track_ler=True,
    )
    result = run_mabs(cfg)
    for d in (3, 5):
        info = result.by_distance[d]
        assert info["ratios"]["R_mean"] == info["ratios"]["R_mean"]
        assert 0.0 <= info["pi"] <= 1.0
    assert result.summary.get("deltas")
    assert "adaptive" in result.summary


def test_offered_load_formula():
    rho = offered_load(E_tau_B=100.0, k_workers=2, C=5, tau_c=10.0)
    assert abs(rho - 100.0 / (2 * 5 * 10.0)) < 1e-12
