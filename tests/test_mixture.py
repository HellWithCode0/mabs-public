"""Tests for mixture statistics and Δ decomposition (Eqs. 8–9)."""

from __future__ import annotations

import numpy as np

from mabs.mixture import compute_mixture, delta_decomposition


def test_expectation_identity_eq8():
    K = np.array([0, 0, 1, 2, 0, 3])
    T = np.array([10.0, 12.0, 40.0, 50.0, 11.0, 60.0])
    m = compute_mixture(K, T)
    assert m.n == 6
    assert m.n_positive == 3
    assert m.n_zero == 3
    assert abs(m.pi - 0.5) < 1e-12
    assert abs(m.mu0 - (10 + 12 + 11) / 3) < 1e-12
    assert abs(m.mu_plus - (40 + 50 + 60) / 3) < 1e-12
    expected = m.pi * m.mu_plus + (1 - m.pi) * m.mu0
    assert abs(m.expectation - expected) < 1e-12
    assert abs(m.expectation - float(T.mean())) < 1e-12


def test_delta_decomposition_first_order():
    Ka = np.array([0, 0, 0, 1, 1])
    Ta = np.array([5.0, 5.0, 5.0, 20.0, 20.0])
    Kb = np.array([0, 1, 1, 1, 1])
    Tb = np.array([4.0, 18.0, 18.0, 18.0, 18.0])
    a = compute_mixture(Ka, Ta)
    b = compute_mixture(Kb, Tb)
    delta = delta_decomposition(a, b, d_a=3, d_b=5)
    # First-order terms should approximately reconstruct ΔE
    approx = delta.term_pi + delta.term_mu_plus + delta.term_mu0
    assert abs(approx - delta.delta_expectation) < abs(delta.delta_expectation) * 0.5 + 5.0
    assert delta.d_a == 3 and delta.d_b == 5
    assert delta.delta_pi == b.pi - a.pi


def test_empty_inputs():
    m = compute_mixture(np.array([]), np.array([]))
    assert m.n == 0
    assert m.pi == 0.0
    assert m.expectation == 0.0


def test_all_zero_K():
    K = np.zeros(10, dtype=int)
    T = np.full(10, 7.0)
    m = compute_mixture(K, T)
    assert m.pi == 0.0
    assert m.mu_plus == 0.0
    assert abs(m.mu0 - 7.0) < 1e-12
    assert abs(m.expectation - 7.0) < 1e-12
