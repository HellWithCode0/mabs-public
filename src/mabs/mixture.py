"""Mixture accounting for post-matching latency.

π = P(K > 0)
μ+ = E[T_post | K > 0]
μ0 = E[T_post | K = 0]
E[T_post] = π μ+ + (1 − π) μ0

Δ decomposition between two distances compares these components.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Optional
import numpy as np


@dataclass
class MixtureStats:
    """Mixture-of-two model for post-matching time."""

    pi: float  # P(K > 0)
    mu_plus: float  # E[T_post | K > 0] (ns)
    mu0: float  # E[T_post | K = 0] (ns)
    expectation: float  # E[T_post] = πμ+ + (1-π)μ0
    n: int
    n_positive: int
    n_zero: int

    def as_dict(self) -> dict:
        return {
            "pi": self.pi,
            "mu_plus": self.mu_plus,
            "mu0": self.mu0,
            "E_T_post": self.expectation,
            "n": self.n,
            "n_positive": self.n_positive,
            "n_zero": self.n_zero,
        }


@dataclass
class DeltaDecomposition:
    """Δ E[T_post] decomposition between distance d_a and d_b."""

    d_a: int
    d_b: int
    delta_expectation: float
    delta_pi: float
    delta_mu_plus: float
    delta_mu0: float
    # Approximate additive split: ΔE ≈ (Δπ)(μ+_ref) + π_ref(Δμ+) + ...
    term_pi: float
    term_mu_plus: float
    term_mu0: float

    def as_dict(self) -> dict:
        return {
            "d_a": self.d_a,
            "d_b": self.d_b,
            "delta_E": self.delta_expectation,
            "delta_pi": self.delta_pi,
            "delta_mu_plus": self.delta_mu_plus,
            "delta_mu0": self.delta_mu0,
            "term_pi": self.term_pi,
            "term_mu_plus": self.term_mu_plus,
            "term_mu0": self.term_mu0,
        }


def compute_mixture(
    K: np.ndarray,
    T_post: np.ndarray,
) -> MixtureStats:
    """Compute mixture statistics from per-window K and T_post arrays."""
    K = np.asarray(K)
    T_post = np.asarray(T_post, dtype=np.float64)
    if K.shape != T_post.shape:
        raise ValueError("K and T_post must have the same shape")
    n = int(K.size)
    if n == 0:
        return MixtureStats(0.0, 0.0, 0.0, 0.0, 0, 0, 0)
    pos = K > 0
    n_pos = int(pos.sum())
    n_zero = n - n_pos
    pi = n_pos / n
    mu_plus = float(T_post[pos].mean()) if n_pos else 0.0
    mu0 = float(T_post[~pos].mean()) if n_zero else 0.0
    expectation = pi * mu_plus + (1.0 - pi) * mu0
    return MixtureStats(pi, mu_plus, mu0, expectation, n, n_pos, n_zero)


def delta_decomposition(
    a: MixtureStats,
    b: MixtureStats,
    d_a: int,
    d_b: int,
    ref: Optional[str] = "a",
) -> DeltaDecomposition:
    """Decompose ΔE[T_post] between two mixture models.

    Using reference mixture ``ref`` ('a' or 'b') for first-order split::

        ΔE ≈ (Δπ) μ+_ref + π_ref (Δμ+) + (Δ(1-π)) μ0_ref + (1-π_ref)(Δμ0)
    """
    delta_e = b.expectation - a.expectation
    delta_pi = b.pi - a.pi
    delta_mu_plus = b.mu_plus - a.mu_plus
    delta_mu0 = b.mu0 - a.mu0
    if ref == "b":
        pi_r, mu_p_r, mu0_r = b.pi, b.mu_plus, b.mu0
    else:
        pi_r, mu_p_r, mu0_r = a.pi, a.mu_plus, a.mu0
    # E = π μ+ + (1-π) μ0
    # dE ≈ dπ μ+ + π dμ+ - dπ μ0 + (1-π) dμ0
    term_pi = delta_pi * (mu_p_r - mu0_r)
    term_mu_plus = pi_r * delta_mu_plus
    term_mu0 = (1.0 - pi_r) * delta_mu0
    return DeltaDecomposition(
        d_a=d_a,
        d_b=d_b,
        delta_expectation=delta_e,
        delta_pi=delta_pi,
        delta_mu_plus=delta_mu_plus,
        delta_mu0=delta_mu0,
        term_pi=term_pi,
        term_mu_plus=term_mu_plus,
        term_mu0=term_mu0,
    )
