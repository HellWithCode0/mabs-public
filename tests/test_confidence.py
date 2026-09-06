"""Tests for mixture-aware confidence score."""

from __future__ import annotations

from mabs.confidence import (
    ConfidenceConfig,
    calibrate_auto_threshold,
    confidence_score,
    predecode_confidence,
    should_retry,
)
import numpy as np


def test_empty_is_high_confidence():
    q = confidence_score(
        K=0, empty_output=True, matching_weight=0.0, syndrome_density=0.0
    )
    assert q > 0.8


def test_dense_hard_is_low_confidence():
    q = confidence_score(
        K=40, empty_output=False, matching_weight=200.0, syndrome_density=0.5,
        distance=3,
    )
    assert q < 0.4


def test_score_bounded():
    for K in (0, 1, 10, 100):
        q = confidence_score(
            K=K, empty_output=K == 0, matching_weight=float(K), syndrome_density=0.01 * K
        )
        assert 0.0 <= q <= 1.0


def test_predecode_and_calibrate():
    assert predecode_confidence(0.0) > predecode_confidence(0.2)
    thr = calibrate_auto_threshold(np.array([0, 1, 2, 8, 64, 100]))
    assert 8 <= thr <= 256


def test_should_retry_gate():
    # Easy window: no retry
    q = confidence_score(K=0, empty_output=True, matching_weight=0.0, syndrome_density=0.0, distance=5)
    assert not should_retry(K=0, matching_weight=20.0, syndrome_density=0.0, Q=q, threshold=0.4, distance=5)
    # Mildly busy: soft-low Q but below hard gate → no retry
    assert not should_retry(K=5, matching_weight=20.0, syndrome_density=0.02, Q=0.2, threshold=0.4, distance=5)
    # Extreme hardness: retry
    q2 = confidence_score(K=40, empty_output=False, matching_weight=200.0, syndrome_density=0.3, distance=5)
    assert should_retry(K=40, matching_weight=200.0, syndrome_density=0.3, Q=q2, threshold=0.4, distance=5)
