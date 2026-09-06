"""Tests for nested timers τ_input, τ_match, τ_post; τ_stage = sum."""

from __future__ import annotations

import time

from mabs.timing import NestedTimers, TimingSample, assert_nesting


def test_stage_is_exact_sum():
    s = TimingSample(tau_input_ns=100, tau_match_ns=200, tau_post_ns=50)
    assert s.tau_stage_ns == 350
    assert_nesting(s)


def test_nested_timers_accumulate():
    t = NestedTimers()
    with t.input():
        time.sleep(0.001)
    with t.match():
        time.sleep(0.001)
    with t.post():
        time.sleep(0.001)
    sample = t.sample()
    assert sample.tau_input_ns > 0
    assert sample.tau_match_ns > 0
    assert sample.tau_post_ns > 0
    assert sample.tau_stage_ns == (
        sample.tau_input_ns + sample.tau_match_ns + sample.tau_post_ns
    )
    assert_nesting(sample)


def test_reset_clears():
    t = NestedTimers()
    with t.input():
        pass
    t.reset()
    s = t.sample()
    assert s.tau_input_ns == 0
    assert s.tau_match_ns == 0
    assert s.tau_post_ns == 0
