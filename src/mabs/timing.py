"""Nested timers: τ_input, τ_match, τ_post; τ_stage = sum."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Optional
import time


@dataclass
class TimingSample:
    """One stage timing sample in nanoseconds."""

    tau_input_ns: int
    tau_match_ns: int
    tau_post_ns: int

    @property
    def tau_stage_ns(self) -> int:
        return self.tau_input_ns + self.tau_match_ns + self.tau_post_ns


class NestedTimers:
    """Context-manager style nested interval timers using time.perf_counter_ns.

    Usage::

        timers = NestedTimers()
        with timers.input():
            ...
        with timers.match():
            ...
        with timers.post():
            ...
        sample = timers.sample()
    """

    def __init__(self) -> None:
        self._input: int = 0
        self._match: int = 0
        self._post: int = 0
        self._t0: Optional[int] = None
        self._phase: Optional[str] = None

    def reset(self) -> None:
        self._input = self._match = self._post = 0
        self._t0 = None
        self._phase = None

    def _start(self, phase: str) -> None:
        if self._phase is not None:
            raise RuntimeError(f"timer already in phase {self._phase}")
        self._phase = phase
        self._t0 = time.perf_counter_ns()

    def _stop(self) -> int:
        if self._t0 is None or self._phase is None:
            raise RuntimeError("timer not running")
        dt = time.perf_counter_ns() - self._t0
        phase = self._phase
        self._t0 = None
        self._phase = None
        if phase == "input":
            self._input += dt
        elif phase == "match":
            self._match += dt
        elif phase == "post":
            self._post += dt
        else:
            raise RuntimeError(f"unknown phase {phase}")
        return dt

    class _Phase:
        def __init__(self, timers: "NestedTimers", phase: str) -> None:
            self._timers = timers
            self._phase = phase

        def __enter__(self) -> "NestedTimers._Phase":
            self._timers._start(self._phase)
            return self

        def __exit__(self, *exc) -> None:
            self._timers._stop()

    def input(self) -> _Phase:
        return self._Phase(self, "input")

    def match(self) -> _Phase:
        return self._Phase(self, "match")

    def post(self) -> _Phase:
        return self._Phase(self, "post")

    def sample(self) -> TimingSample:
        return TimingSample(
            tau_input_ns=self._input,
            tau_match_ns=self._match,
            tau_post_ns=self._post,
        )


def assert_nesting(sample: TimingSample, tol_ns: int = 50) -> None:
    """Verify τ_stage ≈ τ_input + τ_match + τ_post within tolerance.

    The identity is exact by construction of TimingSample; this helper
    exists for tests that may recompute the sum independently.
    """
    s = sample.tau_input_ns + sample.tau_match_ns + sample.tau_post_ns
    if abs(s - sample.tau_stage_ns) > tol_ns:
        raise AssertionError(
            f"nesting violated: stage={sample.tau_stage_ns} sum={s} tol={tol_ns}"
        )
