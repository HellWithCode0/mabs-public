"""MABS-Adaptive: Mixture-Aware Adaptive Boundary Streaming.

ADaPT-style adaptivity on truncated window DEMs + carry-forward.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Optional
import numpy as np

from mabs.confidence import (
    ConfidenceConfig, confidence_score, predecode_confidence, should_retry,
)
from mabs.streaming import CircuitBundle, ShotDecodeResult, WindowRecord
from mabs.streaming_graph import (
    commit_window_edges, get_window_matcher, make_window_spec,
    mask_to_obs_array, window_match_edges,
)
from mabs.timing import NestedTimers


@dataclass
class AdaptiveConfig:
    w_large_factor: float = 3.0
    w_small_factor: float = 2.0
    commit_factor: float = 1.0
    confidence: ConfidenceConfig = field(default_factory=ConfidenceConfig)
    max_retries_per_window: int = 1
    use_predecode_gate: bool = True
    predecode_threshold: float = 0.30
    tune_threshold: bool = True
    target_retry_rate: float = 0.05
    tune_step: float = 0.02
    tune_every: int = 32


@dataclass
class AdaptiveState:
    n_windows: int = 0
    n_retries: int = 0
    threshold: float = 0.40

    @property
    def retry_rate(self) -> float:
        return 0.0 if self.n_windows == 0 else self.n_retries / self.n_windows


def _window_sizes(d: int, cfg: AdaptiveConfig):
    w_large = max(1, int(round(cfg.w_large_factor * d)))
    w_small = max(1, int(round(cfg.w_small_factor * d)))
    if w_small > w_large:
        w_small = w_large
    C = max(1, int(round(cfg.commit_factor * d))
    return w_small, w_large, C
