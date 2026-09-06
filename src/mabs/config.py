"""Configuration for MABS streaming campaigns."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Sequence


@dataclass
class MABSConfig:
    """Parameters for Mixture-Aware Adaptive Boundary Streaming (MABS v2).

    Follows Katoch-style streaming QEC decoder timing on Stim rotated
    surface-code memory with PyMatching, plus ADaPT-style adaptive windows.
    """

    distances: Sequence[int] = (3, 5)
    noise: float = 0.001
    rounds_factor: int = 10  # R = rounds_factor * d
    window_factor: int = 3  # w = window_factor * d (fixed / large policy)
    commit_stride_factor: int = 1  # C = commit_stride_factor * d
    shots: int = 10
    campaigns: int = 1
    kernel: str = "auto"  # naive | loop | vector | auto
    auto_threshold: int = 64
    seed: int = 0
    # Offered-load parameters
    k_workers: int = 1
    tau_c_ns: float | None = None
    quantiles: Sequence[float] = field(default_factory=lambda: (0.5, 0.95, 0.99))
    fit_loglog_slope: bool = True
    # Adaptive policy (MABS-Adaptive)
    adaptive: bool = True
    w_small_factor: float = 2.0
    confidence_threshold: float = 0.40
    max_retries_per_window: int = 1
    track_ler: bool = True

    def rounds(self, d: int) -> int:
        return self.rounds_factor * d

    def window_depth(self, d: int) -> int:
        return self.window_factor * d

    def commit_stride(self, d: int) -> int:
        return self.commit_stride_factor * d

    def window_depth_small(self, d: int) -> int:
        return max(1, int(round(self.w_small_factor * d)))
