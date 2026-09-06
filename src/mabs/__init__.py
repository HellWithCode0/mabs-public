"""MABS v2: Mixture-Aware Adaptive Boundary Streaming for streaming QEC."""

from mabs.config import MABSConfig
from mabs.algorithm import run_mabs, MABSResult
from mabs.timing import NestedTimers, TimingSample
from mabs.mixture import MixtureStats, compute_mixture, delta_decomposition
from mabs.reporting import boundary_ratios, boundary_factor, offered_load, SMeas, summarize
from mabs.confidence import confidence_score, ConfidenceConfig, calibrate_auto_threshold
from mabs.adaptive import AdaptiveConfig, AdaptiveState, stream_shot_adaptive_timed

__all__ = [
    "MABSConfig",
    "run_mabs",
    "MABSResult",
    "NestedTimers",
    "TimingSample",
    "MixtureStats",
    "compute_mixture",
    "delta_decomposition",
    "boundary_ratios",
    "boundary_factor",
    "offered_load",
    "SMeas",
    "summarize",
    "confidence_score",
    "ConfidenceConfig",
    "calibrate_auto_threshold",
    "AdaptiveConfig",
    "AdaptiveState",
    "stream_shot_adaptive_timed",
]

__version__ = "2.1.1"
