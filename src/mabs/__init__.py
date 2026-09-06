"""MABS v4: CASCADE streaming QEC decoder (ExactPatternCache + clique + deferred escalation)."""

from mabs.config import MABSConfig
from mabs.algorithm import run_mabs, MABSResult
from mabs.timing import NestedTimers, TimingSample
from mabs.mixture import MixtureStats, compute_mixture, delta_decomposition
from mabs.reporting import boundary_ratios, boundary_factor, offered_load, SMeas, summarize
from mabs.confidence import confidence_score, ConfidenceConfig, calibrate_auto_threshold
from mabs.adaptive import AdaptiveConfig, AdaptiveState, stream_shot_adaptive_timed
from mabs.v3.slem import SLEMConfig, SLEMState, stream_shot_slem
from mabs.v4.cascade import CASCADEConfig, CASCADEState, stream_shot_cascade

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
    "SLEMConfig",
    "SLEMState",
    "stream_shot_slem",
    "CASCADEConfig",
    "CASCADEState",
    "stream_shot_cascade",
]

__version__ = "4.0.1a2"
