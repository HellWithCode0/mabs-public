"""MABS v3.1 — Sparse Local Escalation Matching (SLEM)."""

from mabs.v3.slem import (
    SLEMConfig,
    SLEMState,
    stream_shot_slem,
    stream_shot_slem_timed,
    prewarm_slem_graphs,
)
from mabs.v3.escalate import EscalationPolicy, should_escalate
from mabs.v3.cluster import Cluster, cluster_syndrome, build_detector_graph

__all__ = [
    "SLEMConfig",
    "SLEMState",
    "stream_shot_slem",
    "stream_shot_slem_timed",
    "prewarm_slem_graphs",
    "EscalationPolicy",
    "should_escalate",
    "Cluster",
    "cluster_syndrome",
    "build_detector_graph",
]

__version__ = "3.1.0a1"
