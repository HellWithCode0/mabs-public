"""Adaptive escalation policy for SLEM."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Optional, Sequence

from mabs.v3.cluster import Cluster


@dataclass
class EscalationPolicy:
    """When to escalate from local decode to full-window Sparse Blossom.

    Tuned for the Katoch sparse mixture: keep escalation low at p≈1e-3 and
    rise only when clusters are large / ambiguous.
    """

    max_local_cluster: int = 2
    max_local_defects: int = 4
    max_local_density: float = 0.02
    cluster_radius: int = 1
    # Soft adaptivity from running empty / escalate rates.
    target_escalate_rate: float = 0.55
    tune: bool = False
    tune_every: int = 64
    tune_step: int = 1  # on max_local_defects


@dataclass
class EscalationDecision:
    escalate: bool
    reason: str


def should_escalate(
    *,
    n_defects: int,
    clusters: Sequence[Cluster],
    syndrome_density: float,
    policy: EscalationPolicy,
) -> EscalationDecision:
    if n_defects == 0:
        return EscalationDecision(False, "empty")
    if n_defects > policy.max_local_defects:
        return EscalationDecision(True, "too_many_defects")
    if syndrome_density > policy.max_local_density:
        return EscalationDecision(True, "density")
    if not clusters:
        return EscalationDecision(True, "no_clusters")
    for c in clusters:
        if c.size > policy.max_local_cluster:
            return EscalationDecision(True, "large_cluster")
        if c.size >= 3 and c.odd:
            return EscalationDecision(True, "odd_large")
    return EscalationDecision(False, "local_ok")


def adapt_policy(policy: EscalationPolicy, escalate_rate: float) -> None:
    """Nudge local thresholds toward a target escalate rate."""
    if not policy.tune:
        return
    if escalate_rate > policy.target_escalate_rate + 0.10:
        # Escalating too often → allow slightly larger local clusters.
        policy.max_local_defects = min(12, policy.max_local_defects + policy.tune_step)
        policy.max_local_cluster = min(4, policy.max_local_cluster + (1 if policy.max_local_defects % 2 == 0 else 0))
    elif escalate_rate < policy.target_escalate_rate - 0.15:
        policy.max_local_defects = max(2, policy.max_local_defects - policy.tune_step)
        policy.max_local_cluster = max(1, min(policy.max_local_cluster, policy.max_local_defects))
