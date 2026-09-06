"""Adaptive escalation policy for SLEM v3.1."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Optional, Sequence

from mabs.v3.cluster import Cluster


@dataclass
class EscalationPolicy:
    """When to escalate from local decode to full-window Sparse Blossom.

    Tuned for the Katoch sparse mixture: keep escalation low at p≈1e-3 and
    rise only when clusters are large / ambiguous.

    v3.1 defaults raise local budgets (defects ≤ 8, cluster ≤ 4) and optionally
    ``tune`` toward ``target_escalate_rate≈0.20``.
    """

    max_local_cluster: int = 2
    max_local_defects: int = 12
    max_local_density: float = 0.05  # looser at low p
    cluster_radius: int = 0  # induced fired-subgraph (fast / sparse-correct)
    # Soft adaptivity from running empty / escalate rates.
    target_escalate_rate: float = 0.25
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
    max_comp: Optional[int] = None,
) -> EscalationDecision:
    """Prefer local whenever n_defects ≤ cap OR all clusters ≤ max_local_cluster."""
    if n_defects == 0:
        return EscalationDecision(False, "empty")
    if syndrome_density > policy.max_local_density and n_defects > policy.max_local_defects:
        return EscalationDecision(True, "density")
    if n_defects <= policy.max_local_defects:
        # Still local-ok even without cluster list (few-defect path).
        if max_comp is not None and max_comp > policy.max_local_cluster:
            return EscalationDecision(True, "large_cluster")
        if clusters:
            for c in clusters:
                if c.size > policy.max_local_cluster:
                    return EscalationDecision(True, "large_cluster")
        return EscalationDecision(False, "local_ok")
    # Many defects: only local if every cluster is tiny.
    if max_comp is not None:
        if max_comp <= policy.max_local_cluster:
            return EscalationDecision(False, "small_clusters")
        return EscalationDecision(True, "too_many_defects")
    if not clusters:
        return EscalationDecision(True, "too_many_defects")
    for c in clusters:
        if c.size > policy.max_local_cluster:
            return EscalationDecision(True, "large_cluster")
    return EscalationDecision(False, "small_clusters")


def adapt_policy(policy: EscalationPolicy, escalate_rate: float) -> None:
    """Nudge local thresholds toward a target escalate rate."""
    if not policy.tune:
        return
    if escalate_rate > policy.target_escalate_rate + 0.08:
        # Escalating too often → allow slightly larger local clusters / defects.
        policy.max_local_defects = min(16, policy.max_local_defects + policy.tune_step)
        if policy.max_local_defects % 2 == 0:
            policy.max_local_cluster = min(4, policy.max_local_cluster + 1)
        # Loosen density slightly at low-p sparse workloads.
        policy.max_local_density = min(0.08, policy.max_local_density + 0.005)
    elif escalate_rate < policy.target_escalate_rate - 0.12:
        # Local too often (risk LER / stage) → tighten.
        policy.max_local_defects = max(2, policy.max_local_defects - policy.tune_step)
        policy.max_local_cluster = max(1, min(policy.max_local_cluster, policy.max_local_defects))
