"""Mixture-aware confidence score Q(window) for adaptive streaming.

Cheap signals (Katoch workload + MWPM soft info):
  - matched-edge count K
  - empty-output path (K == 0 / no commits)
  - matching weight (PyMatching return_weight)
  - syndrome density in the active window

Q ∈ [0, 1]; higher ⇒ easier window ⇒ prefer smaller buffer / skip retry.

Scales grow with code distance so typical sparse windows at larger d still
score as “easy” (K and weight naturally rise with spacetime volume).
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Optional
import numpy as np


@dataclass
class ConfidenceConfig:
    """Weights and scales for Q(window).

    Absolute scales below are *per unit distance*; multiply by ``d`` at
    score time when ``distance`` is provided.
    """

    # Soft saturations per unit distance (d=1 reference)
    K_scale_per_d: float = 1.5  # K_scale = 1.5 * d
    weight_scale_per_d: float = 8.0
    density_scale: float = 0.05  # density is already normalized
    # Fixed fallbacks when distance unknown
    K_scale: float = 8.0
    weight_scale: float = 40.0
    # Component weights (normalized internally)
    w_K: float = 0.35
    w_empty: float = 0.20
    w_weight: float = 0.30
    w_density: float = 0.15
    # Decision threshold: Q >= threshold → trust small window
    threshold: float = 0.40


def _sat(x: float, scale: float) -> float:
    """Map x>=0 through 1 - exp(-x/scale) ∈ [0,1)."""
    if scale <= 0:
        return 1.0 if x > 0 else 0.0
    return float(1.0 - np.exp(-max(x, 0.0) / scale))


def _scales(cfg: ConfidenceConfig, distance: Optional[int]) -> tuple[float, float]:
    if distance is not None and distance > 0:
        return cfg.K_scale_per_d * distance, cfg.weight_scale_per_d * distance
    return cfg.K_scale, cfg.weight_scale


def confidence_score(
    *,
    K: int,
    empty_output: bool,
    matching_weight: float,
    syndrome_density: float,
    cfg: Optional[ConfidenceConfig] = None,
    distance: Optional[int] = None,
) -> float:
    """Compute mixture-aware confidence Q ∈ [0, 1]."""
    if cfg is None:
        cfg = ConfidenceConfig()
    K_scale, weight_scale = _scales(cfg, distance)
    s_empty = 1.0 if empty_output or K == 0 else 0.0
    s_K = 1.0 - _sat(float(K), K_scale)
    s_w = 1.0 - _sat(float(matching_weight), weight_scale)
    s_d = 1.0 - _sat(float(syndrome_density), cfg.density_scale)
    ws = np.array(
        [cfg.w_K, cfg.w_empty, cfg.w_weight, cfg.w_density], dtype=np.float64
    )
    ws = ws / ws.sum()
    q = float(ws[0] * s_K + ws[1] * s_empty + ws[2] * s_w + ws[3] * s_d)
    return float(min(1.0, max(0.0, q)))


def predecode_confidence(
    syndrome_density: float,
    *,
    cfg: Optional[ConfidenceConfig] = None,
) -> float:
    """Cheap pre-match confidence from syndrome density alone."""
    if cfg is None:
        cfg = ConfidenceConfig()
    return float(1.0 - _sat(float(syndrome_density), cfg.density_scale))


def should_retry(
    *,
    K: int,
    matching_weight: float,
    syndrome_density: float,
    Q: float,
    threshold: float,
    distance: int,
    cfg: Optional[ConfidenceConfig] = None,
) -> bool:
    """Hard+soft retry gate: retry only if Q low *and* strong hardness."""
    if cfg is None:
        cfg = ConfidenceConfig()
    if Q >= threshold:
        return False
    K_scale, weight_scale = _scales(cfg, distance)
    # Stricter than the soft Q scale so retries stay rare (preserve speedup)
    hard = (
        K >= max(3, int(1.25 * K_scale))
        or matching_weight >= 1.25 * weight_scale
        or syndrome_density >= 2.5 * cfg.density_scale
    )
    return bool(hard)


def calibrate_auto_threshold(
    K_samples: np.ndarray,
    *,
    quantile: float = 0.75,
    floor: int = 8,
    ceil: int = 256,
) -> int:
    """Calibrate kernel auto-dispatch threshold from observed K distribution."""
    K_samples = np.asarray(K_samples, dtype=np.float64)
    if K_samples.size == 0:
        return 64
    q = float(np.quantile(K_samples, quantile))
    thr = int(round(q))
    return int(min(ceil, max(floor, thr)))
