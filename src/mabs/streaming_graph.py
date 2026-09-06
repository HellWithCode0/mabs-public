"""DEM / matcher / window helpers for sliding-window streaming.

Re-exports truncated-window DEM implementation from ``streaming_dem`` /
``streaming_windows``.
"""

from __future__ import annotations

from mabs.streaming_dem import (  # noqa: F401
    BIG_LAYER, CircuitBundle, DemComponent, WindowMatcher, WindowRecord, WindowSpec,
    _build_edge_faults, _layer_boundaries, _obs_mask, build_window_dem,
    make_window_spec, parse_dem_components, plan_windows,
)
from mabs.streaming_windows import (  # noqa: F401
    _commit_edges_to_prediction, _earliest_layer, _edges_and_weight,
    build_surface_code_bundle, build_window_matcher, commit_window_edges,
    ensure_window_schedule, get_window_matcher, mask_to_obs_array, window_match_edges,
)

__all__ = [
    "WindowRecord", "CircuitBundle", "DemComponent", "WindowSpec", "WindowMatcher",
    "build_surface_code_bundle", "parse_dem_components", "plan_windows",
    "make_window_spec", "build_window_dem", "build_window_matcher",
    "get_window_matcher", "ensure_window_schedule", "commit_window_edges",
    "window_match_edges", "mask_to_obs_array", "_build_edge_faults",
    "_commit_edges_to_prediction", "_earliest_layer", "_edges_and_weight",
]
