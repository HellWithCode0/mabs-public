"""Exact syndrome-pattern cache (canonical module name).

Prefer importing ``ExactPatternCache`` from here. Implementation lives in
``iso_cache`` (historical path); this module is the public surface.
"""
from mabs.v4.iso_cache import (  # noqa: F401
    ExactPatternCache,
    IsoCache,
    CacheKey,
    canonicalize_defects,
    relative_key,
    local_to_global,
    edges_local_to_global,
    edges_global_to_local,
    make_topo_id,
    topo_id_from_wm,
    build_pattern_key,
)

__all__ = [
    "ExactPatternCache",
    "IsoCache",
    "CacheKey",
    "canonicalize_defects",
    "relative_key",
    "local_to_global",
    "edges_local_to_global",
    "edges_global_to_local",
    "make_topo_id",
    "topo_id_from_wm",
    "build_pattern_key",
]
