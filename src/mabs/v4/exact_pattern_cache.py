"""Exact syndrome-pattern cache (canonical module name).

Re-exports from ``iso_cache`` where the implementation currently lives
for a smoother rename; prefer importing ``ExactPatternCache`` from here.
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
]
