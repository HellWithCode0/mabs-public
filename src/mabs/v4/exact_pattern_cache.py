"""Exact syndrome-pattern cache (canonical import: mabs.v4.exact_pattern_cache).

Re-exports from ``iso_cache`` where the ExactPatternCache implementation lives.
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
