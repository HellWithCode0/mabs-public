"""MABS v4 — CASCADE (Cached Approximate Sparse Correction with Amortized Deferred Escalation)."""

from mabs.v4.cascade import (
    CASCADEConfig,
    CASCADEState,
    stream_shot_cascade,
    stream_shot_cascade_timed,
    prewarm_cascade_graphs,
)
from mabs.v4.iso_cache import IsoCache, canonicalize_defects
from mabs.v4.clique_mwpm import clique_mwpm_edges, clique_decode_window

__all__ = [
    "CASCADEConfig",
    "CASCADEState",
    "stream_shot_cascade",
    "stream_shot_cascade_timed",
    "prewarm_cascade_graphs",
    "IsoCache",
    "canonicalize_defects",
    "clique_mwpm_edges",
    "clique_decode_window",
]

__version__ = "4.0.0a1"
