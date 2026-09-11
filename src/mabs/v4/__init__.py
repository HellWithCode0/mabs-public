"""MABS v4 — CASCADE (Cached Approximate Sparse Correction with Amortized Deferred Escalation)."""

from mabs.v4.cascade import (
    CASCADEConfig,
    CASCADEState,
    stream_shot_cascade,
    stream_shot_cascade_timed,
    stream_shot_cascade_adapt,
    prewarm_cascade_graphs,
)
from mabs.v4.exact_pattern_cache import ExactPatternCache, IsoCache, canonicalize_defects
from mabs.v4.clique_mwpm import clique_mwpm_edges, clique_decode_window
from mabs.v4.commit_action import CommitAction, edges_to_commit_action, apply_commit_action
from mabs.v4.pair_lut import PairPathLUT, decode_pair_cached
from mabs.v4.flash_lut import FlashCommitLUT, get_flash_lut, extract_defects, has_numba

__all__ = [
    "CASCADEConfig",
    "CASCADEState",
    "stream_shot_cascade",
    "stream_shot_cascade_timed",
    "stream_shot_cascade_adapt",
    "prewarm_cascade_graphs",
    "ExactPatternCache",
    "IsoCache",
    "canonicalize_defects",
    "clique_mwpm_edges",
    "clique_decode_window",
    "CommitAction",
    "edges_to_commit_action",
    "apply_commit_action",
    "PairPathLUT",
    "decode_pair_cached",
    "FlashCommitLUT",
    "get_flash_lut",
    "extract_defects",
    "has_numba",
]

__version__ = "4.3.0a1"
