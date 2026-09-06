"""CASCADE — Cached Approximate Sparse Correction with Amortized Deferred Escalation.

MABS v4.1 streaming decoder (Aryaman Katoch priority list).

Route (cost-aware)::

    empty → ExactPatternCache (CommitAction) → pair-LUT (K=2)
    → clique → peel easy / blossom residual → blossom

Defaults keep LER = batch. ``defer_hard`` removed.
``adaptive_depth`` / ``mabs_v4_adapt`` is research opt-in (default off).

Implementation split: ``cascade_core`` (config/helpers) + ``cascade_stream`` (router).
"""

from mabs.v4.cascade_core import (
    CASCADEConfig,
    CASCADEState,
    prewarm_cascade_graphs,
)
from mabs.v4.cascade_stream import (
    stream_shot_cascade,
    stream_shot_cascade_adapt,
    stream_shot_cascade_timed,
)

__all__ = [
    "CASCADEConfig",
    "CASCADEState",
    "prewarm_cascade_graphs",
    "stream_shot_cascade",
    "stream_shot_cascade_timed",
    "stream_shot_cascade_adapt",
]
