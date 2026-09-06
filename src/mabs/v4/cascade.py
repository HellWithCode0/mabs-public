"""CASCADE — Cached Approximate Sparse Correction with Amortized Deferred Escalation.

MABS v4.2 streaming decoder (Aryaman Katoch).

**FLASH** (default)::

    empty → K=1 boundary CommitAction LUT → K=2 pair CommitAction LUT
    → K>=3 immediate blossom; sticky blossom opt-in (off by default — stage Pareto)

**FULL** (opt-in ``mode="full"`` / ``use_flash=False``)::

    empty → ExactPatternCache → pair-LUT → clique → peel/residual → blossom

``defer_hard`` removed. ``adaptive_depth`` / ``mabs_v4_adapt`` research opt-in.
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
