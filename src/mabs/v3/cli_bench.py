"""MABS competitive benchmark CLI (v3/v4 harness).

Supports ``--warmup`` and ``--fixed-order`` (and ``--no-shuffle``) for the
paper-benchmark harness: per-cell method shuffle, N_disagree vs batch, and
honest CASCADE write_v4_summary (ExactPatternCache; no stage-speedup claim).

Implementation is assembled from ``_cli_part_*`` modules to keep GitHub MCP
push payloads small; behavior matches the monolithic local harness.
"""
from __future__ import annotations

from mabs.v3._cli_part_0 import PART as _p0
from mabs.v3._cli_part_1 import PART as _p1
from mabs.v3._cli_part_2 import PART as _p2
from mabs.v3._cli_part_3 import PART as _p3
from mabs.v3._cli_part_4 import PART as _p4
from mabs.v3._cli_part_5 import PART as _p5

_SRC = "".join([_p0, _p1, _p2, _p3, _p4, _p5])
_g = globals()
exec(compile(_SRC, __file__, "exec"), _g)

if __name__ == "__main__":
    raise SystemExit(main())
