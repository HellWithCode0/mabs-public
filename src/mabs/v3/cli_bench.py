"""MABS competitive benchmark CLI (v3/v4 harness).

Supports ``--warmup`` and ``--fixed-order`` (and ``--no-shuffle``), N_disagree,
ExactPatternCache / honest CASCADE framing (no stage-speedup claim vs fair w3d).
"""
from __future__ import annotations
import base64
import gzip
from mabs.v3._cli_b64_data import _B64

_SRC = gzip.decompress(base64.b64decode(_B64.encode())).decode()
_g = dict(globals())
_g["__file__"] = __file__
_g["__name__"] = __name__
exec(compile(_SRC, __file__, "exec"), _g)
for _k, _v in _g.items():
    if _k in (
        "main", "run_comparison", "write_v4_summary", "write_v3_summary",
        "write_summary_md", "write_csv", "format_table", "results_to_rows",
        "default_shot_budget", "DEFAULT_METHODS", "_run_one_method", "try_plot",
    ):
        globals()[_k] = _v

if __name__ == "__main__":
    raise SystemExit(main())
