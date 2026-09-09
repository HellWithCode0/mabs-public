"""Console encoding guard shared by the CLI entry points.

MABS prints Greek and maths symbols (pi, Delta, mu, le). On Windows the default
console code page is cp1252, so those characters raise UnicodeEncodeError and
kill the run after the work is already done. Reconfiguring to UTF-8 with
``errors="replace"`` keeps every entry point usable on the acquisition host.
"""

from __future__ import annotations

import sys


def force_utf8_stdio() -> None:
    """Switch stdout/stderr to UTF-8 with replacement. Safe to call repeatedly."""
    for stream in (sys.stdout, sys.stderr):
        reconfigure = getattr(stream, "reconfigure", None)
        if reconfigure is None:
            continue
        try:
            reconfigure(encoding="utf-8", errors="replace")
        except (ValueError, OSError):  # pragma: no cover - detached stream
            pass


__all__ = ["force_utf8_stdio"]
