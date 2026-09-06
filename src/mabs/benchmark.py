"""Thin shim: delegates to mabs.v3.cli_bench (MABS v3 harness)."""
from mabs.v3.cli_bench import *  # noqa: F403
from mabs.v3.cli_bench import main

if __name__ == "__main__":
    raise SystemExit(main())
