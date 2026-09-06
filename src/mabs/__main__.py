"""CLI: python -m mabs — demo / benchmark entrypoint."""

from __future__ import annotations

import json
import sys


def main(argv: list[str] | None = None) -> int:
    argv = list(sys.argv[1:] if argv is None else argv)

    if argv and argv[0] in ("--benchmark", "benchmark"):
        from mabs.benchmark import main as bench_main

        return bench_main(argv[1:])

    from mabs.config import MABSConfig
    from mabs.algorithm import run_mabs

    cfg = MABSConfig(
        distances=(3, 5),
        noise=0.001,
        shots=8,
        campaigns=1,
        kernel="auto",
        auto_threshold=64,
        seed=42,
        adaptive=True,
        track_ler=True,
    )
    i = 0
    while i < len(argv):
        if argv[i] == "--shots" and i + 1 < len(argv):
            cfg.shots = int(argv[i + 1])
            i += 2
        elif argv[i] == "--distances" and i + 1 < len(argv):
            cfg.distances = tuple(int(x) for x in argv[i + 1].split(","))
            i += 2
        elif argv[i] == "--kernel" and i + 1 < len(argv):
            cfg.kernel = argv[i + 1]
            i += 2
        elif argv[i] == "--no-adaptive":
            cfg.adaptive = False
            i += 1
        elif argv[i] in ("-h", "--help"):
            print(
                "Usage: python -m mabs [--shots N] [--distances 3,5] "
                "[--kernel auto] [--no-adaptive]\n"
                "       python -m mabs --benchmark [--quick] ..."
            )
            return 0
        else:
            print(f"Unknown arg: {argv[i]}", file=sys.stderr)
            return 2

    print("MABS v2 — Mixture-Aware Adaptive Boundary Streaming")
    print(
        f"  distances={list(cfg.distances)} shots={cfg.shots} "
        f"campaigns={cfg.campaigns} kernel={cfg.kernel} p={cfg.noise} "
        f"adaptive={cfg.adaptive}"
    )
    result = run_mabs(cfg)
    by_d = result.by_distance
    print("\nPer-distance summary:")
    for d in cfg.distances:
        d = int(d)
        info = by_d.get(d)
        if not info:
            print(f"  d={d}: (no windows)")
            continue
        ratios = info["ratios"]
        pi = info["pi"]
        ler = info.get("ler", result.ler_by_distance.get(d))
        ler_s = f"{ler:.4g}" if ler is not None else "n/a"
        print(
            f"  d={d}: R_mean={ratios['R_mean']:.4f}  "
            f"R_median={ratios['R_median']:.4f}  "
            f"π={pi:.4f}  LER={ler_s}  "
            f"n_windows={info['n_windows']}"
        )
    print("\nAdaptive:")
    print(json.dumps(result.summary.get("adaptive", {}), indent=2, default=str))
    print("\nS_meas:")
    print(json.dumps(result.S_meas, indent=2, default=str))
    if result.summary.get("deltas"):
        print("\nΔ mixture decomposition:")
        for delta in result.summary["deltas"]:
            print(
                f"  d{delta['d_a']}→d{delta['d_b']}: "
                f"ΔE={delta['delta_E']:.3g}  "
                f"Δπ={delta['delta_pi']:.4f}"
            )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
