"""Competitive benchmark harness: LER + timing vs baselines.

Usage:
  python -m mabs.benchmark
  python -m mabs --benchmark
"""

from __future__ import annotations

import argparse
import csv
from pathlib import Path
from typing import Any, Optional, Sequence

from mabs.adaptive import AdaptiveConfig
from mabs.baselines import (
    BaselineResult,
    prepare_samples,
    run_adaptive_baseline,
    run_batch_baseline,
    run_fixed_streaming_baseline,
)
from mabs.confidence import ConfidenceConfig
from mabs.streaming import build_surface_code_bundle


def default_shot_budget(d: int, quick: bool = False) -> int:
    if quick:
        return {3: 200, 5: 100, 7: 50}.get(d, 50)
    # Credible order-of-magnitude LER at p~1e-3
    return {3: 4000, 5: 2000, 7: 800}.get(d, 500)


def run_comparison(
    *,
    distances: Sequence[int] = (3, 5, 7),
    noise: float = 1e-3,
    noises: Optional[Sequence[float]] = None,
    rounds_factor: int = 10,
    seed: int = 42,
    quick: bool = False,
    shot_overrides: Optional[dict[int, int]] = None,
) -> list[BaselineResult]:
    """Run batch / fixed-w3d / fixed-w2d / adaptive comparison."""
    if noises is None:
        noises = (noise,)
    results: list[BaselineResult] = []
    adaptive = AdaptiveConfig(
        w_large_factor=3.0,
        w_small_factor=2.0,
        commit_factor=1.0,
        confidence=ConfidenceConfig(threshold=0.40),
        max_retries_per_window=1,
        use_predecode_gate=True,
        predecode_threshold=0.30,
        tune_threshold=True,
        target_retry_rate=0.05,
    )

    for p in noises:
        for d in distances:
            d = int(d)
            shots = (shot_overrides or {}).get(d, default_shot_budget(d, quick=quick))
            rounds = rounds_factor * d
            bundle = build_surface_code_bundle(d=d, noise=float(p), rounds=rounds)
            syndromes, observables = prepare_samples(
                bundle, shots=shots, seed=int(seed + 17 * d + int(p * 1e6))
            )

            results.append(run_batch_baseline(bundle, syndromes, observables))
            results.append(
                run_fixed_streaming_baseline(
                    bundle,
                    syndromes,
                    observables,
                    window_factor=3.0,
                    method_name="stream_w3d",
                )
            )
            results.append(
                run_fixed_streaming_baseline(
                    bundle,
                    syndromes,
                    observables,
                    window_factor=2.0,
                    method_name="stream_w2d",
                )
            )
            results.append(
                run_adaptive_baseline(
                    bundle, syndromes, observables, adaptive=adaptive
                )
            )
    return results


def results_to_rows(results: Sequence[BaselineResult]) -> list[dict[str, Any]]:
    return [r.as_dict() for r in results]


def write_csv(rows: Sequence[dict[str, Any]], path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    if not rows:
        path.write_text("")
        return
    keys = list(rows[0].keys())
    # union keys
    for r in rows:
        for k in r:
            if k not in keys:
                keys.append(k)
    with path.open("w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=keys)
        w.writeheader()
        for r in rows:
            w.writerow(r)


def format_table(results: Sequence[BaselineResult]) -> str:
    header = (
        f"{'method':16s} {'d':>3s} {'p':>8s} {'shots':>6s} "
        f"{'LER':>10s} {'err':>5s} {'stage_ns':>12s} {'shot_ns':>12s} {'retry':>7s}"
    )
    lines = [header, "-" * len(header)]
    for r in results:
        lines.append(
            f"{r.method:16s} {r.d:3d} {r.noise:8.1e} {r.shots:6d} "
            f"{r.ler:10.6f} {r.logical_errors:5d} {r.mean_stage_ns:12.1f} "
            f"{r.mean_shot_ns:12.1f} {r.retry_rate:7.3f}"
        )
    return "\n".join(lines)


def write_summary_md(results: Sequence[BaselineResult], path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    lines = [
        "# MABS v2 benchmark summary",
        "",
        "Comparison of **batch MWPM**, fixed streaming (`w=3d`, `w=2d`, `C=d`),",
        "and **MABS-Adaptive** on Stim rotated surface-code memory",
        "(`surface_code:rotated_memory_z`, `R=10·d`) with PyMatching.",
        "",
        "## Table",
        "",
        "| method | d | p | shots | LER | errors | mean_stage_ns | mean_shot_ns | retry_rate |",
        "|---|---:|---:|---:|---:|---:|---:|---:|---:|",
    ]
    for r in results:
        lines.append(
            f"| {r.method} | {r.d} | {r.noise:g} | {r.shots} | {r.ler:.6g} | "
            f"{r.logical_errors} | {r.mean_stage_ns:.1f} | {r.mean_shot_ns:.1f} | "
            f"{r.retry_rate:.3f} |"
        )

    # Wins / losses narrative
    lines += ["", "## Wins / losses (auto)", ""]
    by_key: dict[tuple[int, float], dict[str, BaselineResult]] = {}
    for r in results:
        by_key.setdefault((r.d, r.noise), {})[r.method] = r
    for (d, p), m in sorted(by_key.items()):
        batch = m.get("batch")
        w3 = m.get("stream_w3d")
        w2 = m.get("stream_w2d")
        ad = m.get("mabs_adaptive")
        if not (batch and w3 and ad):
            continue
        ler_gap = abs(ad.ler - batch.ler)
        speedup = (
            w3.mean_stage_ns / ad.mean_stage_ns
            if ad.mean_stage_ns and ad.mean_stage_ns > 0
            else float("nan")
        )
        lines.append(
            f"- **d={d}, p={p:g}**: adaptive LER={ad.ler:.4g} vs batch={batch.ler:.4g} "
            f"(gap={ler_gap:.4g}); stage speedup vs w=3d ≈ {speedup:.2f}× "
            f"(retry_rate={ad.retry_rate:.3f})."
        )
        if w2:
            lines.append(
                f"  - fixed w=2d LER={w2.ler:.4g}, stage_ns={w2.mean_stage_ns:.1f}."
            )

    lines += [
        "",
        "## Method notes",
        "",
        "- **batch**: full-shot PyMatching on the complete detector history.",
        "- **stream_w3d / stream_w2d**: residual-syndrome sliding windows with",
        "  commit stride `C=d`; observables accumulated from committed matching",
        "  edges (`fault_ids`). Full DEM graph + masked residual (not a truncated",
        "  per-window DEM).",
        "- **mabs_adaptive**: try `w=2d` when mixture-aware confidence `Q` is high;",
        "  re-decode with `w=3d` when `Q` is low (retry budget 1). Stage times",
        "  include retry cost.",
        "",
        "## Limitations",
        "",
        "- Not claiming Sparse Blossom absolute µs/round leadership.",
        "- Campaign sizes give order-of-magnitude credible LERs, not publication",
        "  threshold plots; increase shots for tighter CIs.",
        "- Truncated per-window DEMs / parallel-window seams are future work.",
        "",
    ]
    path.write_text("\n".join(lines))


def try_plot(results: Sequence[BaselineResult], out_dir: Path) -> Optional[Path]:
    try:
        import matplotlib

        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
    except Exception:
        return None

    # LER vs d for each method at first noise
    noises = sorted({r.noise for r in results})
    if not noises:
        return None
    p0 = noises[0]
    methods = ["batch", "stream_w3d", "stream_w2d", "mabs_adaptive"]
    fig, axes = plt.subplots(1, 2, figsize=(10, 4))
    ax = axes[0]
    for method in methods:
        xs, ys = [], []
        for r in results:
            if r.method == method and r.noise == p0:
                xs.append(r.d)
                ys.append(max(r.ler, 1e-6))
        if xs:
            ax.semilogy(xs, ys, marker="o", label=method)
    ax.set_xlabel("d")
    ax.set_ylabel("LER")
    ax.set_title(f"LER vs d (p={p0:g})")
    ax.legend(fontsize=8)
    ax.grid(True, which="both", alpha=0.3)

    ax = axes[1]
    for method in methods:
        if method == "batch":
            continue
        xs, ys = [], []
        for r in results:
            if r.method == method and r.noise == p0:
                xs.append(r.d)
                ys.append(r.mean_stage_ns)
        if xs:
            ax.plot(xs, ys, marker="o", label=method)
    ax.set_xlabel("d")
    ax.set_ylabel("mean stage time (ns)")
    ax.set_title(f"Stage time vs d (p={p0:g})")
    ax.legend(fontsize=8)
    ax.grid(True, alpha=0.3)
    fig.tight_layout()
    out = out_dir / "benchmark_ler_timing.png"
    out_dir.mkdir(parents=True, exist_ok=True)
    fig.savefig(out, dpi=120)
    plt.close(fig)
    return out


def main(argv: Optional[Sequence[str]] = None) -> int:
    p = argparse.ArgumentParser(description="MABS v2 competitive benchmark")
    p.add_argument("--quick", action="store_true", help="smaller shot budgets")
    p.add_argument("--distances", type=str, default="3,5,7")
    p.add_argument("--noise", type=float, default=1e-3)
    p.add_argument(
        "--noise-sweep",
        type=str,
        default="",
        help="comma-separated p values (overrides --noise)",
    )
    p.add_argument("--seed", type=int, default=42)
    p.add_argument(
        "--out-dir",
        type=str,
        default="",
        help="default: <repo>/results",
    )
    args = p.parse_args(list(argv) if argv is not None else None)

    distances = tuple(int(x) for x in args.distances.split(",") if x.strip())
    if args.noise_sweep.strip():
        noises = tuple(float(x) for x in args.noise_sweep.split(",") if x.strip())
    else:
        noises = (args.noise,)

    # Resolve results dir relative to package repo root when possible
    if args.out_dir:
        out_dir = Path(args.out_dir)
    else:
        # src/mabs/benchmark.py -> repo root
        out_dir = Path(__file__).resolve().parents[2] / "results"
        if not out_dir.parent.exists():
            out_dir = Path("results")

    print("MABS v2 benchmark")
    print(f"  distances={list(distances)} noises={list(noises)} quick={args.quick}")
    results = run_comparison(
        distances=distances,
        noises=noises,
        seed=args.seed,
        quick=args.quick,
    )
    table = format_table(results)
    print()
    print(table)

    rows = results_to_rows(results)
    csv_path = out_dir / "benchmark.csv"
    md_path = out_dir / "SUMMARY.md"
    write_csv(rows, csv_path)
    write_summary_md(results, md_path)
    plot_path = try_plot(results, out_dir)
    print()
    print(f"Wrote {csv_path}")
    print(f"Wrote {md_path}")
    if plot_path:
        print(f"Wrote {plot_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
