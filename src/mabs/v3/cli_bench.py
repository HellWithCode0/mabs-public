"""MABS competitive benchmark CLI (v3/v4 harness)."""
from __future__ import annotations
import argparse, csv, random
from pathlib import Path
from typing import Any, Optional, Sequence
from mabs.adaptive import AdaptiveConfig
from mabs.baselines import (
    BaselineResult,
    attach_disagree,
    prepare_samples,
    run_adaptive_baseline,
    run_batch_baseline,
    run_fixed_streaming_baseline,
)
from mabs.v3.baseline_runner import run_slem_baseline
from mabs.v4.baseline_runner import run_cascade_baseline
from mabs.v4.cascade import CASCADEConfig
from mabs.confidence import ConfidenceConfig
from mabs.streaming import build_surface_code_bundle
from mabs.v3.slem import SLEMConfig

DEFAULT_METHODS = ("batch", "stream_w3d", "stream_w2d", "mabs_adaptive", "mabs_v3", "mabs_v4")

def default_shot_budget(d: int, quick: bool = False) -> int:
    if quick:
        return {3: 200, 5: 100, 7: 50}.get(d, 50)
    return {3: 4000, 5: 2000, 7: 800}.get(d, 500)


def _run_one_method(name, bundle, syndromes, observables, *, adaptive, slem_cfg, warmup):
    if name == "batch":
        return run_batch_baseline(bundle, syndromes, observables, warmup=warmup)
    if name == "stream_w3d":
        return run_fixed_streaming_baseline(
            bundle, syndromes, observables, window_factor=3.0, method_name="stream_w3d", warmup=warmup
        )
    if name == "stream_w2d":
        return run_fixed_streaming_baseline(
            bundle, syndromes, observables, window_factor=2.0, method_name="stream_w2d", warmup=warmup
        )
    if name == "mabs_adaptive":
        return run_adaptive_baseline(bundle, syndromes, observables, adaptive=adaptive, warmup=warmup)
    if name in ("mabs_v3", "slem"):
        return run_slem_baseline(bundle, syndromes, observables, config=slem_cfg, method_name="mabs_v3", warmup=warmup)
    if name in ("mabs_v4", "cascade"):
        return run_cascade_baseline(
            bundle, syndromes, observables, config=CASCADEConfig(), method_name="mabs_v4", warmup=warmup
        )
    raise ValueError(f"unknown method: {name}")


def run_comparison(
    *,
    distances=(3, 5, 7),
    noise=1e-3,
    noises=None,
    rounds_factor=10,
    seed=42,
    quick=False,
    shot_overrides=None,
    methods=None,
    shuffle_methods: bool = True,
    fixed_order: bool = False,
    warmup: int = 5,
):
    """Run methods on shared syndromes; optional per-cell method shuffle + warmup.

    ``fixed_order=True`` (or ``shuffle_methods=False``) keeps CLI method order.
    """
    if noises is None:
        noises = (noise,)
    method_set = tuple(methods) if methods else DEFAULT_METHODS
    results = []
    adaptive = AdaptiveConfig(
        confidence=ConfidenceConfig(threshold=0.40), tune_threshold=True, target_retry_rate=0.05
    )
    slem_cfg = SLEMConfig()
    do_shuffle = bool(shuffle_methods) and (not fixed_order)
    for p in noises:
        for d in distances:
            d = int(d)
            shots = (shot_overrides or {}).get(d, default_shot_budget(d, quick=quick))
            bundle = build_surface_code_bundle(d=d, noise=float(p), rounds=rounds_factor * d)
            syndromes, observables = prepare_samples(
                bundle, shots=shots, seed=int(seed + 17 * d + int(p * 1e6))
            )
            # Always decode batch first for reference preds (may also be in method list)
            batch_res = run_batch_baseline(bundle, syndromes, observables, warmup=warmup)
            ref_preds = batch_res.extra.get("_preds") if batch_res.extra else None

            ordered = list(method_set)
            if do_shuffle:
                rng = random.Random(int(seed + 101 * d + int(p * 1e6)))
                # Keep batch first for reference attachment convenience; shuffle the rest
                others = [m for m in ordered if m != "batch"]
                rng.shuffle(others)
                ordered = (["batch"] if "batch" in ordered else []) + others

            warm = int(warmup)
            cell_results = []
            for name in ordered:
                if name == "batch":
                    # Reuse already-timed batch result
                    cell_results.append(batch_res)
                    continue
                cell_results.append(
                    _run_one_method(
                        name, bundle, syndromes, observables,
                        adaptive=adaptive, slem_cfg=slem_cfg, warmup=warm,
                    )
                )
            for r in cell_results:
                attach_disagree(r, ref_preds)
                results.append(r)
    return results

def results_to_rows(results):
    return [r.as_dict() for r in results]

def write_csv(rows, path: Path):
    path.parent.mkdir(parents=True, exist_ok=True)
    if not rows:
        path.write_text("")
        return
    keys = list(rows[0].keys())
    for r in rows:
        for k in r:
            if k not in keys:
                keys.append(k)
    with path.open("w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=keys)
        w.writeheader()
        for r in rows:
            w.writerow(r)

def format_table(results):
    header = (
        f"{'method':16s} {'d':>3s} {'p':>8s} {'shots':>6s} {'LER':>10s} {'err':>5s} "
        f"{'stage_ns':>12s} {'shot_ns':>12s} {'retry/esc':>9s} {'N_dis':>6s}"
    )
    lines = [header, "-" * len(header)]
    for r in results:
        esc = r.retry_rate
        if r.extra and "escalate_rate" in r.extra:
            esc = float(r.extra["escalate_rate"])
        nd = r.extra.get("n_disagree", "") if r.extra else ""
        lines.append(
            f"{r.method:16s} {r.d:3d} {r.noise:8.1e} {r.shots:6d} {r.ler:10.6f} "
            f"{r.logical_errors:5d} {r.mean_stage_ns:12.1f} {r.mean_shot_ns:12.1f} "
            f"{esc:9.3f} {str(nd):>6s}"
        )
    return "\n".join(lines)

def write_summary_md(results, path: Path):
    path.parent.mkdir(parents=True, exist_ok=True)
    lines = [
        "# MABS benchmark summary",
        "",
        "| method | d | p | shots | LER | errors | mean_stage_ns | mean_shot_ns | escalate/retry | N_disagree |",
        "|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|",
    ]
    for r in results:
        esc = r.retry_rate
        if r.extra and "escalate_rate" in r.extra:
            esc = float(r.extra["escalate_rate"])
        nd = r.extra.get("n_disagree", "") if r.extra else ""
        lines.append(
            f"| {r.method} | {r.d} | {r.noise:g} | {r.shots} | {r.ler:.6g} | {r.logical_errors} | "
            f"{r.mean_stage_ns:.1f} | {r.mean_shot_ns:.1f} | {esc:.3f} | {nd} |"
        )
    lines += [
        "",
        "## Method notes",
        "",
        "- **mabs_v3 / SLEM**: empty skip + local 1–2 + blossom escalate.",
        "- **mabs_v4 / CASCADE**: ExactPatternCache + clique MWPM (K≤6) + blossom escalate.",
        "- **stream_w3d**: single `decode_to_edges_array` per window (fair baseline; no double blossom).",
        "",
    ]
    path.write_text("\n".join(lines))

def write_v3_summary(results, path: Path):
    path.parent.mkdir(parents=True, exist_ok=True)
    lines = [
        "# MABS v3.1 (SLEM) results summary",
        "",
        "## Honest goal",
        "",
        "Not beat Higgott–Gidney absolute µs in pure Python. Pareto-dominate full PyMatching",
        "Sparse Blossom on mean stage time with LER matching batch MWPM.",
        "",
        "**Fair baseline note (4.0.1a2):** `stream_w3d` previously double-decoded",
        "(`decode(return_weight=True)` + `decode_to_edges_array`). It now uses a single",
        "`decode_to_edges_array` call — same as SLEM/CASCADE hard path — so stage_ns",
        "comparisons are honest.",
        "",
        "| method | d | p | shots | LER | mean_stage_ns | escalate | N_disagree |",
        "|---|---:|---:|---:|---:|---:|---:|---:|",
    ]
    for r in results:
        if r.method not in ("batch", "stream_w3d", "mabs_v3", "mabs_v4", "mabs_adaptive"):
            continue
        esc = ""
        if r.extra and "escalate_rate" in r.extra:
            esc = f"{float(r.extra['escalate_rate']):.3f}"
        elif r.method != "batch":
            esc = f"{r.retry_rate:.3f}"
        nd = r.extra.get("n_disagree", "") if r.extra else ""
        lines.append(
            f"| {r.method} | {r.d} | {r.noise:g} | {r.shots} | {r.ler:.6g} | {r.mean_stage_ns:.1f} | {esc} | {nd} |"
        )
    lines += [
        "",
        "## Before / after vs fair stream_w3d",
        "",
        "| d | p | stream_w3d | mabs_v3 | speedup | escalate | LER v3 | LER batch |",
        "|---:|---:|---:|---:|---:|---:|---:|---:|",
    ]
    by = {}
    for r in results:
        by.setdefault((r.d, r.noise), {})[r.method] = r
    for (d, p), m in sorted(by.items()):
        w3, v3, batch = m.get("stream_w3d"), m.get("mabs_v3"), m.get("batch")
        if not (w3 and v3 and batch):
            continue
        sp = w3.mean_stage_ns / v3.mean_stage_ns if v3.mean_stage_ns else float("nan")
        esc = v3.extra.get("escalate_rate", float("nan")) if v3.extra else float("nan")
        lines.append(
            f"| {d} | {p:g} | {w3.mean_stage_ns:.1f} | {v3.mean_stage_ns:.1f} | {sp:.2f}× | "
            f"{esc:.3f} | {v3.ler:.6g} | {batch.ler:.6g} |"
        )
    lines += [
        "",
        "## Claims / limitations",
        "",
        "- Default: LER=batch, stage vs **fair** single-blossom w3d; escalate improved via exact 2-defect local.",
        "- Low-escalate mode (use_cluster_local=True) can reach esc~0.17 at d=5 but slows stage.",
        "- Not claiming C++ Sparse Blossom absolute µs/round.",
        "",
    ]
    path.write_text("\n".join(lines))


def write_v4_summary(results, path: Path):
    path.parent.mkdir(parents=True, exist_ok=True)
    lines = [
        "# MABS v4 (CASCADE) results summary",
        "",
        "## Honest goal",
        "",
        "Cut blossom escalate vs v3.1 via ExactPatternCache + exact clique MWPM (K≤6),",
        "keeping LER = batch and stage competitive with **fair** stream_w3d / v3.",
        "",
        "## Fair streaming baseline (4.0.1a2)",
        "",
        "Prior `stream_w3d` ran **two** PyMatching blossom calls per window",
        "(`decode(..., return_weight=True)` then `decode_to_edges_array`). SLEM/CASCADE",
        "hard path only runs one `decode_to_edges_array`. That inflated w3d stage_ns",
        "and made speedups look better than they were (esp. when escalate≈1).",
        "",
        "**Fix:** `window_match_edges` / `_edges_and_weight` now call only",
        "`decode_to_edges_array` (weight returned as nan; unused for timing).",
        "When escalate≈1, CASCADE stage should be ~parity with fair w3d (one blossom each).",
        "",
        "| method | d | p | shots | LER | mean_stage_ns | escalate | cache_hit | clique | N_disagree |",
        "|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|",
    ]
    for r in results:
        if r.method not in ("batch", "stream_w3d", "mabs_v3", "mabs_v4"):
            continue
        esc = ""
        ch = ""
        cl = ""
        if r.extra:
            if "escalate_rate" in r.extra:
                esc = f"{float(r.extra['escalate_rate']):.3f}"
            if "cache_hit_rate" in r.extra:
                ch = f"{float(r.extra['cache_hit_rate']):.3f}"
            if "clique_rate" in r.extra:
                cl = f"{float(r.extra['clique_rate']):.3f}"
        elif r.method != "batch":
            esc = f"{r.retry_rate:.3f}"
        nd = r.extra.get("n_disagree", "") if r.extra else ""
        lines.append(
            f"| {r.method} | {r.d} | {r.noise:g} | {r.shots} | {r.ler:.6g} | "
            f"{r.mean_stage_ns:.1f} | {esc} | {ch} | {cl} | {nd} |"
        )
    lines += [
        "",
        "## Before / after vs fair w3d / v3",
        "",
        "| d | p | stream_w3d | mabs_v3 (esc) | mabs_v4 (esc) | v4 cache_hit | v4 clique | LER v4 | LER batch |",
        "|---:|---:|---:|---:|---:|---:|---:|---:|---:|",
    ]
    by = {}
    for r in results:
        by.setdefault((r.d, r.noise), {})[r.method] = r
    for (d, p), m in sorted(by.items()):
        w3, v3, v4, batch = m.get("stream_w3d"), m.get("mabs_v3"), m.get("mabs_v4"), m.get("batch")
        if not (w3 and v4 and batch):
            continue
        v3s = (
            f"{v3.mean_stage_ns:.1f} ({float(v3.extra.get('escalate_rate', float('nan'))):.3f})"
            if v3 and v3.extra
            else (f"{v3.mean_stage_ns:.1f}" if v3 else "—")
        )
        esc4 = float(v4.extra.get("escalate_rate", float("nan"))) if v4.extra else float("nan")
        ch4 = float(v4.extra.get("cache_hit_rate", float("nan"))) if v4.extra else float("nan")
        cl4 = float(v4.extra.get("clique_rate", float("nan"))) if v4.extra else float("nan")
        lines.append(
            f"| {d} | {p:g} | {w3.mean_stage_ns:.1f} | {v3s} | {v4.mean_stage_ns:.1f} ({esc4:.3f}) | "
            f"{ch4:.3f} | {cl4:.3f} | {v4.ler:.6g} | {batch.ler:.6g} |"
        )
    lines += [
        "",
        "## Claims / limitations",
        "",
        "- Default CASCADE: LER=batch via truncated DEM + carry XOR; hard path = per-window blossom.",
        "- Escalate cut by ExactPatternCache + clique (default K≤2); d=7 may still escalate≈1.",
        "- CASCADE reduces blossom invocation frequency, but pure-Python shortcut/control",
        "  overhead can outweigh those savings in elapsed stage time (e.g. d=5/1e-3:",
        "  ~28µs vs fair w3d ~15µs). At d=7 escalate≈1 → ~parity with fair blossom.",
        "- **Not claimed:** stage speedup vs fair stream_w3d.",
        "- `defer_hard=True` is experimental (falls back to per-window blossom when prefer_correctness).",
        "- Not claiming C++ Sparse Blossom absolute µs/round.",
        "- Cache is **ExactPatternCache** (exact memoization — not translation-isomorphism).",
        "",
        "## Version",
        "",
        "**4.0.1a2** — fair single-blossom stream_w3d; ExactPatternCache; N_disagree + warmup/shuffle harness; no stage-speedup claim vs fair w3d.",
        "",
    ]
    path.write_text("\n".join(lines))


def try_plot(results, out_dir: Path):
    try:
        import matplotlib

        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
    except Exception:
        return None
    noises = sorted({r.noise for r in results})
    if not noises:
        return None
    p0 = noises[0]
    methods = ["batch", "stream_w3d", "stream_w2d", "mabs_adaptive", "mabs_v3", "mabs_v4"]
    fig, axes = plt.subplots(1, 2, figsize=(10, 4))
    for method in methods:
        xs, ys = [], []
        for r in results:
            if r.method == method and r.noise == p0:
                xs.append(r.d)
                ys.append(max(r.ler, 1e-6))
        if xs:
            axes[0].semilogy(xs, ys, marker="o", label=method)
    axes[0].set_xlabel("d")
    axes[0].set_ylabel("LER")
    axes[0].set_title(f"LER vs d (p={p0:g})")
    axes[0].legend(fontsize=8)
    axes[0].grid(True, which="both", alpha=0.3)
    for method in methods:
        if method == "batch":
            continue
        xs, ys = [], []
        for r in results:
            if r.method == method and r.noise == p0:
                xs.append(r.d)
                ys.append(r.mean_stage_ns)
        if xs:
            axes[1].plot(xs, ys, marker="o", label=method)
    axes[1].set_xlabel("d")
    axes[1].set_ylabel("mean stage time (ns)")
    axes[1].set_title(f"Stage time vs d (p={p0:g})")
    axes[1].legend(fontsize=8)
    axes[1].grid(True, alpha=0.3)
    fig.tight_layout()
    out = out_dir / "benchmark_ler_timing.png"
    out_dir.mkdir(parents=True, exist_ok=True)
    fig.savefig(out, dpi=120)
    plt.close(fig)
    return out

def main(argv: Optional[Sequence[str]] = None) -> int:
    p = argparse.ArgumentParser(description="MABS competitive benchmark")
    p.add_argument("--quick", action="store_true")
    p.add_argument("--distances", type=str, default="3,5,7")
    p.add_argument("--noise", type=float, default=1e-3)
    p.add_argument("--noise-sweep", type=str, default="")
    p.add_argument("--methods", type=str, default="")
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--out-dir", type=str, default="")
    p.add_argument("--warmup", type=int, default=5, help="Untimed warmup shots before timed aggregation")
    p.add_argument(
        "--fixed-order",
        action="store_true",
        help="Do not shuffle method order per cell (default: shuffle non-batch methods)",
    )
    p.add_argument(
        "--no-shuffle",
        action="store_true",
        help="Alias for --fixed-order",
    )
    args = p.parse_args(list(argv) if argv is not None else None)
    distances = tuple(int(x) for x in args.distances.split(",") if x.strip())
    noises = (
        tuple(float(x) for x in args.noise_sweep.split(",") if x.strip())
        if args.noise_sweep.strip()
        else (args.noise,)
    )
    methods = (
        tuple(x.strip() for x in args.methods.split(",") if x.strip())
        if args.methods.strip()
        else DEFAULT_METHODS
    )
    out_dir = Path(args.out_dir) if args.out_dir else Path(__file__).resolve().parents[3] / "results"
    if not out_dir.parent.exists():
        out_dir = Path("results")
    fixed = bool(args.fixed_order or args.no_shuffle)
    print("MABS benchmark")
    print(
        f"  distances={list(distances)} noises={list(noises)} methods={list(methods)} "
        f"quick={args.quick} warmup={args.warmup} fixed_order={fixed}"
    )
    results = run_comparison(
        distances=distances,
        noises=noises,
        seed=args.seed,
        quick=args.quick,
        methods=methods,
        shuffle_methods=not fixed,
        fixed_order=fixed,
        warmup=args.warmup,
    )
    print()
    print(format_table(results))
    write_csv(results_to_rows(results), out_dir / "benchmark.csv")
    write_summary_md(results, out_dir / "SUMMARY.md")
    write_v3_summary(results, out_dir / "v3_SUMMARY.md")
    write_v4_summary(results, out_dir / "v4_SUMMARY.md")
    plot_path = try_plot(results, out_dir)
    print()
    print(f"Wrote {out_dir / 'benchmark.csv'}")
    print(f"Wrote {out_dir / 'SUMMARY.md'}")
    print(f"Wrote {out_dir / 'v3_SUMMARY.md'}")
    print(f"Wrote {out_dir / 'v4_SUMMARY.md'}")
    if plot_path:
        print(f"Wrote {plot_path}")
    return 0

if __name__ == "__main__":
    raise SystemExit(main())
