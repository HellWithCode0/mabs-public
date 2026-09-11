"""MABS competitive benchmark CLI (v3/v4 harness)."""
from __future__ import annotations
import argparse, csv, random
from pathlib import Path
from typing import Any, Optional, Sequence
from mabs._stdio import force_utf8_stdio
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
    if name in ("mabs_v4_flash", "cascade_flash"):
        # FLASH with the CLUSTER route forced off, for side-by-side comparison.
        return run_cascade_baseline(
            bundle, syndromes, observables, config=CASCADEConfig(cluster_route=False),
            method_name="mabs_v4_flash", warmup=warmup,
        )
    if name in ("mabs_v4_cluster", "cascade_cluster"):
        return run_cascade_baseline(
            bundle, syndromes, observables, config=CASCADEConfig(cluster_route=True),
            method_name="mabs_v4_cluster", warmup=warmup,
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
        path.write_text("", encoding="utf-8")
        return
    keys = list(rows[0].keys())
    for r in rows:
        for k in r:
            if k not in keys:
                keys.append(k)
    with path.open("w", newline="", encoding="utf-8") as f:
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
        "- **mabs_v4 / CASCADE FLASH** (default): empty / K=1 boundary CommitAction LUT /",
        "  K=2 pair CommitAction LUT / K≥3 certified CLUSTER route when every cluster is",
        "  a singleton or adjacent pair (on when numba is importable), else blossom.",
        "  `mabs_v4_flash` and `mabs_v4_cluster` force the CLUSTER route off or on. FULL",
        "  mode (`mode=\"full\"`) restores the 4.1 ExactPatternCache + clique + peel path.",
        "- **stream_w3d**: single `decode_to_edges_array` per window (fair baseline; no double blossom).",
        "",
    ]
    path.write_text("\n".join(lines), encoding="utf-8")

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
    path.write_text("\n".join(lines), encoding="utf-8")


def write_v4_summary(results, path: Path):
    path.parent.mkdir(parents=True, exist_ok=True)
    modes = sorted({
        str(r.extra.get("mode", "flash"))
        for r in results
        if r.method.startswith("mabs_v4") and r.extra
    })
    mode_note = ", ".join(modes) if modes else "flash"
    lines = [
        "# MABS v4.3 (CASCADE FLASH + CLUSTER) results summary",
        "",
        f"Generated by `python -m mabs.benchmark`. CASCADE mode: **{mode_note}**.",
        "",
        "## Honest goal",
        "",
        "Make CASCADE shortcuts cheaper than one fair blossom, so mean stage_ns is at",
        "or below fair `stream_w3d`, while keeping LER = batch and N_disagree = 0.",
        "",
        "## FLASH route (default)",
        "",
        "1. One-pass capped defect extract (numba when importable).",
        "2. K=0: empty CommitAction.",
        "3. K=1: boundary CommitAction LUT.",
        "4. K=2: pair CommitAction LUT (prewarm hop-radius 4; miss goes to blossom and fills).",
        "5. K>=3: the CLUSTER route of item 7 when it can certify the window, else",
        "   blossom, committed through `commit_window_edges` like w3d.",
        "6. Sticky blossom is opt-in (`sticky_blossom=True`) and off by default.",
        "7. CLUSTER route (on by default when numba is importable; methods",
        "   `mabs_v4_cluster` and `mabs_v4_flash` force it on or off): a K>=3",
        "   window whose clusters are all singletons or adjacent pairs is answered",
        "   from the LUTs when an LP-duality certificate proves the answer optimal;",
        "   otherwise blossom.",
        "",
        "FULL mode (`mode=\"full\"` / `use_flash=False`) restores the 4.1 research path:",
        "ExactPatternCache, peel, clique and cost routing. The `cache` / `clique` / `peel`",
        "columns below are FULL-mode counters and read 0.000 under the FLASH default.",
        "",
        "## Fair streaming baseline",
        "",
        "`window_match_edges` and `_edges_and_weight` make a single",
        "`decode_to_edges_array` call, the same one blossom the CASCADE hard path makes,",
        "so stage_ns is comparable. Weight is returned as nan and is unused for timing.",
        "",
        "| method | d | p | shots | LER | mean_stage_ns | escalate | empty | pair | cluster | cache | clique | N_disagree |",
        "|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|",
    ]

    def _rate(extra, key):
        if not extra or key not in extra:
            return ""
        return f"{float(extra[key]):.3f}"

    for r in results:
        if r.method not in ("batch", "stream_w3d", "mabs_v3", "mabs_v4", "mabs_v4_flash",
                            "mabs_v4_cluster"):
            continue
        esc = _rate(r.extra, "escalate_rate")
        if not esc and r.method != "batch":
            esc = f"{r.retry_rate:.3f}"
        nd = r.extra.get("n_disagree", "") if r.extra else ""
        lines.append(
            f"| {r.method} | {r.d} | {r.noise:g} | {r.shots} | {r.ler:.6g} | "
            f"{r.mean_stage_ns:.1f} | {esc} | {_rate(r.extra, 'empty_rate')} | "
            f"{_rate(r.extra, 'pair_rate')} | {_rate(r.extra, 'cluster_rate')} | "
            f"{_rate(r.extra, 'cache_rate')} | {_rate(r.extra, 'clique_rate')} | {nd} |"
        )
    def _xrate(r, key):
        if r is None or not r.extra:
            return float("nan")
        return float(r.extra.get(key, float("nan")))

    # Forced-route columns only when those methods actually ran; otherwise the
    # table fills with n/a, which reads as a failure rather than "not run".
    extra_methods = [
        (name, label) for name, label in (
            ("mabs_v4_flash", "mabs_v4_flash (esc)"),
            ("mabs_v4_cluster", "mabs_v4_cluster (esc, cluster)"),
        )
        if any(r.method == name for r in results)
    ]
    header = "| d | p | stream_w3d | mabs_v3 (esc) | mabs_v4 (esc, cluster) | v4/w3d |"
    for _name, label in extra_methods:
        header += f" {label} | /w3d |"
    header += " LER v4 | LER batch |"
    ncols = header.count("|") - 1
    lines += [
        "",
        "## vs fair w3d / v3",
        "",
        "Ratios below 1 mean faster than `stream_w3d`. Cluster is the share of all",
        "windows answered by the certified CLUSTER route.",
        "",
        header,
        "|" + "---:|" * ncols,
    ]
    by = {}
    for r in results:
        by.setdefault((r.d, r.noise), {})[r.method] = r
    ratios = []
    for (d, p), m in sorted(by.items()):
        w3, v3, v4, batch = m.get("stream_w3d"), m.get("mabs_v3"), m.get("mabs_v4"), m.get("batch")
        if not (w3 and v4 and batch):
            continue
        v3s = "n/a"
        if v3 is not None:
            v3s = f"{v3.mean_stage_ns:.1f} ({_xrate(v3, 'escalate_rate'):.3f})"
        rel = v4.mean_stage_ns / w3.mean_stage_ns if w3.mean_stage_ns else float("nan")
        ratios.append(rel)
        row = (
            f"| {d} | {p:g} | {w3.mean_stage_ns:.1f} | {v3s} | "
            f"{v4.mean_stage_ns:.1f} ({_xrate(v4, 'escalate_rate'):.3f}, "
            f"{_xrate(v4, 'cluster_rate'):.3f}) | {rel:.3f} |"
        )
        for name, _label in extra_methods:
            rx = m.get(name)
            if rx is None:
                row += " not run | |"
                continue
            relx = rx.mean_stage_ns / w3.mean_stage_ns if w3.mean_stage_ns else float("nan")
            inner = f"{_xrate(rx, 'escalate_rate'):.3f}"
            if name == "mabs_v4_cluster":
                inner += f", {_xrate(rx, 'cluster_rate'):.3f}"
            row += f" {rx.mean_stage_ns:.1f} ({inner}) | {relx:.3f} |"
        row += f" {v4.ler:.6g} | {batch.ler:.6g} |"
        lines.append(row)

    finite = [x for x in ratios if x == x]
    if finite:
        speed_claim = [
            f"- **Observed on this run:** `mabs_v4` stage / `stream_w3d` stage ranged from "
            f"{min(finite):.2f} to {max(finite):.2f} across the {len(finite)} cells above.",
            "  All methods ran once, in one process, on shared window matchers, so the",
            "  ratios are the comparable number and absolute times are specific to this host.",
            "  Repeat the run before quoting a ratio; one run carries no error bar.",
        ]
    else:
        speed_claim = ["- No cell ran both `mabs_v4` and `stream_w3d`, so no speed comparison."]
    lines += [
        "",
        "## Claims / limitations",
        "",
        "- **Claim:** LER = batch and N_disagree = 0 on the cells in the table above.",
        *speed_claim,
        "- Not claiming C++ Sparse Blossom absolute time per round.",
        "- `defer_hard` was removed in 4.1; a K>=3 window the CLUSTER route cannot",
        "  certify goes straight to blossom.",
        "- The cache in FULL mode is an **ExactPatternCache**, exact memoization, not",
        "  translation isomorphism.",
        "- These are elapsed CPU intervals on this host. They carry no deadline reading;",
        "  that needs the S_arch fields and a residual (see `mabs.reporting.SArch`).",
        "",
        "## Version",
        "",
        "**4.3.0a1** CASCADE FLASH with the certified CLUSTER route (Aryaman Katoch).",
        "",
    ]
    path.write_text("\n".join(lines), encoding="utf-8")


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
    # One fixed colour per method, so a method looks the same in every panel.
    # The first four match matplotlib's default cycle, which earlier plots used.
    colors = {
        "batch": "#1f77b4",
        "stream_w3d": "#ff7f0e",
        "mabs_v3": "#2ca02c",
        "mabs_v4": "#d62728",
        "mabs_v4_flash": "#e377c2",
        "mabs_v4_cluster": "#17becf",
        "stream_w2d": "#8c564b",
        "mabs_adaptive": "#9467bd",
    }
    present = [m for m in colors if any(r.method == m for r in results)]
    # Small horizontal offsets so methods with identical LER stay visible.
    offset = {m: (i - (len(present) - 1) / 2) * 0.04 for i, m in enumerate(present)}
    fig, axes = plt.subplots(len(noises), 2, figsize=(10, 4 * len(noises)), squeeze=False)
    for row, p in enumerate(noises):
        ax_ler, ax_t = axes[row]
        any_zero = False
        for method in present:
            pts = sorted(
                (r.d, r.ler, r.shots) for r in results if r.method == method and r.noise == p
            )
            if not pts:
                continue
            c = colors[method]
            dx = offset[method]
            hit = [(d + dx, ler) for d, ler, _s in pts if ler > 0]
            zero = [(d + dx, 1.0 / s) for d, ler, s in pts if ler <= 0 and s > 0]
            if hit:
                ax_ler.semilogy([x for x, _ in hit], [y for _, y in hit], marker="o",
                                color=c, label=method)
            else:
                ax_ler.plot([], [], marker="o", color=c, label=method)
            if zero:
                any_zero = True
                # No errors observed: 1/shots is an upper bound, not a rate.
                ax_ler.semilogy([x for x, _ in zero], [y for _, y in zero], linestyle="none",
                                marker="v", markerfacecolor="none", markeredgecolor=c,
                                markersize=8)
        ax_ler.set_xlabel("d")
        ax_ler.set_ylabel("LER")
        title = f"LER vs d (p={p:g})"
        if any_zero:
            title += "\nopen triangle: 0 errors, drawn at 1/shots (upper bound)"
        ax_ler.set_title(title, fontsize=10)
        ax_ler.legend(fontsize=8)
        ax_ler.grid(True, which="both", alpha=0.3)
        for method in present:
            if method == "batch":
                continue  # batch decodes a whole shot at once; not a per-window stage
            pts = sorted(
                (r.d, r.mean_stage_ns) for r in results if r.method == method and r.noise == p
            )
            if pts:
                ax_t.plot([d for d, _ in pts], [t / 1e3 for _, t in pts], marker="o",
                          color=colors[method], label=method)
        ax_t.set_xlabel("d")
        ax_t.set_ylabel("mean stage time per window (us)")
        ax_t.set_title(f"Stage time vs d (p={p:g})", fontsize=10)
        ax_t.legend(fontsize=8)
        ax_t.grid(True, alpha=0.3)
    fig.tight_layout()
    out = out_dir / "benchmark_ler_timing.png"
    out_dir.mkdir(parents=True, exist_ok=True)
    fig.savefig(out, dpi=120)
    plt.close(fig)
    return out

def main(argv: Optional[Sequence[str]] = None) -> int:
    force_utf8_stdio()
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
