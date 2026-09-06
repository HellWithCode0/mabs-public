# MABS v2 — Mixture-Aware Adaptive Boundary Streaming

**MABS-Adaptive** is a streaming quantum error-correction (QEC) control /
measurement algorithm: correct sliding-window MWPM on Stim rotated surface-code
memory, with mixture-aware confidence and ADaPT-style adaptive window sizing.

Matching is delegated to [PyMatching](https://github.com/oscarhiggott/PyMatching)
on [Stim](https://github.com/quantumlib/Stim) circuits. MABS is **not** a new
matching kernel — it is a streaming *policy* + measurement harness designed to
close the LER gap to batch MWPM while reducing mean stage time vs a fixed
`w=3d` window.

## Algorithm (precise)

### 1. Correct streaming baseline (residual overlapping recovery)

1. Build `surface_code:rotated_memory_z` with `R = 10·d` rounds and circuit-level
   noise `p`; construct the DEM and a **full-graph** PyMatching matcher.
2. Partition detectors into time layers from Stim detector coordinates.
3. Maintain a **residual syndrome**. For each window `[t, t+w)` with commit
   stride `C` (defaults `w=3d`, `C=d` ⇒ buffer `≈2d`):
   - Mask residual bits to the active window and run MWPM
     (`decode_to_edges_array` + `return_weight`).
   - **Commit** edges whose earliest finite endpoint lies in `[t, t+C)`:
     XOR edge `fault_ids` into the predicted observables; clear both endpoints
     in the residual; clear the commit-region residual.
   - Advance by `C`. The final window commits through the last layer.
4. Logical error ⇔ predicted observables ≠ sampled observable flips.

**Approximation (honest):** matching uses the full DEM graph with a masked /
residual syndrome, not a truncated per-window DEM. Empirically, for the
proportional policy (`w≈3d`, `C=d`) — and typically also `w=2d` — LER matches
batch MWPM within sampling noise at `p~10⁻³` (see `results/`).

### 2. Mixture-aware confidence `Q(window)`

Cheap signals (Katoch workload + soft info):

| signal | meaning |
|--------|---------|
| `K` | matched-edge count |
| empty-output | `K==0` / nothing committed |
| matching weight | PyMatching MWPM weight |
| syndrome density | nonzero fraction in the active window |

`Q ∈ [0,1]` is a weighted soft score; high `Q` ⇒ easy / sparse window.

### 3. Adaptive window policy (ADaPT-like)

- Prefer **small window** `w_small=2d` when pre-decode density confidence and
  post-decode `Q` are high.
- **Re-decode** with `w_large=3d` when `Q` is low (retry budget 1).
- Stage timers accumulate across retries (honest cost).
- Track retry rate; optionally nudge the `Q` threshold toward a target retry rate.
- Nested timers `τ_input / τ_match / τ_post / τ_stage` and offered-load `ρ`
  reporting retained from v1.

### 4. Kernel auto-tuning

Post-matching kernels: `naive` / `loop` / `vector` / `auto`. Auto dispatches on
`K` vs threshold; `calibrate_auto_threshold` suggests a threshold from the
observed `K` distribution.

## How this differs from ADaPT / parallel-window

| | ADaPT (typ.) | Parallel window | **MABS-Adaptive** |
|--|--------------|-----------------|-------------------|
| Goal | adaptive decoder effort | parallel spacetime windows | streaming policy + mixture timing |
| Inner decoder | various | any | PyMatching MWPM |
| Adaptivity | retry / escalate decode | fixed window geometry | mixture-aware `Q` → `w=2d`↔`3d` |
| Accounting | — | throughput | nested `τ_*`, `R_mean`, `ρ`, `π` |

## Install

```bash
git clone https://github.com/HellWithCode0/mabs.git
cd mabs
pip install -e .
# or: pip install -e ".[dev]"
```

Requires Python ≥ 3.10. Dependencies: `stim`, `pymatching`, `numpy`, `pytest`,
`matplotlib`.

## Run

```bash
# Fast demo (R_mean, π, LER, retry rate)
python -m mabs

# Competitive benchmark table (+ writes results/)
python -m mabs --benchmark
python -m mabs.benchmark --quick          # smaller shot budgets
python -m mabs.benchmark --distances 3,5 --noise 0.001
```

## Tests

```bash
pytest -q
```

## Reproduce benchmarks

```bash
pip install -e .
python -m mabs.benchmark                 # default d∈{3,5,7}, p=1e-3
# outputs:
#   results/benchmark.csv
#   results/SUMMARY.md
#   results/benchmark_ler_timing.png
```

Shot budgets (default): d=3 → 4000, d=5 → 2000, d=7 → 800. These give
order-of-magnitude credible LERs, not publication-grade threshold CIs.

## Package layout

```
src/mabs/
  streaming_graph.py  # DEM/matcher/commit helpers
  streaming.py     # correct residual sliding-window MWPM + LER
  confidence.py    # Q(window) mixture-aware confidence
  adaptive.py      # MABS-Adaptive policy + retry accounting
  baselines.py     # batch / fixed-w / adaptive runners
  benchmark.py     # comparison harness (CLI)
  kernels.py       # naive / loop / vector / auto commit kernels
  timing.py        # NestedTimers
  mixture.py       # π, μ₊, μ₀, Δ decomposition
  reporting.py     # R_q, OLS, offered load, S_meas
  algorithm.py     # run_mabs()
  config.py        # MABSConfig
results/           # committed CSV + SUMMARY.md from benchmarks
```

## Claims (evidence-based only)

See `results/SUMMARY.md` for the latest numbers. Typical intended claim shape
(verify against your run):

- Fixed `w=3d,C=d` streaming **closes the LER gap to batch** MWPM (agreement
  within sampling noise).
- **MABS-Adaptive** keeps LER near batch / `w=3d` while **reducing mean stage
  time** vs fixed `w=3d` on at least some `(d,p)` cells (especially larger `d`),
  by decoding most windows at `w=2d` and retrying rarely.

**Do not claim:** “new industry norm”, or beating Sparse Blossom absolute
µs/round, without hardware-matched timing data.

## Remaining gaps to true SOTA

- No GPU / FPGA / Sparse-Blossom µs-per-round bakeoff.
- No truncated per-window DEM with rough time-boundaries (parallel-window
  graph surgery).
- No correlated matching / belief-matching / neural decoder inners.
- No lattice-surgery / multi-patch streaming workloads.
- Benchmarks are single-process CPython + PyMatching, not a control-stack demo.

## License

MIT
