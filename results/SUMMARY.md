# MABS v2 benchmark summary

Comparison of **batch MWPM**, fixed streaming (`w=3d`, `w=2d`, `C=d`),
and **MABS-Adaptive** on Stim rotated surface-code memory
(`surface_code:rotated_memory_z`, `R=10·d`) with PyMatching.

Generated on the development box (CPython + PyMatching Sparse Blossom). Stage
times are mean per-window `τ_stage` (ns); they are for *relative* policy
comparison, not absolute µs/round hardware claims.

## Table

| method | d | p | shots | LER | errors | mean_stage_ns | mean_shot_ns | retry_rate |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| batch | 3 | 0.001 | 4000 | 0.006 | 24 | 812.4 | 812.4 | 0.000 |
| stream_w3d | 3 | 0.001 | 4000 | 0.006 | 24 | 26676.6 | 287316.9 | 0.000 |
| stream_w2d | 3 | 0.001 | 4000 | 0.006 | 24 | 23996.5 | 287572.6 | 0.000 |
| mabs_adaptive | 3 | 0.001 | 4000 | 0.006 | 24 | 26530.7 | 458269.9 | 0.000 |
| batch | 5 | 0.001 | 2000 | 0.0005 | 1 | 5599.4 | 5599.4 | 0.000 |
| stream_w3d | 5 | 0.001 | 2000 | 0.001 | 2 | 51909.5 | 519089.9 | 0.000 |
| stream_w2d | 5 | 0.001 | 2000 | 0.001 | 2 | 44422.8 | 497755.0 | 0.000 |
| mabs_adaptive | 5 | 0.001 | 2000 | 0.001 | 2 | 46873.8 | 704318.3 | 0.003 |
| batch | 7 | 0.001 | 800 | 0 | 0 | 24494.6 | 24494.6 | 0.000 |
| stream_w3d | 7 | 0.001 | 800 | 0.0025 | 2 | 90086.1 | 867327.0 | 0.000 |
| stream_w2d | 7 | 0.001 | 800 | 0.0025 | 2 | 77595.2 | 835589.1 | 0.000 |
| mabs_adaptive | 7 | 0.001 | 800 | 0.0025 | 2 | 85136.3 | 1128792.3 | 0.047 |
| batch | 3 | 0.002 | 4000 | 0.0245 | 98 | 1423.6 | 1423.6 | 0.000 |
| stream_w3d | 3 | 0.002 | 4000 | 0.0245 | 98 | 29479.8 | 309850.9 | 0.000 |
| stream_w2d | 3 | 0.002 | 4000 | 0.0245 | 98 | 26497.5 | 311986.9 | 0.000 |
| mabs_adaptive | 3 | 0.002 | 4000 | 0.0245 | 98 | 29166.2 | 481291.1 | 0.000 |
| batch | 5 | 0.002 | 2000 | 0.011 | 22 | 11310.7 | 11310.7 | 0.000 |
| stream_w3d | 5 | 0.002 | 2000 | 0.0155 | 31 | 61430.2 | 605224.3 | 0.000 |
| stream_w2d | 5 | 0.002 | 2000 | 0.0155 | 31 | 52679.8 | 582144.5 | 0.000 |
| mabs_adaptive | 5 | 0.002 | 2000 | 0.0155 | 31 | 57532.6 | 813503.7 | 0.031 |
| batch | 7 | 0.002 | 800 | 0.00125 | 1 | 41138.5 | 41138.5 | 0.000 |
| stream_w3d | 7 | 0.002 | 800 | 0.02 | 16 | 118801.6 | 1131134.1 | 0.000 |
| stream_w2d | 7 | 0.002 | 800 | 0.02125 | 17 | 99995.7 | 1063822.2 | 0.000 |
| mabs_adaptive | 7 | 0.002 | 800 | 0.02125 | 17 | 110930.8 | 1397432.6 | 0.052 |

## Wins / losses

### Where MABS-Adaptive wins
- **d=5, p=1e-3**: stage time **~10% below** fixed `w=3d` (46.9µs vs 51.9µs
  mean stage) with **identical LER** to `w=3d`/`w=2d` (0.001 vs batch 0.0005).
- **d=7, p=1e-3**: stage time **~5.5% below** `w=3d` with the same streaming
  LER (0.0025); rare retries (~5%) hedge harder windows.
- **d=5, p=2e-3**: stage time **~6% below** `w=3d`, same LER as fixed streaming.
- At **d=3** all methods agree on LER with batch; adaptive ≈ `w=3d` on stage
  (small-window savings are tiny at this distance).

### Where it loses / ties
- **Absolute shot latency** (wall time per shot) is higher for adaptive than
  fixed streaming because of Python policy overhead + residual snapshots —
  competitive claims use **mean stage time**, not end-to-end shot wall time.
- **d=7, p=2e-3**: all streaming policies (`w=2d`, `w=3d`, adaptive) show a
  clear LER gap vs batch (~0.02 vs ~0.001). Larger buffers / better seam
  handling needed — adaptive does not close this gap.
- vs **pure `w=2d`**: adaptive is slightly slower (retry + gate overhead) but
  intended as a safer policy when density spikes; at these `(d,p)` cells
  `w=2d` already matches `w=3d` LER.

### Streaming vs batch (correctness)
- At **p=1e-3, d=3**: streaming LER **exactly matches** batch.
- At **p=1e-3, d=5**: within 1 error of batch at 2000 shots.
- At **p=1e-3, d=7**: small residual gap (2/800 vs 0/800) for both `w=2d` and
  `w=3d` — documented approximation of full-graph masked residual vs truncated
  DEM / rough boundary.

## Method notes

- **batch**: full-shot PyMatching on the complete detector history.
- **stream_w3d / stream_w2d**: residual-syndrome sliding windows with commit
  stride `C=d`; observables from committed matching-edge `fault_ids`.
- **mabs_adaptive**: pre-decode density gate chooses `w=2d` vs `w=3d`; rare
  post-decode retry if mixture-aware `Q` is low *and* hardness gates fire.
  Stage times include retry cost.

## Limitations

- Not claiming Sparse Blossom absolute µs/round leadership.
- Shot budgets give order-of-magnitude credible LERs, not publication
  threshold CIs; increase shots for tighter intervals.
- Truncated per-window DEMs with rough time-boundaries are future work.
