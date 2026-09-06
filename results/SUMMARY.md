# MABS v2 benchmark summary

**Streaming fix (v2.1):** truncated per-window DEMs with past-drop /
future-truncate + carry-forward. Closes d=7, p=2e-3 LER gap
(~0.02 → ~0.00125, matching batch).

Comparison of **batch MWPM**, fixed streaming (`w=3d`, `w=2d`, `C=d`),
and **MABS-Adaptive** on Stim rotated surface-code memory
(`surface_code:rotated_memory_z`, `R=10·d`) with PyMatching.

## Table

| method | d | p | shots | LER | errors | mean_stage_ns | mean_shot_ns | retry_rate |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| batch | 3 | 0.001 | 4000 | 0.006 | 24 | 807.6 | 807.6 | 0.000 |
| stream_w3d | 3 | 0.001 | 4000 | 0.006 | 24 | 12213.4 | 181291.7 | 0.000 |
| stream_w2d | 3 | 0.001 | 4000 | 0.006 | 24 | 12017.6 | 197136.3 | 0.000 |
| mabs_adaptive | 3 | 0.001 | 4000 | 0.006 | 24 | 13396.0 | 308471.4 | 0.000 |
| batch | 5 | 0.001 | 2000 | 0.0005 | 1 | 5840.3 | 5840.3 | 0.000 |
| stream_w3d | 5 | 0.001 | 2000 | 0.0005 | 1 | 20872.6 | 399669.4 | 0.000 |
| stream_w2d | 5 | 0.001 | 2000 | 0.0005 | 1 | 17615.5 | 362830.8 | 0.000 |
| mabs_adaptive | 5 | 0.001 | 2000 | 0.0005 | 1 | 19290.6 | 393406.1 | 0.002 |
| batch | 7 | 0.001 | 800 | 0 | 0 | 24387.8 | 24387.8 | 0.000 |
| stream_w3d | 7 | 0.001 | 800 | 0 | 0 | 43645.5 | 1547153.6 | 0.000 |
| stream_w2d | 7 | 0.001 | 800 | 0 | 0 | 32737.7 | 1206514.8 | 0.000 |
| mabs_adaptive | 7 | 0.001 | 800 | 0 | 0 | 34246.3 | 567081.4 | 0.036 |
| batch | 3 | 0.002 | 4000 | 0.0245 | 98 | 1604.2 | 1604.2 | 0.000 |
| stream_w3d | 3 | 0.002 | 4000 | 0.0245 | 98 | 14007.0 | 201755.5 | 0.000 |
| stream_w2d | 3 | 0.002 | 4000 | 0.0245 | 98 | 12394.5 | 198588.8 | 0.000 |
| mabs_adaptive | 3 | 0.002 | 4000 | 0.0245 | 98 | 14514.9 | 315326.0 | 0.000 |
| batch | 5 | 0.002 | 2000 | 0.011 | 22 | 11173.1 | 11173.1 | 0.000 |
| stream_w3d | 5 | 0.002 | 2000 | 0.011 | 22 | 26115.4 | 439713.9 | 0.000 |
| stream_w2d | 5 | 0.002 | 2000 | 0.011 | 22 | 20580.6 | 384040.6 | 0.000 |
| mabs_adaptive | 5 | 0.002 | 2000 | 0.011 | 22 | 23701.7 | 433874.2 | 0.026 |
| batch | 7 | 0.002 | 800 | 0.00125 | 1 | 40478.7 | 40478.7 | 0.000 |
| stream_w3d | 7 | 0.002 | 800 | 0.00125 | 1 | 61293.0 | 1649987.6 | 0.000 |
| stream_w2d | 7 | 0.002 | 800 | 0.00125 | 1 | 45688.4 | 1342731.1 | 0.000 |
| mabs_adaptive | 7 | 0.002 | 800 | 0.00125 | 1 | 47258.9 | 696224.0 | 0.043 |


## Before / after (d=7, p=2e-3)

| method | before LER | after LER |
|--------|-----------:|----------:|
| batch | 0.00125 | 0.00125 |
| stream_w3d | **0.02** | **0.00125** |
| stream_w2d | 0.02125 | 0.00125 |
| mabs_adaptive | 0.02125 | 0.00125 |

Streaming now matches batch on these shot budgets (LER gap closed via truncated
window DEMs with past-drop / future-truncate + carry-forward).

## Wins / losses (auto)

- **d=3, p=0.001**: adaptive LER=0.006 vs batch=0.006 (gap=0); stage speedup vs w=3d ≈ 0.91× (retry_rate=0.000).
  - fixed w=2d LER=0.006, stage_ns=12017.6.
- **d=3, p=0.002**: adaptive LER=0.0245 vs batch=0.0245 (gap=0); stage speedup vs w=3d ≈ 0.97× (retry_rate=0.000).
  - fixed w=2d LER=0.0245, stage_ns=12394.5.
- **d=5, p=0.001**: adaptive LER=0.0005 vs batch=0.0005 (gap=0); stage speedup vs w=3d ≈ 1.08× (retry_rate=0.002).
  - fixed w=2d LER=0.0005, stage_ns=17615.5.
- **d=5, p=0.002**: adaptive LER=0.011 vs batch=0.011 (gap=0); stage speedup vs w=3d ≈ 1.10× (retry_rate=0.026).
  - fixed w=2d LER=0.011, stage_ns=20580.6.
- **d=7, p=0.001**: adaptive LER=0 vs batch=0 (gap=0); stage speedup vs w=3d ≈ 1.27× (retry_rate=0.036).
  - fixed w=2d LER=0, stage_ns=32737.7.
- **d=7, p=0.002**: adaptive LER=0.00125 vs batch=0.00125 (gap=0); stage speedup vs w=3d ≈ 1.30× (retry_rate=0.043).
  - fixed w=2d LER=0.00125, stage_ns=45688.4.

## Method notes

- **batch**: full-shot PyMatching on the complete detector history.
- **stream_w3d / stream_w2d**: truncated per-window DEMs (past-drop /
  future-truncate) with commit stride `C=d` and carry-forward; matchers
  cached per unique `(t_lo,t_hi,commit_end)` geometry.
- **mabs_adaptive**: try `w=2d` when mixture-aware confidence `Q` is high;
  re-decode with `w=3d` when `Q` is low (retry budget 1). Same window-DEM
  path; stage times include retry cost.

## Limitations

- Not claiming Sparse Blossom absolute µs/round leadership.
- Campaign sizes give order-of-magnitude credible LERs, not publication
  threshold plots; increase shots for tighter CIs.
- Parallel-window / multi-worker seams are out of scope for this harness.
