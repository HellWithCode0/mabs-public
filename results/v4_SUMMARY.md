# MABS v4 (CASCADE) results summary

## Honest goal

Cut blossom **escalate** vs v3.1 SLEM via **CASCADE** (ExactPatternCache + exact
clique MWPM), keep **LER = batch**, and keep mean stage competitive with
`stream_w3d` / v3.

Not claiming C++ Sparse Blossom absolute µs/round.

## Fair streaming baseline fix (4.0.1a1) — CRITICAL

Prior `stream_w3d` ran **two** PyMatching blossom calls per window
(`decode(..., return_weight=True)` **and** `decode_to_edges_array`).
SLEM/CASCADE hard path only runs one `decode_to_edges_array`. That inflated
w3d `stage_ns` (~2×) and made speedups vs w3d look better than they were
(especially when escalate≈1).

**Fix:** `window_match_edges` / `_edges_and_weight` now call only
`decode_to_edges_array` (weight returned as `nan`; unused for timing).

After the fix, when escalate≈1, CASCADE stage is **~parity** with fair w3d
(one blossom each) plus small Python CASCADE overhead — **not** a large win.

## Measured table — 4.0.1a1 fair baseline

| method | d | p | shots | LER | mean_stage_ns | escalate | cache_hit | clique | N_disagree |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| batch | 5 | 0.001 | 2000 | 0.0005 | 4585 |  |  |  | 0 |
| stream_w3d | 5 | 0.001 | 2000 | 0.0005 | **14956** |  |  |  | 0 |
| mabs_v3 | 5 | 0.001 | 2000 | 0.0005 | 22093 | 0.741 |  |  | 0 |
| **mabs_v4** | 5 | 0.001 | 2000 | **0.0005** | **28259** | **0.670** | 0.171 | 0.167 | 0 |
| batch | 7 | 0.001 | 800 | 0 | 15086 |  |  |  | 0 |
| stream_w3d | 7 | 0.001 | 800 | 0 | **26254** |  |  |  | 0 |
| mabs_v3 | 7 | 0.001 | 800 | 0 | 29907 | 0.991 |  |  | 0 |
| **mabs_v4** | 7 | 0.001 | 800 | **0** | **30986** | **0.988** | 0.045 | 0.008 | 0 |
| batch | 5 | 0.002 | 2000 | 0.011 | 9819 |  |  |  | 0 |
| stream_w3d | 5 | 0.002 | 2000 | 0.011 | **18572** |  |  |  | 0 |
| mabs_v3 | 5 | 0.002 | 2000 | 0.011 | 21788 | 0.957 |  |  | 0 |
| **mabs_v4** | 5 | 0.002 | 2000 | **0.011** | **25051** | **0.944** | 0.070 | 0.034 | 0 |
| batch | 7 | 0.002 | 800 | 0.00125 | 31486 |  |  |  | 0 |
| stream_w3d | 7 | 0.002 | 800 | 0.00125 | **38616** |  |  |  | 0 |
| mabs_v3 | 7 | 0.002 | 800 | 0.00125 | 40175 | 1.000 |  |  | 0 |
| **mabs_v4** | 7 | 0.002 | 800 | **0.00125** | **40774** | **1.000** | 0.000 | 0.000 | 0 |

At escalate≈1 (d=7,p=0.002): v4 stage 40774 ≈ fair w3d 38616 (~parity + Python overhead).

Version: **4.0.1a1**

## Colab

```python
!rm -rf mabs-public
!git clone https://github.com/HellWithCode0/mabs-public.git
%cd mabs-public
!pip install -e . -q
!pytest -q
!python -m mabs.benchmark --methods batch,stream_w3d,mabs_v3,mabs_v4 \
  --distances 5,7 --noise-sweep 0.001,0.002 --warmup 5
```
