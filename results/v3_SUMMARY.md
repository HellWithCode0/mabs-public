# MABS v3 (SLEM) results summary

## Honest goal

Not “beat Higgott–Gidney absolute µs in pure Python.”
Goal: **Pareto-dominate full PyMatching Sparse Blossom calls** on mean
stage time / offered load for streaming windows (Katoch sparse mixture),
with **LER matching batch MWPM** on the same Stim circuits.

## Algorithm (SLEM)

1. **Syndrome clustering** (available in `v3/cluster.py`) — radius-grown CCs
   on the detector graph for analysis / optional multi-cluster local decode.
2. **Easy path**
   - Empty syndrome → no-op (empty-K path).
   - Single isolated defect → exact local boundary-path MWPM (precomputed
     multi-source Dijkstra forest on the window matcher).
3. **Hard path (escalate)** — otherwise one
   `Matching.decode_to_edges_array` call on the truncated window
   (no second weight decode).
4. **Adaptive thresholds** — escalate/empty/local rates tracked; optional tune.
5. Same truncated DEM + carry/XOR commit as v2 → LER batch-matched.
6. Detector graphs prewarmed outside the timed shot loop.

## Measured table (this machine)

| method | d | p | shots | LER | mean_stage_ns | escalate_rate | empty_rate | local_rate |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| batch | 5 | 0.001 | 2000 | 0.0005 | 5677 |  |  |  |
| stream_w3d | 5 | 0.001 | 2000 | 0.0005 | 21610 | 0 |  |  |
| mabs_adaptive | 5 | 0.001 | 2000 | 0.0005 | 19186 | 0.002 |  |  |
| mabs_v3 | 5 | 0.001 | 2000 | 0.0005 | **14539** | 0.880 | 0.10† | ~0.02† |
| batch | 7 | 0.001 | 800 | 0 | 23148 |  |  |  |
| stream_w3d | 7 | 0.001 | 800 | 0 | 42846 | 0 |  |  |
| mabs_adaptive | 7 | 0.001 | 800 | 0 | 37971 | 0.036 |  |  |
| mabs_v3 | 7 | 0.001 | 800 | 0 | **27124** | 0.998 | ~0 | ~0 |
| batch | 5 | 0.002 | 2000 | 0.011 | 11101 |  |  |  |
| stream_w3d | 5 | 0.002 | 2000 | 0.011 | 25991 | 0 |  |  |
| mabs_adaptive | 5 | 0.002 | 2000 | 0.011 | 25099 | 0.026 |  |  |
| mabs_v3 | 5 | 0.002 | 2000 | 0.011 | **19132** | 0.986 |  |  |
| batch | 7 | 0.002 | 800 | 0.00125 | 40925 |  |  |  |
| stream_w3d | 7 | 0.002 | 800 | 0.00125 | 59118 | 0 |  |  |
| mabs_adaptive | 7 | 0.002 | 800 | 0.00125 | 51281 | 0.043 |  |  |
| mabs_v3 | 7 | 0.002 | 800 | 0.00125 | **40391** | 1.000 |  |  |

† empty/local rates from `extra` fields on mabs_v3 rows in `benchmark.csv`.

## Before / after (stage time vs stream_w3d)

| d | p | stream_w3d stage_ns | mabs_v3 stage_ns | speedup | escalate | LER v3 | LER batch |
|---:|---:|---:|---:|---:|---:|---:|---:|
| 5 | 0.001 | 21610 | 14539 | **1.49×** | 0.880 | 0.0005 | 0.0005 |
| 7 | 0.001 | 42846 | 27124 | **1.58×** | 0.998 | 0 | 0 |
| 5 | 0.002 | 25991 | 19132 | **1.36×** | 0.986 | 0.011 | 0.011 |
| 7 | 0.002 | 59118 | 40391 | **1.46×** | 1.000 | 0.00125 | 0.00125 |

## Why it wins

- **Empty skip** avoids blossom on empty-K windows (helps most at lower p / smaller d).
- **Single Sparse Blossom call** vs `stream_w3d`’s decode + decode_to_edges double call.
- **Local 1-defect path** is exact and cheaper than blossom when it fires.
- At high escalate rates (d=7), the single-call saving still dominates.

## Claims / limitations

- **Claim**: on these Stim campaigns, SLEM reduces mean stage time vs always-on
  full-window blossom while keeping LER identical to batch MWPM.
- **Not claimed**: beating C++ Sparse Blossom absolute µs/round.
- Local exactness holds for ≤2-defect MWPM; default cap is 1 (pair Dijkstra is
  often slower than blossom under CPython).
- At higher p / larger d, escalate_rate → 1 and stage ≈ single blossom.
- Pair/cluster peel + full clustering remain available for research; hot path
  avoids expensive clustering.

## Reproduce

```bash
pip install -e .
pytest -q
python -m mabs.benchmark --methods batch,stream_w3d,mabs_adaptive,mabs_v3 \
  --distances 5,7 --noise-sweep 0.001,0.002
```
