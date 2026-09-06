# MABS v4.1 (CASCADE) results summary

## Honest goal

Ship Aryaman Katoch’s CASCADE priority list (CommitAction cache, pair LUT,
defect gate, topology-safe keys, cost-aware routing, frequency admission,
LER-safe peel/residual, adaptive_depth research hook, `defer_hard` removed),
keep **LER = batch**, and **do not claim stage speedup** vs fair `stream_w3d`
unless benches show it.

Not claiming C++ Sparse Blossom absolute µs/round.

## Fair streaming baseline

`window_match_edges` uses a **single** `decode_to_edges_array`. SLEM/CASCADE hard path matches that.

## Algorithm (4.1.0a1)

Route: empty → gate → cost → ExactPatternCache(CommitAction) → PairPathLUT → clique → peel/residual → blossom.

`defer_hard` removed. `adaptive_depth` / `mabs_v4_adapt` research opt-in.

## Measured table — fair harness (warmup=5)

| method | d | p | shots | LER | mean_stage_ns | escalate | pair | peel | N_disagree |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| stream_w3d | 5 | 0.001 | 2000 | 0.0005 | **14408** |  |  |  | 0 |
| mabs_v3 | 5 | 0.001 | 2000 | 0.0005 | 21219 | 0.741 |  |  | 0 |
| **mabs_v4** | 5 | 0.001 | 2000 | **0.0005** | **34665** | **0.734** | 0.161 | 0.006 | **0** |
| stream_w3d | 7 | 0.001 | 800 | 0 | **25888** |  |  |  | 0 |
| **mabs_v4** | 7 | 0.001 | 800 | **0** | **34104** | **0.990** | 0.008 | 0.001 | **0** |
| stream_w3d | 5 | 0.002 | 2000 | 0.011 | **18598** |  |  |  | 0 |
| **mabs_v4** | 5 | 0.002 | 2000 | **0.011** | **30194** | **0.955** | 0.034 | 0.002 | **0** |
| stream_w3d | 7 | 0.002 | 800 | 0.00125 | **39876** |  |  |  | 0 |
| **mabs_v4** | 7 | 0.002 | 800 | **0.00125** | **45400** | **1.000** | 0 | 0 | **0** |

**Not claimed:** stage speedup vs fair stream_w3d.

Version: **4.1.0a1** — Author: Aryaman Katoch
