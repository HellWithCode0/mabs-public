# MABS v4.1 (CASCADE) results summary

## Honest goal

Ship Aryaman Katoch's CASCADE priority list (CommitAction cache, pair LUT,
defect gate, topology-safe keys, cost-aware routing, frequency admission,
LER-safe peel/residual, adaptive_depth research hook, `defer_hard` removed),
keep **LER = batch**, and **do not claim stage speedup** vs fair `stream_w3d`
unless benches show it.

Not claiming C++ Sparse Blossom absolute us/round.

## Fair streaming baseline

`window_match_edges` uses a **single** `decode_to_edges_array` (no paired
`decode(..., return_weight=True)`). SLEM/CASCADE hard path matches that.

## Algorithm (4.1.0a1)

**CASCADE** = Cached Approximate Sparse Correction with Amortized Deferred Escalation

| Path | When | Action |
|------|------|--------|
| empty | K=0 | no-op CommitAction |
| gate | K > gate_max (default max(12, 4·clique_cap)) | blossom |
| cost | estimated local us >= blossom budget | blossom early |
| cache | ExactPatternCache hit (topology-safe key) | apply **CommitAction** directly |
| pair | K<=2 | PairPathLUT (lazy Dijkstra fill) |
| clique | 2 < K <= clique_cap | exact clique MWPM |
| peel | single induced component <= peel_cap | local exact; else residual |
| residual / escalate | else | one blossom (full window when residual) |

### Cache key schema (topology-safe)

Absolute (default)::

    (topo_id, sorted_local_defects)

Relative (opt-in `canonicalize_relative=True`)::

    (topo_id, "rel", relative_offsets)

`topo_id = (d, window_depth, commit_stride, n_local, det_lo, graph_token)`.

Values: **CommitAction** (obs XOR + carry globals) - not raw edge arrays.
Frequency admission: insert after >= `cache_admit_after` (default 2) misses.

`defer_hard` **removed** - hard windows always blossom immediately.

`adaptive_depth` / method `mabs_v4_adapt`: research opt-in (default off).

## Measured table - fair harness (warmup=5, fixed order)

| method | d | p | shots | LER | mean_stage_ns | escalate | cache_hit | pair | peel | residual | N_disagree |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| batch | 5 | 0.001 | 2000 | 0.0005 | 4693 |  |  |  |  |  | 0 |
| stream_w3d | 5 | 0.001 | 2000 | 0.0005 | **14408** |  |  |  |  |  | 0 |
| mabs_v3 | 5 | 0.001 | 2000 | 0.0005 | 21219 | 0.741 |  |  |  |  | 0 |
| **mabs_v4** | 5 | 0.001 | 2000 | **0.0005** | **34665** | **0.734** | 0.012 | 0.161 | 0.006 | 0.290 | **0** |
| batch | 7 | 0.001 | 800 | 0 | 15003 |  |  |  |  |  | 0 |
| stream_w3d | 7 | 0.001 | 800 | 0 | **25888** |  |  |  |  |  | 0 |
| mabs_v3 | 7 | 0.001 | 800 | 0 | 29589 | 0.991 |  |  |  |  | 0 |
| **mabs_v4** | 7 | 0.001 | 800 | **0** | **34104** | **0.990** | 0.000 | 0.008 | 0.001 | 0.029 | **0** |
| batch | 5 | 0.002 | 2000 | 0.011 | 10141 |  |  |  |  |  | 0 |
| stream_w3d | 5 | 0.002 | 2000 | 0.011 | **18598** |  |  |  |  |  | 0 |
| mabs_v3 | 5 | 0.002 | 2000 | 0.011 | 22073 | 0.957 |  |  |  |  | 0 |
| **mabs_v4** | 5 | 0.002 | 2000 | **0.011** | **30194** | **0.955** | 0.001 | 0.034 | 0.002 | 0.119 | **0** |
| batch | 7 | 0.002 | 800 | 0.00125 | 32619 |  |  |  |  |  | 0 |
| stream_w3d | 7 | 0.002 | 800 | 0.00125 | **39876** |  |  |  |  |  | 0 |
| mabs_v3 | 7 | 0.002 | 800 | 0.00125 | 42586 | 1.000 |  |  |  |  | 0 |
| **mabs_v4** | 7 | 0.002 | 800 | **0.00125** | **45400** | **1.000** | 0.000 | 0.000 | 0.000 | 0.000 | **0** |

## Claims / limitations

- **Claim:** LER = batch on these cells; N_disagree = 0.
- **Claim:** escalate slightly below v3 at d=5 via pair-LUT + cache + peel.
- **Not claimed:** stage speedup vs fair `stream_w3d` (v4 stage is higher -
  pure-Python routing/pair/peel overhead outweighs fewer blossoms here).
- **Not claimed:** large escalate cut at d=7 (still ~1).
- Multi-component peel disabled under `prefer_correctness` (LER footgun).
- Cache is **ExactPatternCache** storing CommitActions - not isomorphism.

## Reproduce

```bash
pip install -e .
pytest -q
python -m mabs.benchmark --methods batch,stream_w3d,mabs_v3,mabs_v4 \
  --distances 5,7 --noise-sweep 0.001,0.002 --warmup 5 --fixed-order
```

Version: **4.1.0a1** - Author: Aryaman Katoch
