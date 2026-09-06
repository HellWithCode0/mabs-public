# MABS v4 (CASCADE) results summary

## Honest goal

Cut blossom **escalate** vs v3.1 SLEM via **CASCADE** (ExactPatternCache + exact
clique MWPM), keep **LER = batch**, and keep mean stage competitive with
`stream_w3d` / v3.

Not claiming C++ Sparse Blossom absolute µs/round.

## Fair streaming baseline fix (4.0.1a2) — CRITICAL

Prior `stream_w3d` ran **two** PyMatching blossom calls per window
(`decode(..., return_weight=True)` **and** `decode_to_edges_array`).
SLEM/CASCADE hard path only runs one `decode_to_edges_array`. That inflated
w3d `stage_ns` (~2×) and made speedups vs w3d look better than they were
(especially when escalate≈1).

**Fix:** `window_match_edges` / `_edges_and_weight` now call only
`decode_to_edges_array` (weight returned as `nan`; unused for timing).

After the fix, when escalate≈1, CASCADE stage is **~parity** with fair w3d
(one blossom each) plus small Python CASCADE overhead — **not** a large win.


## Stage time vs fair stream_w3d (honest)

CASCADE reduces blossom invocation frequency, but on this pure-Python implementation
shortcut/control overhead outweighs those savings in elapsed stage time at d=5/1e-3
(28µs vs fair w3d ~15µs). At d=7 escalate≈1 → ~parity with fair blossom.
**Do not claim stage speedup** vs fair `stream_w3d`.

## Algorithm

**CASCADE** = Cached Approximate Sparse Correction with Amortized Deferred Escalation

| Path | When | Action |
|------|------|--------|
| empty | K=0 | no-op |
| cache | `clique_cap < K ≤ cache_k_max` and exact key hit | replay stored edges |
| clique | `K ≤ clique_cap` (default **2**) | exact 1–2 defect MWPM |
| escalate | else | one `decode_to_edges_array`; store if K≤cache_k_max |

- Truncated per-window DEMs + carry XOR commit → LER-safe default.
- Cache is **ExactPatternCache** (exact absolute/global keys — **not** translation-isomorphism).
- `use_component_clique` **off** (LER risk). `defer_hard` experimental.

## Measured table — fair baseline (4.0.1a2) (this machine)

Warmup=5; fixed method order; same syndromes per cell; N_disagree vs batch.

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

### vs prior (unfair double-blossom) w3d numbers

| d | p | old unfair w3d | fair w3d | old v4 | new v4 | note |
|---:|---:|---:|---:|---:|---:|---|
| 5 | 0.001 | 23285 | 14956 | 28629 | 28259 | w3d now faster than v3/v4 on stage |
| 7 | 0.002 | 59158 | 38616 | 40107 | 40774 | escalate=1 → ~parity (+small overhead) |

## Pareto vs fair w3d / v3

| d | p | fair w3d | v3 (esc) | v4 (esc) | Δesc vs v3 | LER |
|---:|---:|---:|---:|---:|---:|---|
| 5 | 0.001 | 14956 | 22093 (0.741) | 28259 (0.670) | **−0.071** | = batch |
| 5 | 0.002 | 18572 | 21788 (0.957) | 25051 (0.944) | −0.013 | = batch |
| 7 | 0.001 | 26254 | 29907 (0.991) | 30986 (0.988) | −0.003 | = batch |
| 7 | 0.002 | 38616 | 40175 (1.000) | 40774 (1.000) | 0 | = batch |

## Claims / limitations

- **Claim:** LER = batch on these cells; escalate below v3 at d=5 p=1e-3 via cache + K≤2 clique.
- **Claim (fair):** at escalate≈1, v4 stage ≈ fair single-blossom w3d (slightly above due to Python path overhead).
- **Not claimed:** stage speedup vs fair w3d (v3/v4 currently slower on stage; easy-path Python cost).
- **Not claimed:** esc&lt;0.5 at d=5 without stage regression; large esc cut at d=7.
- **Not claimed:** beating C++ Sparse Blossom absolute µs/round.
- Cache is exact pattern memoization — do **not** call it isomorphism.

## Reproduce

```bash
pip install -e .
pytest -q
python -m mabs.benchmark --methods batch,stream_w3d,mabs_v3,mabs_v4 \
  --distances 5,7 --noise-sweep 0.001,0.002 --warmup 5
```

Version: **4.0.1a2**
