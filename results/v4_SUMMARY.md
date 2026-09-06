# MABS v4 (CASCADE) results summary

## Honest goal

Cut blossom **escalate** vs v3.1 SLEM via **CASCADE** (iso-cache + exact clique
MWPM), keep **LER = batch**, and keep mean stage competitive with `stream_w3d` / v3.

Not claiming C++ Sparse Blossom absolute µs/round.

## Algorithm

**CASCADE** = Cached Approximate Sparse Correction with Amortized Deferred Escalation

| Path | When | Action |
|------|------|--------|
| empty | K=0 | no-op |
| cache | `clique_cap < K ≤ cache_k_max` and global key hit | replay stored edges |
| clique | `K ≤ clique_cap` (default **2**) | exact 1–2 defect MWPM (raise cap to 3–6 for research) |
| escalate | else | one `decode_to_edges_array`; store in cache if K≤cache_k_max |

- Truncated per-window DEMs + carry XOR commit (same as v2/v3) → LER-safe default.
- `use_component_clique` **off** (induced components ignore cross-quiescent matchings → LER risk).
- `defer_hard` experimental; default per-window blossom for hard cases.

## Measured table — default v4.0.0a1 (this machine)

| method | d | p | shots | LER | mean_stage_ns | escalate | cache_hit | clique |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| batch | 5 | 0.001 | 2000 | 0.0005 | 6663 |  |  |  |
| stream_w3d | 5 | 0.001 | 2000 | 0.0005 | 23285 |  |  |  |
| mabs_v3 | 5 | 0.001 | 2000 | 0.0005 | 22688 | 0.741 |  |  |
| **mabs_v4** | 5 | 0.001 | 2000 | **0.0005** | **28629** | **0.670** | 0.171 | 0.167 |
| batch | 7 | 0.001 | 800 | 0 | 24463 |  |  |  |
| stream_w3d | 7 | 0.001 | 800 | 0 | 43518 |  |  |  |
| mabs_v3 | 7 | 0.001 | 800 | 0 | 29014 | 0.991 |  |  |
| **mabs_v4** | 7 | 0.001 | 800 | **0** | **30111** | **0.988** | 0.045 | 0.008 |
| batch | 5 | 0.002 | 2000 | 0.011 | 11426 |  |  |  |
| stream_w3d | 5 | 0.002 | 2000 | 0.011 | 25710 |  |  |  |
| mabs_v3 | 5 | 0.002 | 2000 | 0.011 | 23054 | 0.957 |  |  |
| **mabs_v4** | 5 | 0.002 | 2000 | **0.011** | **24511** | **0.944** | 0.070 | 0.034 |
| batch | 7 | 0.002 | 800 | 0.00125 | 40848 |  |  |  |
| stream_w3d | 7 | 0.002 | 800 | 0.00125 | 59158 |  |  |  |
| mabs_v3 | 7 | 0.002 | 800 | 0.00125 | 41113 | 1.000 |  |  |
| **mabs_v4** | 7 | 0.002 | 800 | **0.00125** | **40107** | **1.000** | 0.000 | 0.000 |

## Pareto vs v3 / w3d

| d | p | w3d stage | v3 (esc) | v4 (esc) | Δesc | LER |
|---:|---:|---:|---:|---:|---:|---|
| 5 | 0.001 | 23285 | 22688 (0.741) | 28629 (0.670) | **−0.071** | = batch |
| 5 | 0.002 | 25710 | 23054 (0.957) | 24511 (0.944) | −0.013 | = batch |
| 7 | 0.001 | 43518 | 29014 (0.991) | 30111 (0.988) | −0.003 | = batch |
| 7 | 0.002 | 59158 | 41113 (1.000) | 40107 (1.000) | 0 | = batch |

## Research: `clique_cap=4` (not default)

At d=5, p=1e-3, ~800 shots: esc≈**0.52**, stage≈**157µs** (≫ w3d). Exact K=3–4
clique needs several Dijkstras on the window graph; C++ blossom wins on the clock.
**Cannot hit esc&lt;0.5 ∧ LER=batch ∧ stage≤w3d** in pure Python at this working point
(same honest Pareto wall as v3.1 low-escalate mode).

## Claims / limitations

- **Claim**: default CASCADE keeps LER = batch on these cells; escalate clearly below
  v3.1 at d=5 p=1e-3 (0.67 vs 0.74) via cache + K≤2 clique; stage near w3d/v3
  (slightly above at d=5/1e-3; ≤ w3d elsewhere, sometimes &lt; v3).
- **Not claimed**: esc&lt;0.5 at d=5 without stage regression; esc improvement at d=7
  (windows too dense; escalate stays ≈1).
- **Not claimed**: beating C++ Sparse Blossom absolute µs/round.
- Component-wise clique and aggressive `defer_hard` left off (LER risk).

## Reproduce

```bash
pip install -e .
pytest -q
python -m mabs.benchmark --methods batch,stream_w3d,mabs_v3,mabs_v4 \
  --distances 5,7 --noise-sweep 0.001,0.002
```

Version: **4.0.0a1**
