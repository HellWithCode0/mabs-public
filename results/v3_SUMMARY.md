# MABS v3.1 (SLEM) results summary

## Honest goal

Not “beat Higgott–Gidney absolute µs in pure Python.”
Goal: **Pareto-dominate full PyMatching Sparse Blossom calls** on mean
stage time / offered load for streaming windows (Katoch sparse mixture),
with **LER matching batch MWPM** on the same Stim circuits.

## Algorithm (SLEM v3.1)

1. **Syndrome clustering** — `v3/cluster.py` (radius-grown) + **`v3/union_find.py`**
   induced fired-subgraph components (radius 0, fast).
2. **Easy path (default)**
   - Empty → no-op.
   - **Exact 1–2 defect** local MWPM (precomputed boundary forest + Dijkstra pair).
3. **Optional low-escalate path** (`use_cluster_local=True`)
   - Induced components of size ≤ `max_local_cluster` (default 2) decoded
     with boundary paths / direct-edge pair matching.
   - Escalate when any component is larger or `n_defects > max_local_defects`.
4. **Hard path** — one `Matching.decode_to_edges_array` (no second weight decode).
5. Detector graphs **prewarmed**; bound paths cached on the graph.
6. Adaptive policy available (`tune=True`, `target_escalate_rate≈0.20–0.25`).

## Pareto: escalate ≪ 1 vs stage ≤ w3d (pure Python)

At p≈1e-3, streaming windows of depth 3d typically contain **many adjacent
defect pairs**. Decoding those locally in pure Python (UF + Dijkstra / peel)
is usually **slower than C++ Sparse Blossom**, so:

| Priority | Config | d=5 esc@1e-3 | d=7 esc@1e-3 | LER | stage vs w3d |
|----------|--------|-------------:|-------------:|-----|--------------|
| **Default (shipped)** | `local_defect_cap=2`, `use_cluster_local=False` | ~0.74 | ~0.99 | = batch | **≤ / faster** |
| Low-escalate research | `use_cluster_local=True`, `max_local_defects=12`, `max_local_cluster=2` | **~0.17** | ~0.73 | mild inflation risk | **~0.45–0.75×** (slower) |

**Cannot hit all three** (esc&lt;0.20 ∧ LER=batch ∧ stage≤w3d) for **d=7** in
pure Python: windows are too dense; esc&lt;0.20 needs multi-cluster local that
loses to blossom on the clock (or approximates and risks LER).

Shipped default = **LER + stage Pareto**, with escalate improved vs v3.0 at d=5
via exact 2-defect local (`0.88 → 0.74`). Low-escalate mode is one flag away.

## Measured table — default v3.1 (this machine)

| method | d | p | shots | LER | mean_stage_ns | escalate_rate | empty_rate | local_rate |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| batch | 5 | 0.001 | 2000 | 0.0005 | 5810 |  |  |  |
| stream_w3d | 5 | 0.001 | 2000 | 0.0005 | 21506 | 0 |  |  |
| mabs_v3 | 5 | 0.001 | 2000 | 0.0005 | **20608** | 0.741 | † | † |
| batch | 7 | 0.001 | 800 | 0 | 23650 |  |  |  |
| stream_w3d | 7 | 0.001 | 800 | 0 | 44006 | 0 |  |  |
| mabs_v3 | 7 | 0.001 | 800 | 0 | **29789** | 0.991 | † | † |
| batch | 5 | 0.002 | 2000 | 0.011 | 11448 |  |  |  |
| stream_w3d | 5 | 0.002 | 2000 | 0.011 | 25410 | 0 |  |  |
| mabs_v3 | 5 | 0.002 | 2000 | 0.011 | **21035** | 0.957 | † | † |
| batch | 7 | 0.002 | 800 | 0.00125 | 39353 |  |  |  |
| stream_w3d | 7 | 0.002 | 800 | 0.00125 | 58451 | 0 |  |  |
| mabs_v3 | 7 | 0.002 | 800 | 0.00125 | **40812** | 1.000 | † | † |

† See `extra` fields / `benchmark.csv` (`empty_rate`, `local_rate`).

## Before / after (stage vs stream_w3d) — default

| d | p | stream_w3d | mabs_v3.0 (esc) | mabs_v3.1 (esc) | speedup v3.1 | LER |
|---:|---:|---:|---:|---:|---:|---:|
| 5 | 0.001 | 21506 | 14539 (0.880) | **20608 (0.741)** | **1.04×** | = batch |
| 7 | 0.001 | 44006 | 27124 (0.998) | **29789 (0.991)** | **1.48×** | = batch |
| 5 | 0.002 | 25410 | 19132 (0.986) | **21035 (0.957)** | **1.21×** | = batch |
| 7 | 0.002 | 58451 | 40391 (1.000) | **40812 (1.000)** | **1.43×** | = batch |

v3.0 was faster on stage with near-always escalate (single blossom vs double
decode). v3.1 spends more on exact 2-defect local → slightly higher stage at
d=5 but still ≤ w3d, with lower escalate and identical LER.

## Low-escalate research mode (opt-in)

```python
from mabs.v3 import SLEMConfig, EscalationPolicy
cfg = SLEMConfig(
    local_defect_cap=2,
    use_cluster_local=True,
    policy=EscalationPolicy(max_local_cluster=2, max_local_defects=12, tune=True,
                            target_escalate_rate=0.20),
)
```

Approx. (same seed family): d=5,p=1e-3 → esc≈0.17, stage ~2× w3d, LER may
drift slightly vs batch (induced-subgraph approx). See `results/v3_1_pareto.csv`.

## Claims / limitations

- **Claim**: default SLEM keeps LER = batch and mean stage_ns ≤ stream_w3d on
  these campaigns; escalate improved at d=5 via exact 2-defect local.
- **Not claimed**: esc&lt;0.20 at d=7 without stage or LER tradeoff in pure Python.
- **Not claimed**: beating C++ Sparse Blossom absolute µs/round.
- Numba left unused (correctness / reliability first).

## Reproduce

```bash
pip install -e .
pytest -q
python -m mabs.benchmark --methods batch,stream_w3d,mabs_v3 \
  --distances 5,7 --noise-sweep 0.001,0.002
```
