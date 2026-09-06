# MABS v3.1 — Sparse Local Escalation Matching (SLEM)

**MABS** (Mixture-Aware / Adaptive Boundary Streaming) is a streaming quantum
error-correction control layer on Stim rotated surface-code memory + PyMatching.

## Honest goal (v3.1)

**Not** "beat Higgott–Gidney absolute µs in pure Python."

**Goal:** **Pareto-dominate full PyMatching Sparse Blossom calls** on mean stage
time / offered load for streaming windows on the typical **sparse** workload
(Katoch mixture: many empty/easy windows), with **LER matching batch MWPM** on
the same Stim circuits.

## MABS v3.1 — SLEM

1. **Clustering** — detector-graph CCs (`v3/cluster.py`) + induced fired-subgraph
   UF (`v3/union_find.py`).
2. **Easy path (default)**
   - Empty → no-op.
   - Exact **1–2 defect** local MWPM (cached boundary forest + pair Dijkstra).
3. **Optional cluster local** (`use_cluster_local=True`) for lower escalate rates
   (see Pareto note in `results/v3_SUMMARY.md`).
4. **Escalate** — one Sparse Blossom `decode_to_edges_array` call.
5. Same truncated DEM + carry/XOR commit → LER batch-matched (default config).
6. Numba unused by default (correctness-first).

### Default vs low-escalate Pareto

| Mode | escalate @ d=5,p=1e-3 | LER | stage vs `stream_w3d` |
|------|----------------------:|-----|------------------------|
| **Default** (`local_defect_cap=2`) | ~0.74 | = batch | ≤ / faster |
| `use_cluster_local=True`, md=12 | ~0.17 | slight risk | slower (~2×) |

Esc<0.20 ∧ LER=batch ∧ stage≤w3d is **not simultaneously achievable at d=7**
in pure Python; default ships the LER+stage point. Details: `results/v3_SUMMARY.md`.

## Install

```bash
git clone https://github.com/HellWithCode0/mabs.git
cd mabs
pip install -e .
```

**Colab (v3.1):**

```python
!rm -rf mabs
!git clone https://github.com/HellWithCode0/mabs.git
%cd mabs
!pip install -e . -q
!pytest -q
!python -m mabs.benchmark --methods batch,stream_w3d,mabs_v3 --distances 5,7 --noise 0.001
!python -m mabs.benchmark --methods batch,stream_w3d,mabs_v3 --distances 5,7 --noise 0.002
```

Public mirror: `https://github.com/HellWithCode0/mabs-public.git`

Requires Python ≥ 3.10. Dependencies: `stim`, `pymatching`, `numpy`, `pytest`,
`matplotlib`.

## Run

```bash
python -m mabs
python -m mabs.benchmark --methods batch,stream_w3d,mabs_adaptive,mabs_v3
python -m mabs.benchmark --distances 5,7 --noise 0.001
python -m mabs.benchmark --quick
```

## Tests

```bash
pytest -q
```

Includes v3 empty-path / 1–2 defect exactness / LER agreement / escalate smoke
(`tests/test_v3_slem.py`).

## Package layout

```
src/mabs/
  v3/
    slem.py           # SLEM streaming shot decoder
    cluster.py        # detector-graph clustering + bound-path cache
    union_find.py     # induced fired-subgraph UF
    local_decode.py   # exact 1–2 + cluster peel + blossom helper
    escalate.py       # adaptive escalation policy
  streaming_dem.py
  streaming_windows.py
  adaptive.py         # MABS-Adaptive (v2)
  baselines.py
  benchmark.py
results/
  SUMMARY.md
  v3_SUMMARY.md
  v3_1_pareto.csv
```

## Version

`3.1.0a1`

## License

MIT
