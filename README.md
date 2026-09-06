# MABS v4 — CASCADE

**MABS** (Mixture-Aware / Adaptive Boundary Streaming) is a streaming quantum
error-correction control layer on Stim rotated surface-code memory + PyMatching.

## Honest goal (v4)

**Not** “beat Higgott–Gidney absolute µs in pure Python.”

**Goal:** cut Sparse Blossom *escalate* vs v3.1 SLEM via **CASCADE**
(iso-cache + exact clique MWPM), keep **LER = batch**, and keep mean stage
competitive with `stream_w3d` / v3 where possible.

## CASCADE (v4)

**Cached Approximate Sparse Correction with Amortized Deferred Escalation**

1. **Empty** → no-op.
2. **Iso-cache** — global detector-id keys; stores blossom outcomes for moderate
   hard windows (`clique_cap < K ≤ cache_k_max`) so repeated syndromes skip blossom.
3. **Clique MWPM** — exact matching for `K ≤ clique_cap` (default **2** for stage
   Pareto; raise to 3–6 to cut escalate further at pure-Python Dijkstra cost).
4. **Hard** — one `Matching.decode_to_edges_array` per window (LER-safe).
   Optional `defer_hard` is experimental.
5. Same truncated DEM + carry/XOR commit as v2/v3.

### Default vs low-escalate

| Mode | escalate @ d=5,p=1e-3 | LER | stage vs `stream_w3d` |
|------|----------------------:|-----|------------------------|
| **Default** (`clique_cap=2`, cache on) | below v3 (~0.65–0.67) | = batch | ≈ / slightly above |
| Research (`clique_cap=4`) | ≈0.49 | = batch | slower (Python Dijkstra) |

`use_component_clique=True` can cut escalate further but **risks LER** (off by default).

## Install

```bash
git clone https://github.com/HellWithCode0/mabs-public.git
cd mabs-public
pip install -e .
```

**Colab (public mirror only):**

```python
!rm -rf mabs-public
!git clone https://github.com/HellWithCode0/mabs-public.git
%cd mabs-public
!pip install -e . -q
!pytest -q
!python -m mabs.benchmark --methods batch,stream_w3d,mabs_v3,mabs_v4 --distances 5,7 --noise 0.001
!python -m mabs.benchmark --methods batch,stream_w3d,mabs_v3,mabs_v4 --distances 5,7 --noise 0.002
```

Private: `https://github.com/HellWithCode0/mabs.git`

Requires Python ≥ 3.10. Dependencies: `stim`, `pymatching`, `numpy`, `pytest`,
`matplotlib`.

## Run

```bash
python -m mabs
python -m mabs.benchmark --methods batch,stream_w3d,mabs_v3,mabs_v4
python -m mabs.benchmark --distances 5,7 --noise 0.001
python -m mabs.benchmark --quick
```

## Tests

```bash
pytest -q
```

Includes v4 cache / clique / LER smoke (`tests/test_v4_cascade.py`) and v3 SLEM tests.

## Package layout

```
src/mabs/
  v4/
    cascade.py        # CASCADE streaming shot decoder
    iso_cache.py      # global-key LRU syndrome cache
    clique_mwpm.py    # exact MWPM on K≤6 complete graph + boundary
    baseline_runner.py
  v3/                 # SLEM (still available as mabs_v3)
  streaming_dem.py
  streaming_windows.py
  adaptive.py
  baselines.py
  benchmark.py
results/
  v4_SUMMARY.md
  v3_SUMMARY.md
```

## Version

`4.0.0a1`

## License

MIT
