# MABS v4.2 — CASCADE FLASH

**MABS** (Mixture-Aware / Adaptive Boundary Streaming) is a streaming quantum
error-correction control layer on Stim rotated surface-code memory + PyMatching.

Author: **Aryaman Katoch**.

## Honest goal (v4.2)

Make CASCADE shortcuts **cheaper than one fair blossom** so mean `stage_ns` is
competitive with (ideally ≤) fair `stream_w3d`, while keeping **LER = batch**.

4.1 lost on stage (~35µs vs w3d ~14µs) because Python UF/peel/cost/cache routing
cost more than it saved when only ~27% of windows skipped blossom.

## FLASH (default) vs FULL

**FLASH** (`mode="flash"`, default):

1. One-pass defect extract (numba if importable).
2. K=0 → empty CommitAction.
3. K=1 → boundary CommitAction LUT (prewarmed).
4. K=2 → pair CommitAction LUT (prewarm hop-radius 4; miss → blossom + fill).
5. K≥3 → immediate blossom (no UF / peel / ExactPatternCache).
6. Sticky blossom **opt-in** (`sticky_blossom=True`); off by default.

**FULL** (`mode="full"` / `use_flash=False`): 4.1 ExactPatternCache, peel, clique,
cost routing — research, not default.

`defer_hard` removed. Fair baseline: single `decode_to_edges_array`.

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
!python -m mabs.benchmark --methods batch,stream_w3d,mabs_v3,mabs_v4 \
  --distances 5,7 --noise-sweep 0.001,0.002 --warmup 5 --fixed-order
```

Private: `https://github.com/HellWithCode0/mabs.git`

Requires Python ≥ 3.10. Dependencies: `stim`, `pymatching`, `numpy`, `pytest`,
`matplotlib`. Optional: `numba` (faster defect scan).

## Run

```bash
python -m mabs
python -m mabs.benchmark --methods batch,stream_w3d,mabs_v3,mabs_v4 \
  --distances 5,7 --noise-sweep 0.001,0.002 --warmup 5 --fixed-order
```

## Tests

```bash
pytest -q
```

Includes FLASH LUT vs blossom (K≤2), sticky behavior, FLASH default, LER smoke.

## Package layout

```
src/mabs/v4/
  cascade.py / cascade_core.py / cascade_stream.py / cascade_route.py
  flash_lut.py            # CommitAction boundary+pair LUTs + numba extract
  exact_pattern_cache.py  # FULL mode
  commit_action.py
  pair_lut.py
  baseline_runner.py
results/v4_SUMMARY.md
```

## Version

`4.2.0a1`

## License

MIT
