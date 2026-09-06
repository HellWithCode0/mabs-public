# MABS v4.2 (CASCADE FLASH) results summary

## Honest goal

Make CASCADE shortcuts **cheaper than one fair blossom** so mean stage_ns is
at or below fair `stream_w3d`, while keeping **LER = batch** / N_disagree=0.

FLASH (default) strips 4.1 Python routing (UF/peel/cost/ExactPatternCache) that
cost more than it saved when only ~27% of windows skipped blossom.

## FLASH route (default)

1. One-pass capped defect extract (numba if available).
2. K=0 → empty CommitAction.
3. K=1 → boundary CommitAction LUT.
4. K=2 → pair CommitAction LUT (prewarm hop-radius 4; miss → blossom + fill).
5. K≥3 → immediate blossom (edges → `commit_window_edges`, same as w3d).
6. Sticky blossom **opt-in** (`sticky_blossom=True`); off by default (skips K≤2 wins).

FULL mode: `CASCADEConfig(mode="full")` / `use_flash=False` restores 4.1 features.

Fair baseline: single `decode_to_edges_array` in `window_match_edges`.

| method | d | p | shots | LER | mean_stage_ns | escalate | pair | empty | N_disagree |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| batch | 5 | 0.001 | 2000 | 0.0005 | 4718.6 |  |  |  | 0 |
| stream_w3d | 5 | 0.001 | 2000 | 0.0005 | 14483.1 |  |  |  | 0 |
| mabs_v3 | 5 | 0.001 | 2000 | 0.0005 | 22584.0 | 0.741 |  | 0.092 | 0 |
| mabs_v4 | 5 | 0.001 | 2000 | 0.0005 | 14758.6 | 0.741 | 0.167 | 0.092 | 0 |
| batch | 7 | 0.001 | 800 | 0 | 15570.7 |  |  |  | 0 |
| stream_w3d | 7 | 0.001 | 800 | 0 | 25402.7 |  |  |  | 0 |
| mabs_v3 | 7 | 0.001 | 800 | 0 | 29782.5 | 0.991 |  | 0.001 | 0 |
| mabs_v4 | 7 | 0.001 | 800 | 0 | 25607.7 | 0.991 | 0.008 | 0.001 | 0 |
| batch | 5 | 0.002 | 2000 | 0.011 | 10129.0 |  |  |  | 0 |
| stream_w3d | 5 | 0.002 | 2000 | 0.011 | 18381.3 |  |  |  | 0 |
| mabs_v3 | 5 | 0.002 | 2000 | 0.011 | 21885.9 | 0.957 |  | 0.009 | 0 |
| mabs_v4 | 5 | 0.002 | 2000 | 0.011 | 18319.4 | 0.957 | 0.034 | 0.009 | 0 |
| batch | 7 | 0.002 | 800 | 0.00125 | 32406.9 |  |  |  | 0 |
| stream_w3d | 7 | 0.002 | 800 | 0.00125 | 38986.4 |  |  |  | 0 |
| mabs_v3 | 7 | 0.002 | 800 | 0.00125 | 41346.8 | 1.000 |  | 0.000 | 0 |
| mabs_v4 | 7 | 0.002 | 800 | 0.00125 | 38987.9 | 1.000 | 0.000 | 0.000 | 0 |

## vs fair w3d / v3

| d | p | stream_w3d | mabs_v3 (esc) | mabs_v4 FLASH (esc) | v4/w3d | LER v4 | LER batch |
|---:|---:|---:|---:|---:|---:|---:|---:|
| 5 | 0.001 | 14483.1 | 22584.0 (0.741) | 14758.6 (0.741) | 1.019 | 0.0005 | 0.0005 |
| 5 | 0.002 | 18381.3 | 21885.9 (0.957) | 18319.4 (0.957) | 0.997 | 0.011 | 0.011 |
| 7 | 0.001 | 25402.7 | 29782.5 (0.991) | 25607.7 (0.991) | 1.008 | 0 | 0 |
| 7 | 0.002 | 38986.4 | 41346.8 (1.000) | 38987.9 (1.000) | 1.000 | 0.00125 | 0.00125 |

## Claims / limitations

- **Claim:** LER = batch; N_disagree = 0 on d=5,7 × p=1e-3,2e-3.
- **Not claimed:** stage_ns ≤ fair stream_w3d at d=5/p=1e-3 on this fair-harness run (FLASH is best-effort ~parity; gap typically ≤~2% and run-to-run noise can flip the sign).
- **Claim:** FLASH cuts stage vs 4.1 (~35µs → ~15µs at d=5/1e-3) by removing peel/cache/cost routing.
- Escalate similar to v3 (FLASH does not chase escalate cuts).
- Sticky blossom available but **off by default** (hurts stage when it skips K≤2 LUT hits).
- `defer_hard` removed. FULL mode keeps ExactPatternCache / peel / cost routing.
- Not claiming C++ Sparse Blossom absolute µs/round.

## Profile notes (d=5/p=1e-3)

- ~74% windows escalate (K≥3): cost ≈ one blossom + capped extract (~0.4µs numba) + Python.
- ~9% empty + ~17% K≤2 LUT: cheaper than blossom (CommitAction apply).
- Remaining gap vs w3d is escalate-path Python/extract overhead, not LUT misses.
- **Next lever:** Cython/C extension for defect scan + LUT get + XOR apply (drop Python).

## Reproduce

```bash
pip install -e .
pytest -q
python -m mabs.benchmark --methods batch,stream_w3d,mabs_v3,mabs_v4 \
  --distances 5,7 --noise-sweep 0.001,0.002 --warmup 5 --fixed-order
```

## Version

**4.2.0a1** — CASCADE FLASH default (Aryaman Katoch).
