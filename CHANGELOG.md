# Changelog

## 4.2.0a1 — CASCADE FLASH (default)

Aryaman Katoch: make shortcuts cheaper than one fair blossom.

- **FLASH default**: empty / K=1 boundary CommitAction LUT / K=2 pair CommitAction
  LUT / K≥3 immediate blossom; no UF/peel/ExactPatternCache/cost routing on hot path.
- Prewarm boundary + pair hop-radius 4; optional numba capped defect scan.
- Sticky blossom opt-in (off by default — stage Pareto).
- **FULL** mode (`mode="full"` / `use_flash=False`) keeps 4.1 research path.
- CommitAction carry parity fix (XOR mod 2, not set-unique).
- Fair single-blossom `window_match_edges` retained. LER = batch; N_disagree = 0.
- Stage ~parity with fair `stream_w3d` (see `results/v4_SUMMARY.md` — no false claim).
- Author: Aryaman Katoch.

## 4.1.0a1 — CASCADE priority list

Aryaman Katoch CASCADE priorities (verify-first implementation):

1. **CommitAction cache** — ExactPatternCache stores obs XOR + carry flips; apply on hit.
2. **PairPathLUT** — lazy (det_a, det_b) path cache; replaces per-hit K=2 Dijkstra.
3. **Defect-count gate** — `gate_max` (default max(12, 4·clique_cap)) → blossom.
4. **Topology-safe keys** — `(d, window_depth, commit_stride, n_local, det_lo, graph_token, defects)`.
5. **Cost-aware routing** — escalate when estimated local cost ≥ blossom budget; route counters.
6. **Frequency admission** — admit after ≥2 misses (`cache_admit_after`).
7. **Relative canonicalization** — opt-in `canonicalize_relative=True` under topo id.
8. **Peel / residual** — LER-safe single-component peel; multi-comp → full blossom.
9. **adaptive_depth** / `mabs_v4_adapt` — research opt-in (default off).
10. **`defer_hard` removed** — hard windows always immediate blossom.

Fair single-blossom `window_match_edges` retained. LER = batch on d=5,7 × p=1e-3,2e-3
(N_disagree=0). **No stage-speedup claim** vs fair stream_w3d on these benches.
Author: Aryaman Katoch.

## 4.0.1a2 — sync fair harness + honest CASCADE framing

- Sync paper-benchmark harness: `--warmup`, `--fixed-order`, N_disagree, ExactPatternCache wording.
- Honest stage framing (no claim of stage speedup vs fair `stream_w3d`).
- Author: Aryaman Katoch.

## 4.0.1a1 — fair single-blossom baseline

- **CRITICAL:** single `decode_to_edges_array` in `window_match_edges`.
- Renamed IsoCache → ExactPatternCache.
- Author: Aryaman Katoch.

## 4.0.0a1

- Initial CASCADE (v4) release.
