# Changelog

## 4.0.1a2 — sync fair harness + honest CASCADE framing

- Sync paper-benchmark harness to public/private remotes: `--warmup`, `--fixed-order`,
  per-cell shuffle, `N_disagree` columns, and canonical `results/benchmark.csv` with mabs_v4.
- Public docs: scrub remaining **iso-cache** wording → **ExactPatternCache**.
- Honest stage framing (no claim of stage speedup vs fair `stream_w3d`):
  CASCADE reduces blossom invocation frequency, but on this pure-Python implementation
  shortcut/control overhead outweighs those savings in elapsed stage time at d=5/1e-3
  (28µs vs fair w3d ~15µs). At d=7 escalate≈1 → ~parity with fair blossom.
- Author: Aryaman Katoch.

## 4.0.1a1 — fair single-blossom baseline

- **CRITICAL:** `window_match_edges` / `_edges_and_weight` now call only
  `decode_to_edges_array` (no paired `decode(..., return_weight=True)`).
  Prior double blossom made `stream_w3d` stage_ns unfair vs SLEM/CASCADE.
- Renamed `IsoCache` → `ExactPatternCache` (exact memoization, not isomorphism).
  Thin `IsoCache` alias + `iso_cache` module re-export kept.
- Benchmark harness: `N_disagree` vs batch, per-cell method shuffle ( `--fixed-order`
  to disable), short warmup (`--warmup`, default 5).
- Author: Aryaman Katoch. Canonical `results/benchmark.csv` includes mabs_v4 rows.

## 4.0.0a1

- Initial CASCADE (v4) release: cache + clique MWPM + deferred escalation.
