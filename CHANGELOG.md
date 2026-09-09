# Changelog

## Unreleased — correctness and reporting fixes

Aryaman Katoch: fixes found by auditing the package against the manuscript.

- **`syndrome_density` was truncated in FLASH.** The hot path stops the defect
  scan at `cap=3`, which is all the route needs, but the record field inherited
  the cap, so every busy window reported at most `3 / n_local`. The sticky path
  reported `0.0`. Density is now counted once per window, outside every timed
  interval, so the record is right and `tau_match` carries no instrumentation.
- **Both CLI entry points crashed on Windows.** `python -m mabs` died with
  UnicodeEncodeError on the first line containing pi, and
  `python -m mabs.benchmark` died writing `SUMMARY.md` after finishing the run.
  Report writes now specify `encoding="utf-8"` and the entry points reconfigure
  stdio (`mabs._stdio.force_utf8_stdio`).
- **The generated `results/v4_SUMMARY.md` described version 4.0.1a2.** Running
  the documented benchmark command overwrote the curated 4.2 file with stale
  copy that still advertised `defer_hard`. The writer now emits the FLASH route,
  labels `cache` / `clique` as FULL-mode counters, and prints the mode it ran.
- **`gate_max` is a FULL-mode knob.** `test_defect_gate_escalates` passed a
  FLASH config and asserted a FULL counter, so it failed. Split into a FULL test
  and a FLASH test that pins the K>=3 behaviour.
- `stream_shot_cascade_adapt` copies the caller's config instead of setting
  `adaptive_depth` on it in place.
- New reporting, following the manuscript's Table I checklist:
  `paired_overhead` (tau_stage - tau_match on the same rows),
  `boundary_slope_contrast` (Delta alpha_I(q) with the ratio-slope identity
  residual), `host_metadata`, and `SArch` / `deadline_burden`, which refuses a
  deadline reading without slack, round duration and an attributed residual.
  `SMeas` gained `boundary`, `execution`, `statistic`, `fit_interval`,
  `fit_model` and `host`.
- New tests: reporting spec (10), FLASH density guards (2), and a parity guard
  that holds the inlined FLASH hot path to `route_window_flash`, which nothing
  had exercised. Suite is 60 to 75 tests.
- Added `.gitignore`.

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
