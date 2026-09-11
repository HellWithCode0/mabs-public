# MABS v4.3: CASCADE FLASH + certified CLUSTER route

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

## Speed: the certified CLUSTER route

Most windows that FLASH sends to blossom (K>=3) are not one hard matching
problem. They are several small, well separated clusters, and in most of them
every cluster is a single defect or an adjacent pair. The CLUSTER route answers
those windows from the existing FLASH lookup tables instead of calling blossom:

1. **Shape.** Every induced cluster has at most 2 defects exactly when every
   fired detector has at most one fired neighbour, so no union-find is needed.
2. **Proof.** An LP-duality certificate checks that the per-cluster answer is a
   minimum-weight matching for the whole window, with a margin that refuses
   near ties. A precomputed all-pairs distance table makes it one gather.
3. **Uniqueness.** At prewarm, any detector or pair whose near-optimal paths
   commit different effects is refused, so the committed effect is the same
   for every solution within the margin, including PyMatching's.
4. **One compiled call.** With numba, shape, proof, lookups and XOR composition
   run in a single kernel. Anything uncertain falls back to blossom.

Measured on the author's Windows laptop, numba 0.67, one method per process,
3 repetitions, mean `tau_stage` per window in microseconds:

| cell | FLASH | FLASH + CLUSTER | fair `stream_w3d` |
|---|---:|---:|---:|
| d=5, p=1e-3 | 9.82 | **6.11** (38% less) | 11.89 |
| d=5, p=2e-3 | 13.48 | **11.64** (14% less) | 14.71 |
| d=7, p=1e-3 | 19.53 | **16.48** (16% less) | 21.65 |

The route answers 51 to 60 percent of all windows in these cells, and LER
equals batch in every one. Checked end to end against a batch PyMatching decode
over 4,100 shots at d=3,5,7,9 and p=1e-3, 2e-3: N_disagree = 0.

In denser cells most windows fail the shape test and the route roughly breaks
even (same protocol, 2 repetitions): d=3 p=1e-3 3.75 to 3.39 us (10% less),
d=7 p=2e-3 27.49 to 28.52 us (4% more, inside the run-to-run range), d=9
p=1e-3 36.78 to 35.96 us (2% less).

**Default.** `cluster_route=None` means auto: on when numba is importable, off
otherwise, because without numba the route is about 2x slower than blossom.
`CASCADEConfig(cluster_route=False)` gives plain FLASH. The bench methods
`mabs_v4_flash` and `mabs_v4_cluster` force each side for comparison.
The same change also replaced the no-numba defect scan, a Python loop over the
window buffer, with one `np.flatnonzero` call. That alone cuts FLASH stage time
by 25 to 43 percent on hosts without numba and brings it level with `stream_w3d`.

What is proved and what is measured: the certificate and the uniqueness check
are exact under the float weights of the window graph. That they also match
PyMatching's choice relies on PyMatching's internal weight discretisation
staying below the 0.05 margin, which is measured (0 disagreements), not proved.

The all-pairs table is dense, so it is built only for windows of at most 2500
detectors (d<=9 at w=3d). It adds 0.6 s of prewarm at d=5, 2.4 s at d=7 and
37 s at d=9. Larger windows take the ordinary blossom path.

## Reporting

`mabs.reporting` follows the reporting checklist of the manuscript this decoder
accompanies.

- `S_meas` accompanies every timing claim: timestamp endpoints, execution model
  and timed implementation, `w(d)` and `C`, statistic and workload, fit interval
  and model when a fit is reported, and host metadata.
- `boundary_ratios` gives `R_q(d) = q(tau_stage) / q(tau_match)` for mean,
  median, q95 and q99. `paired_overhead` gives `tau_stage - tau_match` and
  `tau_stage / tau_match` on the identical per-window rows. The two answer
  different questions and both are reported.
- `boundary_slope_contrast` gives `Delta alpha_I(q)` and, beside it,
  `alpha_I(R_q)`. These are the same number by the OLS identity, so
  `identity_residual` is a check on the fit range, not a second result.
- `SArch` holds the conditional architectural fields: `c(d)`, `Lambda(d)`,
  `tau_c` and a residual. `deadline_burden` raises rather than defaulting any of
  them, because a CPU timing run does not on its own license a deadline claim.

## Tests

```bash
pytest -q
```

Includes FLASH LUT vs blossom (K≤2), sticky behavior, FLASH default, LER smoke,
recorded-density guards, the reporting spec, and a parity guard holding the
inlined FLASH hot path to `route_window_flash`.

## Package layout

```
src/mabs/
  reporting.py            # S_meas / S_arch, R_q, paired overhead, slope contrast
  _stdio.py               # UTF-8 console guard for the CLI entry points
src/mabs/v4/
  cascade.py / cascade_core.py / cascade_stream.py / cascade_route.py
  flash_lut.py            # CommitAction boundary+pair LUTs + numba extract
  cluster_route.py        # certified CLUSTER route: shape test, LP certificate, compiled kernel
  exact_pattern_cache.py  # FULL mode
  commit_action.py
  pair_lut.py
  baseline_runner.py
results/v4_SUMMARY.md     # regenerated by python -m mabs.benchmark
```

## Version

`4.3.0a1`

## License

MIT
