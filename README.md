# MABS v2 — Mixture-Aware Adaptive Boundary Streaming

**MABS-Adaptive** is a streaming quantum error-correction (QEC) control /
measurement algorithm: correct sliding-window MWPM on Stim rotated surface-code
memory, with mixture-aware confidence and ADaPT-style adaptive window sizing.

Matching is delegated to [PyMatching](https://github.com/oscarhiggott/PyMatching)
on [Stim](https://github.com/quantumlib/Stim) circuits. MABS is **not** a new
matching kernel — it is a streaming *policy* + measurement harness designed to
match batch MWPM LER while reducing mean stage time vs a fixed `w=3d` window.

## Algorithm (precise)

### 1. Correct streaming baseline (truncated window DEM + carry-forward)

1. Build `surface_code:rotated_memory_z` with `R = 10·d` rounds and circuit-level
   noise `p`; parse the flattened DEM into graphlike components once.
2. Partition detectors into time layers from Stim detector coordinates
   (detectors are time-sorted).
3. For each unique window geometry `(t_lo, t_hi, commit_end)`, **prebuild** a
   filtered Stim `DetectorErrorModel` + `pymatching.Matching` (cached on the
   circuit bundle):
   - Remap in-window detectors to contiguous local ids.
   - **Drop** mechanisms that reach *before* the window start (never truncate
     them onto a shared past boundary — that leaks logical failures).
   - **Truncate** future-reaching mechanisms to in-window detectors (boundary).
4. Online path for each window `[t, t+w)` with commit stride `C`
   (defaults `w=3d`, `C=d` ⇒ buffer `≈2d`):
   - Input: `buf = syndrome[lo:hi] XOR carry[lo:hi]`; clear carry in-window.
   - Match on the cached window matcher (`decode_to_edges_array`).
   - Commit edges whose earlier endpoint lies in `[t, t+C)`: XOR observable
     masks; for commit-boundary straddlers, XOR the far endpoint into `carry`.
   - Advance by `C`. The final window commits through the last layer.
5. Logical error ⇔ predicted observables ≠ sampled observable flips.

This closes the former d=7, p=2e-3 streaming vs batch LER gap (~0.02 → ~0.001);
see `results/SUMMARY.md`.

## Install

```bash
git clone https://github.com/HellWithCode0/mabs-public.git
cd mabs-public
pip install -e .
```

**Colab re-clone:**

```python
!rm -rf mabs-public
!git clone https://github.com/HellWithCode0/mabs-public.git
%cd mabs-public
!pip install -e . -q
!pytest -q
!python -m mabs.benchmark --distances 5,7 --noise 0.002
```

Also: `https://github.com/HellWithCode0/mabs.git`

Requires Python ≥ 3.10. Dependencies: `stim`, `pymatching`, `numpy`, `pytest`, `matplotlib`.

## License

MIT
