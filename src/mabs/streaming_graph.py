"""DEM / matcher / commit helpers for sliding-window streaming.

Residual overlapping recovery: full DEM + PyMatching; commit edges by earliest
layer in [t,t+C); XOR fault_ids into observables. Approx: full graph + masked
residual (not truncated per-window DEM). Near-batch LER for w≈3d,C=d at p~1e-3.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any
import numpy as np

try:
    import stim
except ImportError as e:  # pragma: no cover
    raise ImportError("stim is required for mabs.streaming") from e

try:
    import pymatching
except ImportError as e:  # pragma: no cover
    raise ImportError("pymatching is required for mabs.streaming") from e


@dataclass
class WindowRecord:
    """Per-window timing, mixture, and confidence observables."""

    d: int
    campaign: int
    shot: int
    window_index: int
    K: int
    tau_input_ns: int
    tau_match_ns: int
    tau_post_ns: int
    tau_stage_ns: int
    empty_output: bool
    commit_lo: int
    commit_hi: int
    n_committed: int = 0
    matching_weight: float = 0.0
    syndrome_density: float = 0.0
    window_depth: int = 0
    retried: bool = False
    confidence: float = 1.0

    def as_dict(self) -> dict[str, Any]:
        return {
            "d": self.d,
            "campaign": self.campaign,
            "shot": self.shot,
            "window_index": self.window_index,
            "K": self.K,
            "tau_input_ns": self.tau_input_ns,
            "tau_match_ns": self.tau_match_ns,
            "tau_post_ns": self.tau_post_ns,
            "tau_stage_ns": self.tau_stage_ns,
            "empty_output": self.empty_output,
            "commit_lo": self.commit_lo,
            "commit_hi": self.commit_hi,
            "n_committed": self.n_committed,
            "matching_weight": self.matching_weight,
            "syndrome_density": self.syndrome_density,
            "window_depth": self.window_depth,
            "retried": self.retried,
            "confidence": self.confidence,
        }


@dataclass
class CircuitBundle:
    """Prebuilt Stim circuit, DEM, matcher, detector layout, edge→fault map."""

    d: int
    circuit: Any
    dem: Any
    matching: Any
    detector_time: np.ndarray  # contiguous layer index per detector
    n_detectors: int
    n_layers: int
    layers: list[np.ndarray]
    edge_faults: dict[tuple[int, int], set[int]]
    num_observables: int
    noise: float = 0.0
    rounds: int = 0


def _build_edge_faults(matching: Any) -> dict[tuple[int, int], set[int]]:
    """Map undirected matching-graph endpoints (-1 = boundary) → fault_ids."""
    edge_faults: dict[tuple[int, int], set[int]] = {}
    for n1, n2, data in matching.edges():
        a = -1 if n1 is None else int(n1)
        b = -1 if n2 is None else int(n2)
        fids = set(data.get("fault_ids", set()) or set())
        edge_faults[(a, b)] = fids
        edge_faults[(b, a)] = fids
    return edge_faults


def build_surface_code_bundle(d: int, noise: float, rounds: int) -> CircuitBundle:
    """Build rotated surface-code memory + PyMatching matcher + time layers."""
    circuit = stim.Circuit.generated(
        "surface_code:rotated_memory_z",
        distance=d,
        rounds=rounds,
        after_clifford_depolarization=noise,
        after_reset_flip_probability=noise,
        before_measure_flip_probability=noise,
        before_round_data_depolarization=noise,
    )
    dem = circuit.detector_error_model(decompose_errors=True)
    matching = pymatching.Matching.from_detector_error_model(dem)

    n_detectors = circuit.num_detectors
    coords = circuit.get_detector_coordinates()
    raw_time = np.zeros(n_detectors, dtype=np.int64)
    for det_id, coord in coords.items():
        if len(coord) == 0:
            raw_time[det_id] = 0
        else:
            raw_time[det_id] = int(round(coord[-1]))

    unique_times = sorted(int(x) for x in np.unique(raw_time))
    time_to_layer = {t: i for i, t in enumerate(unique_times)}
    layers_idx = np.array(
        [time_to_layer[int(t)] for t in raw_time], dtype=np.int64
    )
    n_layers = len(time_to_layer)
    layers: list[np.ndarray] = [
        np.flatnonzero(layers_idx == ell).astype(np.int64) for ell in range(n_layers)
    ]

    return CircuitBundle(
        d=d,
        circuit=circuit,
        dem=dem,
        matching=matching,
        detector_time=layers_idx,
        n_detectors=n_detectors,
        n_layers=n_layers,
        layers=layers,
        edge_faults=_build_edge_faults(matching),
        num_observables=int(circuit.num_observables),
        noise=float(noise),
        rounds=int(rounds),
    )


def _earliest_layer(a: int, b: int, detector_time: np.ndarray, n_det: int) -> int:
    la = -1 if a < 0 or a >= n_det else int(detector_time[a])
    lb = -1 if b < 0 or b >= n_det else int(detector_time[b])
    if la < 0 and lb < 0:
        return -1
    if la < 0:
        return lb
    if lb < 0:
        return la
    return la if la <= lb else lb


def _edges_and_weight(matching: Any, syndrome: np.ndarray) -> tuple[np.ndarray, float]:
    """Matched edges (N,2) with boundary as -1, plus MWPM weight."""
    try:
        _corr, weight = matching.decode(syndrome, return_weight=True)
        weight = float(weight)
    except Exception:
        weight = 0.0
    try:
        pairs = matching.decode_to_edges_array(syndrome)
    except Exception:
        return np.zeros((0, 2), dtype=np.int64), weight
    if pairs is None or len(pairs) == 0:
        return np.zeros((0, 2), dtype=np.int64), weight
    arr = np.asarray(pairs, dtype=np.int64)
    if arr.ndim == 1:
        arr = arr.reshape(-1, 2)
    n = matching.num_detectors
    arr = np.where(arr >= n, -1, arr)
    arr = np.where(arr < -1, -1, arr)
    return arr, weight


def _commit_edges_to_prediction(
    edges: np.ndarray,
    *,
    bundle: CircuitBundle,
    commit_lo: int,
    commit_hi: int,
    residual: np.ndarray,
    pred_obs: np.ndarray,
) -> tuple[int, np.ndarray]:
    """Commit edges with earliest layer in [commit_lo, commit_hi).

    XORs fault_ids into ``pred_obs`` and clears both endpoints in ``residual``.
    Returns (n_committed, edge_layers for post-kernel).
    """
    n_det = bundle.n_detectors
    n_fault = bundle.matching.num_fault_ids
    if edges.size == 0:
        return 0, np.zeros((0, 2), dtype=np.int64)

    edge_layers = np.empty_like(edges)
    n_committed = 0
    for i in range(edges.shape[0]):
        a = int(edges[i, 0])
        b = int(edges[i, 1])
        la = -1 if a < 0 or a >= n_det else int(bundle.detector_time[a])
        lb = -1 if b < 0 or b >= n_det else int(bundle.detector_time[b])
        edge_layers[i, 0] = la
        edge_layers[i, 1] = lb
        earliest = _earliest_layer(a, b, bundle.detector_time, n_det)
        if earliest < 0:
            continue
        if commit_lo <= earliest < commit_hi:
            fids = bundle.edge_faults.get((a, b), set())
            for fid in fids:
                if 0 <= int(fid) < n_fault:
                    pred_obs[int(fid)] ^= 1
            if 0 <= a < n_det:
                residual[a] = 0
            if 0 <= b < n_det:
                residual[b] = 0
            n_committed += 1
    return n_committed, edge_layers
