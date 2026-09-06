"""Window DEM parse + construction (past-drop / future-truncate)."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Dict, List, Sequence, Tuple
import numpy as np

try:
    import stim
except ImportError as e:  # pragma: no cover
    raise ImportError("stim is required for mabs.streaming") from e

try:
    import pymatching
except ImportError as e:  # pragma: no cover
    raise ImportError("pymatching is required for mabs.streaming") from e


BIG_LAYER = 2**30


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


@dataclass(frozen=True)
class DemComponent:
    """One graphlike piece of a decomposed detector error mechanism."""

    p: float
    dets: Tuple[int, ...]
    obs: Tuple[int, ...]


@dataclass(frozen=True)
class WindowSpec:
    """One window of the sliding-window schedule."""

    index: int
    start_layer: int
    end_layer: int
    commit_end: int
    det_lo: int
    det_hi: int
    commit_det_hi: int

    @property
    def n_local(self) -> int:
        return self.det_hi - self.det_lo


@dataclass
class WindowMatcher:
    """Prebuilt per-window matching object + post-matching lookup tables."""

    spec: WindowSpec
    matching: Any
    n_local: int
    num_nodes: int
    buf: np.ndarray
    layer_of: np.ndarray
    gnode: np.ndarray
    in_commit: np.ndarray
    gnode_list: list
    in_commit_list: list
    edge_keys: np.ndarray
    edge_obs: np.ndarray
    obs_by_pair: Dict[int, int]
    key_stride: int
    commit_start: int
    commit_end: int
    det_lo: int
    det_hi: int


@dataclass
class CircuitBundle:
    """Prebuilt Stim circuit, DEM, matcher, detector layout, window cache."""

    d: int
    circuit: Any
    dem: Any
    matching: Any
    detector_time: np.ndarray
    n_detectors: int
    n_layers: int
    layers: list
    edge_faults: dict
    num_observables: int
    noise: float = 0.0
    rounds: int = 0
    components: List[DemComponent] = field(default_factory=list)
    layer_lo: np.ndarray = field(default_factory=lambda: np.zeros(0, dtype=np.int64))
    window_cache: Dict[Tuple[int, int, int], WindowMatcher] = field(default_factory=dict)


def _build_edge_faults(matching: Any) -> dict:
    edge_faults = {}
    for n1, n2, data in matching.edges():
        a = -1 if n1 is None else int(n1)
        b = -1 if n2 is None else int(n2)
        fids = set(data.get("fault_ids", set()) or set())
        edge_faults[(a, b)] = fids
        edge_faults[(b, a)] = fids
    return edge_faults


def parse_dem_components(dem: Any) -> List[DemComponent]:
    out: List[DemComponent] = []
    for instr in dem.flattened():
        if instr.type != "error":
            continue
        prob = float(instr.args_copy()[0])
        dets: List[int] = []
        obs: List[int] = []
        pieces: List[Tuple[List[int], List[int]]] = []
        for target in instr.targets_copy():
            if target.is_separator():
                pieces.append((dets, obs))
                dets, obs = [], []
            elif target.is_relative_detector_id():
                dets.append(int(target.val))
            elif target.is_logical_observable_id():
                obs.append(int(target.val))
        pieces.append((dets, obs))
        for d, o in pieces:
            if not d and not o:
                continue
            out.append(DemComponent(prob, tuple(sorted(set(d))), tuple(sorted(set(o)))))
    return out


def _layer_boundaries(detector_time: np.ndarray, n_layers: int) -> np.ndarray:
    layer_lo = np.searchsorted(detector_time, np.arange(n_layers + 1), side="left").astype(np.int64)
    layer_lo[n_layers] = len(detector_time)
    return layer_lo


def plan_windows(detector_time: np.ndarray, n_layers: int, w: int, C: int) -> List[WindowSpec]:
    if w < C:
        raise ValueError("window depth w must be at least the commit stride C")
    layer_lo = _layer_boundaries(detector_time, n_layers)
    specs: List[WindowSpec] = []
    start = 0
    idx = 0
    while start < n_layers:
        end = min(start + w, n_layers)
        commit_end = min(start + C, n_layers)
        if end <= commit_end or end == n_layers:
            commit_end = end
        specs.append(WindowSpec(
            index=idx, start_layer=start, end_layer=end, commit_end=commit_end,
            det_lo=int(layer_lo[start]), det_hi=int(layer_lo[end]),
            commit_det_hi=int(layer_lo[commit_end]),
        ))
        if commit_end >= n_layers:
            break
        start = commit_end
        idx += 1
    return specs


def make_window_spec(bundle: CircuitBundle, start: int, w: int, C: int, index: int = 0) -> WindowSpec:
    n_layers = bundle.n_layers
    end = min(start + w, n_layers)
    commit_end = min(start + C, n_layers)
    if end <= commit_end or end == n_layers:
        commit_end = end
    layer_lo = bundle.layer_lo
    return WindowSpec(
        index=index, start_layer=start, end_layer=end, commit_end=commit_end,
        det_lo=int(layer_lo[start]), det_hi=int(layer_lo[end]),
        commit_det_hi=int(layer_lo[commit_end]),
    )


def build_window_dem(
    components: Sequence[DemComponent],
    spec: WindowSpec,
    *,
    past_policy: str = "drop",
    num_observables: int = 1,
) -> Any:
    if past_policy not in ("drop", "truncate"):
        raise ValueError("past_policy must be 'drop' or 'truncate'")
    lo, hi = spec.det_lo, spec.det_hi
    lines: List[str] = []
    for comp in components:
        dets = comp.dets
        if not dets:
            continue
        reaches_past = dets[0] < lo
        if reaches_past and past_policy == "drop":
            continue
        kept = [d - lo for d in dets if lo <= d < hi]
        if not kept:
            continue
        if len(kept) > 2:
            raise ValueError("component is not graphlike after windowing")
        targets = " ".join(f"D{d}" for d in kept)
        if comp.obs:
            targets += " " + " ".join(f"L{o}" for o in comp.obs)
        lines.append(f"error({comp.p!r}) {targets}")
    n_local = spec.n_local
    if n_local <= 0:
        raise ValueError("empty window")
    lines.append(f"detector D{n_local - 1}")
    for o in range(max(1, num_observables)):
        lines.append(f"logical_observable L{o}")
    return stim.DetectorErrorModel("\n".join(lines))


def _obs_mask(fault_ids) -> int:
    mask = 0
    for f in fault_ids:
        mask |= 1 << int(f)
    return mask
