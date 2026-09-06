"""Window matchers, commit/carry, and circuit bundle construction."""

from __future__ import annotations

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

from mabs.streaming_dem import (
    BIG_LAYER,
    CircuitBundle,
    DemComponent,
    WindowMatcher,
    WindowRecord,
    WindowSpec,
    _build_edge_faults,
    _layer_boundaries,
    _obs_mask,
    build_window_dem,
    parse_dem_components,
    plan_windows,
)

def build_window_matcher(
    components: Sequence[DemComponent],
    spec: WindowSpec,
    detector_time: np.ndarray,
    n_detectors: int,
    *,
    past_policy: str = "drop",
    num_observables: int = 1,
) -> WindowMatcher:
    sub = build_window_dem(components, spec, past_policy=past_policy, num_observables=num_observables)
    matching = pymatching.Matching.from_detector_error_model(sub)
    n_local = spec.n_local
    num_nodes = int(matching.num_nodes)
    if num_nodes < n_local:
        raise ValueError(f"window {spec.index}: matcher has {num_nodes} nodes but window holds {n_local} detectors")

    layer_of = np.empty(n_local + 1, dtype=np.int32)
    layer_of[:n_local] = detector_time[spec.det_lo:spec.det_hi]
    layer_of[n_local] = BIG_LAYER
    in_commit = layer_of < spec.commit_end

    trash = n_detectors
    gnode = np.empty(n_local + 1, dtype=np.int64)
    gnode[:n_local] = np.arange(spec.det_lo, spec.det_hi, dtype=np.int64)
    gnode[n_local] = trash

    stride = n_detectors + 2
    pairs: Dict[int, int] = {}
    for u, v, attrs in matching.edges():
        if u >= n_local:
            continue
        vv = n_local if v is None else int(v)
        if vv > n_local:
            continue
        gu = int(gnode[u])
        gv = int(gnode[vv])
        mask = _obs_mask(attrs.get("fault_ids", ()) or ())
        k1 = gu * stride + gv
        k2 = gv * stride + gu
        if k1 in pairs or k2 in pairs:
            pairs[k1] = pairs.get(k1, 0) | mask
            pairs[k2] = pairs.get(k2, 0) | mask
            continue
        pairs[k1] = mask
        pairs[k2] = mask

    if pairs:
        key_arr = np.fromiter(pairs.keys(), dtype=np.int64, count=len(pairs))
        obs_arr = np.fromiter(pairs.values(), dtype=np.uint64, count=len(pairs))
        order = np.argsort(key_arr, kind="stable")
        sorted_keys = np.append(key_arr[order], np.iinfo(np.int64).max)
        sorted_obs = np.append(obs_arr[order], np.uint64(0))
    else:
        sorted_keys = np.array([np.iinfo(np.int64).max], dtype=np.int64)
        sorted_obs = np.array([0], dtype=np.uint64)

    return WindowMatcher(
        spec=spec, matching=matching, n_local=n_local, num_nodes=num_nodes,
        buf=np.zeros(num_nodes, dtype=np.uint8), layer_of=layer_of, gnode=gnode,
        in_commit=in_commit, gnode_list=gnode.tolist(), in_commit_list=in_commit.tolist(),
        edge_keys=sorted_keys, edge_obs=sorted_obs, obs_by_pair=pairs, key_stride=stride,
        commit_start=spec.start_layer, commit_end=spec.commit_end,
        det_lo=spec.det_lo, det_hi=spec.det_hi,
    )


def get_window_matcher(bundle: CircuitBundle, spec: WindowSpec) -> WindowMatcher:
    key = (spec.start_layer, spec.end_layer, spec.commit_end)
    wm = bundle.window_cache.get(key)
    if wm is None:
        wm = build_window_matcher(
            bundle.components, spec, bundle.detector_time, bundle.n_detectors,
            past_policy="drop", num_observables=bundle.num_observables,
        )
        bundle.window_cache[key] = wm
    return wm


def ensure_window_schedule(bundle: CircuitBundle, w: int, C: int) -> List[WindowMatcher]:
    specs = plan_windows(bundle.detector_time, bundle.n_layers, w, C)
    return [get_window_matcher(bundle, s) for s in specs]


def build_surface_code_bundle(d: int, noise: float, rounds: int) -> CircuitBundle:
    circuit = stim.Circuit.generated(
        "surface_code:rotated_memory_z", distance=d, rounds=rounds,
        after_clifford_depolarization=noise, after_reset_flip_probability=noise,
        before_measure_flip_probability=noise, before_round_data_depolarization=noise,
    )
    dem = circuit.detector_error_model(decompose_errors=True)
    matching = pymatching.Matching.from_detector_error_model(dem)
    components = parse_dem_components(dem)

    n_detectors = circuit.num_detectors
    coords = circuit.get_detector_coordinates()
    raw_time = np.zeros(n_detectors, dtype=np.int64)
    for det_id, coord in coords.items():
        raw_time[det_id] = 0 if len(coord) == 0 else int(round(coord[-1]))

    if n_detectors > 1 and np.any(np.diff(raw_time) < 0):
        raise ValueError("detectors are not sorted by time layer; window DEM slicing requires contiguous ranges")

    unique_times = sorted(int(x) for x in np.unique(raw_time))
    time_to_layer = {t: i for i, t in enumerate(unique_times)}
    layers_idx = np.array([time_to_layer[int(t)] for t in raw_time], dtype=np.int64)
    n_layers = len(time_to_layer)
    layers = [np.flatnonzero(layers_idx == ell).astype(np.int64) for ell in range(n_layers)]
    layer_lo = _layer_boundaries(layers_idx, n_layers)

    return CircuitBundle(
        d=d, circuit=circuit, dem=dem, matching=matching, detector_time=layers_idx,
        n_detectors=n_detectors, n_layers=n_layers, layers=layers,
        edge_faults=_build_edge_faults(matching), num_observables=int(circuit.num_observables),
        noise=float(noise), rounds=int(rounds), components=components, layer_lo=layer_lo,
        window_cache={},
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


def _edges_and_weight(matching: Any, syndrome: np.ndarray) -> tuple:
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


def commit_window_edges(edges: np.ndarray, wm: WindowMatcher, carry: np.ndarray, *, carry_forward: bool = True) -> tuple:
    if edges is None or len(edges) == 0:
        return 0, 0
    arr = np.asarray(edges, dtype=np.int64)
    if arr.ndim == 1:
        arr = arr.reshape(-1, 2)
    if arr.size == 0:
        return 0, 0
    stride = wm.key_stride
    incom = wm.in_commit_list
    gnode = wm.gnode_list
    table = wm.obs_by_pair
    obs = 0
    n_committed = 0
    for u, v in arr.tolist():
        u = int(u); v = int(v)
        if u < 0: u = wm.n_local
        if v < 0: v = wm.n_local
        if u > wm.n_local: u = wm.n_local
        if v > wm.n_local: v = wm.n_local
        cu = incom[u]; cv = incom[v]
        if cu or cv:
            gu = gnode[u]; gv = gnode[v]
            obs ^= int(table.get(gu * stride + gv, 0))
            n_committed += 1
            if carry_forward and cu != cv:
                carry[gv if cu else gu] ^= 1
    return obs, n_committed


def window_match_edges(wm: WindowMatcher, buf: np.ndarray) -> tuple:
    try:
        _corr, weight = wm.matching.decode(buf, return_weight=True)
        weight = float(weight)
    except Exception:
        weight = 0.0
    try:
        pairs = wm.matching.decode_to_edges_array(buf)
    except Exception:
        return np.zeros((0, 2), dtype=np.int64), weight
    if pairs is None or len(pairs) == 0:
        return np.zeros((0, 2), dtype=np.int64), weight
    arr = np.asarray(pairs, dtype=np.int64)
    if arr.ndim == 1:
        arr = arr.reshape(-1, 2)
    return arr, weight


def mask_to_obs_array(mask: int, n_obs: int) -> np.ndarray:
    out = np.zeros(max(n_obs, 1), dtype=np.uint8)
    for i in range(out.size):
        out[i] = (mask >> i) & 1
    return out[:n_obs] if n_obs > 0 else out[:0]


def _commit_edges_to_prediction(edges, *, bundle, commit_lo, commit_hi, residual, pred_obs):
    """Legacy full-graph commit (tests / fallback). Prefer commit_window_edges."""
    n_det = bundle.n_detectors
    n_fault = bundle.matching.num_fault_ids
    if edges.size == 0:
        return 0, np.zeros((0, 2), dtype=np.int64)
    edge_layers = np.empty_like(edges)
    n_committed = 0
    for i in range(edges.shape[0]):
        a = int(edges[i, 0]); b = int(edges[i, 1])
        la = -1 if a < 0 or a >= n_det else int(bundle.detector_time[a])
        lb = -1 if b < 0 or b >= n_det else int(bundle.detector_time[b])
        edge_layers[i, 0] = la; edge_layers[i, 1] = lb
        earliest = _earliest_layer(a, b, bundle.detector_time, n_det)
        if earliest < 0:
            continue
        if commit_lo <= earliest < commit_hi:
            for fid in bundle.edge_faults.get((a, b), set()):
                if 0 <= int(fid) < n_fault:
                    pred_obs[int(fid)] ^= 1
            if 0 <= a < n_det: residual[a] = 0
            if 0 <= b < n_det: residual[b] = 0
            n_committed += 1
    return n_committed, edge_layers
