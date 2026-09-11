"""CLUSTER route: answer K>=3 windows from per-cluster LUTs, with a proof.

WHY
The FLASH router keys on K, the number of fired detectors in a window, and
sends every K>=3 window to a full-window blossom. On this decoder a K>=3 window
is usually not one hard matching problem but several well-separated tiny ones:

    d=5, p=1e-3, 428 sampled K>=3 windows
      largest induced component per window: {2: 366, 3: 12, 4: 46, 5: 2, 6: 2}
    d=7, p=1e-3, 538 sampled K>=3 windows
      largest induced component per window: {2: 294, 3: 35, 4: 178, ...}

In most windows that pay for a blossom over 250 to 1000 nodes, every cluster is
a singleton or a pair, and the repo already holds an exact O(1) answer for both:
the prewarmed FLASH boundary and pair CommitAction LUTs.

STEP 1, SHAPE, WITHOUT UNION-FIND
The retired 4.1 peel path ran union-find in pure Python and lost: its control
plane cost more than the blossom it skipped. General connected components are
more than this question needs, because

    every induced component has at most 2 vertices
      if and only if
    every fired detector has at most one fired neighbour.

(A connected component on 3 or more vertices contains a vertex of degree 2 or
more; maximum degree 1 makes the induced subgraph a matching.) So a pair found
this way is always a single graph edge, which is what lets step 3 precompile
every answer the route can ever need.

STEP 2, OPTIMALITY, BY LP DUALITY
Radius-0 clustering alone is not sound: a singleton can prefer a nearby
defect in another cluster over the boundary. Measured: 7 of 847 decomposable
windows committed something different from blossom, and in all 7 the
decomposed solution was strictly heavier.

So every decomposition is certified before it is used. For minimum-weight
perfect matching with a boundary, a matching M is optimal if there is a vertex
potential y with

    y(u) + y(v) <= D(u, v)   for every pair of defects,
    y(u)        <= B(u)      for every defect (boundary copies fixed at 0),

and equality on every edge M uses. Odd-set duals may be taken as zero, so this
is a valid certificate for the full perfect-matching LP, not just its bipartite
relaxation. The decomposed solution fixes y on each cluster: a singleton gets
y = B, a pair kept together splits its path weight, a pair sent to the boundary
gets y = B on both. Intra-cluster constraints then hold by construction and the
only work is the cross-cluster check against a precomputed all-pairs table.

Measured: the certificate held on 97 to 100 percent of decomposable windows, it
rejected all 7 wrong ones, and on 840 of 840 certified windows the committed
effect matched blossom exactly.

STEP 3, ONE COMPILED CALL
With numba, the shape test, the certificate, the per-cluster lookups and the
XOR composition all run in one compiled pass. The answers come from flat
arrays compiled at prewarm from the window's own FLASH LUT: one CommitAction per
local detector (to the boundary) and one per CSR edge (an adjacent pair). The
carry flips are written to a scratch array and applied in the post stage, so
the timed boundary between matching and post-matching work is unchanged.

Measured on the step-3-less version (Python lookups), the lookups were the
largest single cost of the route: 4.4 us at d=5 against 1.7 us for the kernel.

MARGIN
Certified optimality is under the float weights of DetectorGraph. PyMatching
discretises weights internally, so an exact tie can resolve differently, and a
pair sitting on the together-versus-boundary tie can go either way. Both cases
are refused: cross constraints must hold with slack ``margin`` and a pair must
be ``margin`` away from its tie, else the window goes to blossom.

COST OF THE TABLE
The all-pairs table is dense, float32, computed once per distinct window
structure (bulk windows share one) and only when ``n_local`` is at most
``max_nodes``. Above that the route stays off for the window and it takes the
ordinary blossom path.

WITHOUT NUMBA
The numpy implementation below is kept as the reference and for hosts without
numba. It is correct but slower than one blossom call, so the route should stay
off there.
"""

from __future__ import annotations

import hashlib
from dataclasses import dataclass
from typing import Any, Dict, List, Optional, Tuple
import numpy as np

from mabs.v3.cluster import DetectorGraph, _get_graph_cache
from mabs.v3.local_decode import decode_one_defect, decode_two_defects
from mabs.v4.commit_action import CommitAction, edges_to_commit_action
from mabs.v4.flash_lut import FlashCommitLUT, get_flash_lut

_EMPTY_PAIRS = np.zeros((0, 2), dtype=np.int64)
_EMPTY_IDS = np.zeros(0, dtype=np.int64)

# Default window size above which no all-pairs table is built. 2500 covers the
# d<=9 windows of a w=3d schedule (n_local = 3d(d^2-1), so 2160 at d=9).
DEFAULT_MAX_NODES = 2500
DEFAULT_MARGIN = 0.05

# Status codes returned by the compiled kernel.
_OK = 0
_BAIL_SHAPE = -1
_BAIL_CERT = -2
_BAIL_LUT = -3

# Rows of the packed int64 workspace handed to the kernel. Packing keeps the
# numba call down to a dozen arguments; dispatch cost scales with argument
# count and was about 1.1 us for sixteen.
_W_FIRED, _W_PARTNER, _W_PEDGE, _W_PA, _W_PB, _W_PE, _W_SG, _W_NODES, _W_CL, _W_POS = range(10)
_W_ROWS = 10

# Columns of the per-edge and per-node action tables.
_E_V, _E_OK, _E_OBS, _E_NC, _E_C0, _E_C1 = range(6)
_B_OK, _B_OBS, _B_NC, _B_C0, _B_C1 = range(5)

try:
    from numba import njit

    @njit(cache=True)
    def _nb_route_fused(buf, n, indptr, edge_i, bnd_i, cpool, D, B, margin, ws, y, flips):
        """Shape test, certificate, lookups and XOR composition in one pass.

        Returns (status, obs_xor, n_committed, n_flips). On _OK, the first
        n_flips entries of ``flips`` are global carry indices to XOR. Row
        _W_POS of ``ws`` must be -1 everywhere on entry; it is restored.
        """
        fired = ws[_W_FIRED]
        partner = ws[_W_PARTNER]
        pedge = ws[_W_PEDGE]
        pos = ws[_W_POS]
        k = 0
        for i in range(n):
            if buf[i]:
                fired[k] = i
                pos[i] = k
                partner[k] = -1
                k += 1
        shape_ok = True
        for a in range(k):
            u = fired[a]
            for e in range(indptr[u], indptr[u + 1]):
                j = pos[edge_i[e, _E_V]]
                if j >= 0:
                    if partner[a] == -1:
                        partner[a] = j
                        pedge[a] = e
                    elif partner[a] != j:
                        shape_ok = False
                        break
            if not shape_ok:
                break
        for a in range(k):
            pos[fired[a]] = -1
        if not shape_ok:
            return -1, 0, 0, 0

        pa = ws[_W_PA]
        pb = ws[_W_PB]
        pe = ws[_W_PE]
        sg = ws[_W_SG]
        m = 0
        s = 0
        for a in range(k):
            j = partner[a]
            if j < 0:
                sg[s] = fired[a]
                s += 1
            elif j > a:
                pa[m] = fired[a]
                pb[m] = fired[j]
                pe[m] = pedge[a]
                m += 1

        nodes = ws[_W_NODES]
        cl = ws[_W_CL]
        t = 0
        for i in range(m):
            a = pa[i]
            b = pb[i]
            wab = np.float64(D[a, b])
            Ba = B[a]
            Bb = B[b]
            split_cost = Ba + Bb
            if abs(wab - split_cost) <= margin:
                return -2, 0, 0, 0
            if wab < split_cost:
                ya = min(Ba, max(wab - Bb, 0.5 * wab))
                yb = wab - ya
            else:
                ya = Ba
                yb = Bb
            nodes[t] = a
            y[t] = ya
            cl[t] = i
            t += 1
            nodes[t] = b
            y[t] = yb
            cl[t] = i
            t += 1
        for i in range(s):
            c = sg[i]
            nodes[t] = c
            y[t] = B[c]
            cl[t] = m + i
            t += 1
        for i in range(t):
            if not (y[i] < 1e30):
                return -2, 0, 0, 0
        for i in range(t):
            ui = nodes[i]
            yi = y[i]
            ci = cl[i]
            for j in range(i + 1, t):
                if cl[j] != ci:
                    if np.float64(D[ui, nodes[j]]) - yi - y[j] < margin:
                        return -2, 0, 0, 0

        # Every answer must exist before anything is written.
        for i in range(m):
            if edge_i[pe[i], _E_OK] == 0:
                return -3, 0, 0, 0
        for i in range(s):
            if bnd_i[sg[i], _B_OK] == 0:
                return -3, 0, 0, 0

        obs = 0
        nc = 0
        nf = 0
        for i in range(m):
            e = pe[i]
            obs ^= edge_i[e, _E_OBS]
            nc += edge_i[e, _E_NC]
            for q in range(edge_i[e, _E_C0], edge_i[e, _E_C1]):
                flips[nf] = cpool[q]
                nf += 1
        for i in range(s):
            c = sg[i]
            obs ^= bnd_i[c, _B_OBS]
            nc += bnd_i[c, _B_NC]
            for q in range(bnd_i[c, _B_C0], bnd_i[c, _B_C1]):
                flips[nf] = cpool[q]
                nf += 1
        return 0, obs, nc, nf

    @njit(cache=True)
    def _nb_apply_flips(carry, flips, nf):
        size = carry.shape[0]
        for i in range(nf):
            g = flips[i]
            if 0 <= g < size:
                carry[g] ^= 1

    _HAS_NUMBA = True
except Exception:  # pragma: no cover - graceful fallback
    _HAS_NUMBA = False
    _nb_route_fused = None  # type: ignore
    _nb_apply_flips = None  # type: ignore


def has_numba() -> bool:
    return bool(_HAS_NUMBA)


@dataclass
class ClusterStats:
    """Why cluster attempts fell back to blossom."""

    routed: int = 0
    bail_shape: int = 0
    bail_cert: int = 0
    bail_lut: int = 0
    no_table: int = 0

    def as_dict(self) -> dict:
        return {
            "routed": self.routed,
            "bail_shape": self.bail_shape,
            "bail_cert": self.bail_cert,
            "bail_lut": self.bail_lut,
            "no_table": self.no_table,
        }


def build_weighted_csr(graph: DetectorGraph) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """CSR over local detectors: (indptr, indices, weights).

    The boundary node is dropped, self loops are dropped, and parallel edges
    keep their minimum weight. The shape test reads (indptr, indices); the
    all-pairs table also reads the weights.
    """
    n = int(graph.n_local)
    rows_v: List[np.ndarray] = []
    rows_w: List[np.ndarray] = []
    counts = np.zeros(n, dtype=np.int64)
    for u in range(n):
        best: Dict[int, float] = {}
        for v, w in graph.adj[u]:
            if 0 <= v < n and v != u:
                if v not in best or w < best[v]:
                    best[v] = float(w)
        keys = sorted(best)
        rows_v.append(np.asarray(keys, dtype=np.int64))
        rows_w.append(np.asarray([best[k] for k in keys], dtype=np.float64))
        counts[u] = len(keys)
    indptr = np.zeros(n + 1, dtype=np.int64)
    np.cumsum(counts, out=indptr[1:])
    indices = np.concatenate(rows_v) if n else _EMPTY_IDS
    weights = np.concatenate(rows_w) if n else np.zeros(0, dtype=np.float64)
    return indptr, indices, weights


def boundary_distances(graph: DetectorGraph) -> np.ndarray:
    n = int(graph.n_local)
    B = np.full(n, np.inf, dtype=np.float64)
    for u, dist in graph.bound_dist.items():
        if 0 <= int(u) < n:
            B[int(u)] = float(dist)
    return B


def all_pairs_distances(
    n: int, indptr: np.ndarray, indices: np.ndarray, weights: np.ndarray
) -> np.ndarray:
    """Dense shortest-path table over local detectors, float32.

    Floyd-Warshall, one vectorised relaxation per pivot. Paths through the
    boundary are excluded on purpose: in the boundary-copy formulation a route
    u -> boundary -> v is the two boundary edges, already priced by B(u) + B(v).
    float32 rounding is orders of magnitude below the certificate margin.
    """
    D = np.full((n, n), np.inf, dtype=np.float32)
    np.fill_diagonal(D, 0.0)
    rows = np.repeat(np.arange(n, dtype=np.int64), np.diff(indptr))
    w32 = weights.astype(np.float32)
    D[rows, indices] = np.minimum(D[rows, indices], w32)
    D[indices, rows] = np.minimum(D[indices, rows], w32)
    for k in range(n):
        np.minimum(D, D[:, k, None] + D[None, k, :], out=D)
    return D


# Bulk windows in a schedule have identical local graphs, so one table serves
# them all. Keyed by a digest of the weighted CSR and the boundary distances.
_APSP_CACHE: Dict[str, np.ndarray] = {}


def _structure_key(n, indptr, indices, weights, B) -> str:
    h = hashlib.sha1()
    h.update(np.int64(n).tobytes())
    h.update(indptr.tobytes())
    h.update(indices.tobytes())
    h.update(np.round(weights, 9).tobytes())
    h.update(np.round(np.where(np.isfinite(B), B, -1.0), 9).tobytes())
    return h.hexdigest()


def _compose_effect(a, b):
    return (a[0] ^ b[0], a[1] ^ b[1])


def degenerate_flags(
    n: int,
    adj: List[List[Tuple[int, float]]],
    indptr: np.ndarray,
    indices: np.ndarray,
    B: np.ndarray,
    D: np.ndarray,
    effect,
    *,
    tol: float = DEFAULT_MARGIN,
    cap: int = 2,
) -> Tuple[np.ndarray, np.ndarray]:
    """Flag answers whose committed effect is not unique among near-optimal paths.

    The certificate fixes the pairing up to ``margin``. It does not fix which of
    two equal-weight paths a cluster uses, and two such paths can commit
    different effects: a different observable parity, or a different carry
    global where they cross the commit boundary. PyMatching breaks such ties its
    own way, so an answer is only safe when every path within ``tol`` of optimal
    commits the same effect.

    ``effect(u, v)`` returns the hashable (obs, frozenset(carry)) effect of the
    single edge u-v, with v == n meaning the boundary. Returns (bad_node,
    bad_edge): bad_node[u] marks a singleton answer, bad_edge[e] the adjacent
    pair on CSR edge e.
    """
    identity = (0, frozenset())
    bad_node = np.zeros(n, dtype=bool)
    E: List[Optional[set]] = [None] * n
    # Boundary effects, over the DAG of near-tight predecessors, cheapest first.
    for u in np.argsort(B).tolist():
        if not np.isfinite(B[u]):
            continue
        s: set = set()
        for v, w in adj[u]:
            if v == -1 or v >= n:
                if abs(w - B[u]) <= tol:
                    s.add(effect(u, n))
            elif v >= 0 and B[v] < B[u] and abs(B[v] + w - B[u]) <= tol and E[v]:
                ev = effect(u, v)
                for x in E[v]:
                    s.add(_compose_effect(x, ev))
            if len(s) >= cap:
                break
        E[u] = s
        bad_node[u] = len(s) != 1

    bad_edge = np.zeros(int(indices.size), dtype=bool)
    if D is None:
        return bad_node, bad_edge
    for u in range(n):
        for e in range(int(indptr[u]), int(indptr[u + 1])):
            v = int(indices[e])
            dab = float(D[u, v])
            if not dab < B[u] + B[v]:
                # Sent to the boundary: unique exactly when both halves are.
                bad_edge[e] = bool(bad_node[u] or bad_node[v])
                continue
            Du = D[u].astype(np.float64)
            ball = np.flatnonzero(Du <= dab + tol)
            ball = ball[np.argsort(Du[ball])]
            P = {u: {identity}}
            for x in ball.tolist():
                if x == u:
                    continue
                s = set()
                for y, w in adj[x]:
                    if 0 <= y < n and y in P and Du[y] < Du[x] and abs(Du[y] + w - Du[x]) <= tol:
                        ex = effect(y, x)
                        for z in P[y]:
                            s.add(_compose_effect(z, ex))
                    if len(s) >= cap:
                        break
                if s:
                    P[x] = s
            bad_edge[e] = len(P.get(v, ())) != 1
    return bad_node, bad_edge


def _pack_actions(actions: List[Optional[CommitAction]], ncols: int, extra: Optional[np.ndarray],
                  cpool: List[int]) -> Tuple[np.ndarray, bool]:
    """Flatten CommitActions into an int64 table plus a shared carry pool.

    Returns (table, fits). ``fits`` is False when an observable mask does not fit
    in int64, in which case the compiled path must not be used.
    """
    rows = len(actions)
    tab = np.zeros((rows, ncols), dtype=np.int64)
    fits = True
    base = 1 if extra is not None else 0
    if extra is not None:
        tab[:, 0] = extra
    for r, act in enumerate(actions):
        c0 = len(cpool)
        if act is not None:
            if not (0 <= int(act.obs_xor) < (1 << 62)):
                fits = False
            tab[r, base + 0] = 1
            tab[r, base + 1] = int(act.obs_xor) if fits else 0
            tab[r, base + 2] = int(act.n_committed)
            cpool.extend(int(g) for g in act.carry_globals)
        tab[r, base + 3] = c0
        tab[r, base + 4] = len(cpool)
    return tab, fits


class ClusterTable:
    """Per-window-matcher data for the CLUSTER route, built once at prewarm."""

    __slots__ = ("n_local", "indptr", "indices", "isfired", "pos", "D", "B", "key",
                 "edge_i", "bnd_i", "cpool", "ws", "y", "flips", "fused_ok",
                 "bad_node", "bad_pair", "n_degenerate")

    def __init__(self, graph: DetectorGraph, lut: Optional[FlashCommitLUT] = None,
                 *, build_apsp: bool = True):
        self.n_local = int(graph.n_local)
        self.indptr, self.indices, weights = build_weighted_csr(graph)
        self.B = boundary_distances(graph)
        # Numpy reference path scratch; touched only at fired positions.
        self.isfired = np.zeros(self.n_local, dtype=np.uint8)
        self.pos = np.zeros(self.n_local, dtype=np.int64)
        self.key = _structure_key(self.n_local, self.indptr, self.indices, weights, self.B)
        self.D: Optional[np.ndarray] = None
        if build_apsp:
            D = _APSP_CACHE.get(self.key)
            if D is None:
                D = all_pairs_distances(self.n_local, self.indptr, self.indices, weights)
                _APSP_CACHE[self.key] = D
            self.D = D
        n = self.n_local
        self.ws = np.zeros((_W_ROWS, max(n, 1)), dtype=np.int64)
        self.ws[_W_POS, :] = -1
        self.y = np.zeros(max(2 * n, 1), dtype=np.float64)
        self.edge_i = np.zeros((0, 6), dtype=np.int64)
        self.bnd_i = np.zeros((0, 5), dtype=np.int64)
        self.cpool = np.zeros(1, dtype=np.int64)
        self.flips = np.zeros(1, dtype=np.int64)
        self.fused_ok = False
        # Answers refused because their committed effect is not unique among
        # near-optimal paths. Empty until actions are compiled from a LUT.
        self.bad_node = np.zeros(n, dtype=bool)
        self.bad_pair: set = set()
        self.n_degenerate = (0, 0)
        if lut is not None:
            self._compile_actions(lut)

    def _compile_actions(self, lut: FlashCommitLUT) -> None:
        """Freeze the window's boundary and adjacent-pair answers into arrays.

        Entries the LUT already holds are reused, so the compiled answers are
        the same objects the Python path returns. Entries it lacks are computed
        here from the same exact routines the LUT prewarm uses, and are NOT
        written back: turning the CLUSTER route on must not change what the
        K=1 and K=2 routes see.
        """
        n = self.n_local
        graph = lut.graph
        wm = lut.wm
        cpool: List[int] = []
        bnd: List[Optional[CommitAction]] = []
        for u in range(n):
            act = lut.boundary[u] if u < len(lut.boundary) else None
            if act is None:
                edges = decode_one_defect(graph, u)
                act = None if edges is None else edges_to_commit_action(edges, wm)
            bnd.append(act)
        pairs: List[Optional[CommitAction]] = []
        for u in range(n):
            for e in range(int(self.indptr[u]), int(self.indptr[u + 1])):
                v = int(self.indices[e])
                act = lut.pairs.get(FlashCommitLUT.pair_key(u, v))
                if act is None:
                    edges = decode_two_defects(graph, u, v)
                    act = None if edges is None else edges_to_commit_action(edges, wm)
                pairs.append(act)

        # Refuse any answer whose committed effect is not unique among paths
        # within the margin of optimal; those windows go to blossom instead.
        cache: Dict[Tuple[int, int], Tuple[int, frozenset]] = {}

        def effect(a: int, b: int):
            key = (a, b) if a <= b else (b, a)
            eff = cache.get(key)
            if eff is None:
                act1 = edges_to_commit_action(np.array([[a, b]], dtype=np.int64), wm)
                eff = (int(act1.obs_xor), frozenset(act1.carry_globals))
                cache[key] = eff
            return eff

        bad_node, bad_edge = degenerate_flags(
            n, graph.adj, self.indptr, self.indices, self.B, self.D, effect,
            tol=DEFAULT_MARGIN,
        )
        # A node with no finite boundary distance has no singleton answer at
        # all; that is already a LUT miss, not a degeneracy.
        bad_node &= np.isfinite(self.B)
        self.bad_node = bad_node
        for u in np.flatnonzero(bad_node).tolist():
            bnd[u] = None
        self.bad_pair = set()
        for u in range(n):
            for e in range(int(self.indptr[u]), int(self.indptr[u + 1])):
                if bad_edge[e]:
                    pairs[e] = None
                    self.bad_pair.add(FlashCommitLUT.pair_key(u, int(self.indices[e])))
        self.n_degenerate = (int(bad_node.sum()), len(self.bad_pair))
        self.bnd_i, fits_b = _pack_actions(bnd, 5, None, cpool)
        self.edge_i, fits_e = _pack_actions(pairs, 6, self.indices, cpool)
        self.cpool = np.asarray(cpool if cpool else [0], dtype=np.int64)
        per_action = max(
            [int(r[_B_C1] - r[_B_C0]) for r in self.bnd_i] +
            [int(r[_E_C1] - r[_E_C0]) for r in self.edge_i] + [1]
        )
        self.flips = np.zeros(max(n, 1) * per_action, dtype=np.int64)
        self.fused_ok = bool(fits_b and fits_e and self.D is not None)


def get_cluster_table(wm: Any, *, max_nodes: int = DEFAULT_MAX_NODES) -> Optional[ClusterTable]:
    """The window's ClusterTable, or None when the window is too large."""
    graph = _get_graph_cache(wm)
    if int(graph.n_local) > int(max_nodes):
        return None
    tab = getattr(graph, "_cluster_table", None)
    if tab is None or tab.n_local != int(graph.n_local):
        tab = ClusterTable(graph, get_flash_lut(wm))
        setattr(graph, "_cluster_table", tab)
    return tab


# ---------------------------------------------------------------------------
# Numpy reference path
# ---------------------------------------------------------------------------

def pair_up(
    fired: np.ndarray, tab: ClusterTable
) -> Optional[Tuple[np.ndarray, np.ndarray]]:
    """Split the fired set into isolated pairs and singletons, or give up.

    Returns ``(pairs, singles)``: pairs has shape (m, 2) of local detector ids,
    singles is 1-D. Returns None when some induced component has 3 or more
    defects.
    """
    k = int(fired.size)
    if k == 0:
        return _EMPTY_PAIRS, _EMPTY_IDS
    indptr = tab.indptr
    starts = indptr[fired]
    cnt = indptr[fired + 1] - starts
    total = int(cnt.sum())
    if total == 0:
        return _EMPTY_PAIRS, fired.copy()

    isfired = tab.isfired
    pos = tab.pos
    isfired[fired] = 1
    pos[fired] = np.arange(k, dtype=np.int64)
    try:
        # Gather every neighbour of every fired detector at once. `owner` says
        # which fired index each gathered neighbour belongs to.
        owner = np.repeat(np.arange(k, dtype=np.int64), cnt)
        ends = np.cumsum(cnt)
        nbrs = tab.indices[np.repeat(starts - (ends - cnt), cnt) + np.arange(total, dtype=np.int64)]
        keep = isfired[nbrs] != 0
        if not keep.any():
            return _EMPTY_PAIRS, fired.copy()
        own = owner[keep]
        par = nbrs[keep]
        if np.bincount(own, minlength=k).max() > 1:
            return None
        partner = np.full(k, -1, dtype=np.int64)
        partner[own] = pos[par]
    finally:
        isfired[fired] = 0

    idx = np.arange(k, dtype=np.int64)
    lo = np.flatnonzero(partner > idx)
    pairs = np.stack((fired[lo], fired[partner[lo]]), axis=1) if lo.size else _EMPTY_PAIRS
    return pairs, fired[partner < 0]


def certify(
    pairs: np.ndarray,
    singles: np.ndarray,
    tab: ClusterTable,
    *,
    margin: float = DEFAULT_MARGIN,
) -> bool:
    """True when the per-cluster solution is provably a minimum-weight matching.

    Builds the dual potential the decomposed solution implies and checks every
    cross-cluster constraint with slack ``margin``. Also refuses a pair whose
    together-versus-boundary choice is within ``margin`` of a tie, because the
    matcher's discretised weights could break that tie the other way.
    """
    D = tab.D
    B = tab.B
    if D is None:
        return False
    m = int(pairs.shape[0])
    if m:
        a = pairs[:, 0]
        b = pairs[:, 1]
        wab = D[a, b].astype(np.float64)
        Ba = B[a]
        Bb = B[b]
        split_cost = Ba + Bb
        if np.any(np.abs(wab - split_cost) <= margin):
            return False
        together = wab < split_cost
        # Kept together: y(a) + y(b) = w_ab with y <= B on each side. Split as
        # evenly as the boundary bounds allow, which keeps both potentials as
        # small as possible and so leaves the most cross slack.
        ya_t = np.minimum(Ba, np.maximum(wab - Bb, wab * 0.5))
        ya = np.where(together, ya_t, Ba)
        yb = np.where(together, wab - ya_t, Bb)
        nodes = np.concatenate((a, b, singles))
        y = np.concatenate((ya, yb, B[singles]))
        cl = np.concatenate((np.arange(m), np.arange(m), m + np.arange(singles.size)))
    else:
        nodes = singles
        y = B[singles]
        cl = np.arange(singles.size)
    if not np.all(np.isfinite(y)):
        return False
    Dsub = D[nodes[:, None], nodes[None, :]]
    slack = Dsub - (y[:, None] + y[None, :])
    cross = cl[:, None] != cl[None, :]
    return not bool(np.any(cross & (slack < margin)))


def compose_actions(actions: List[CommitAction]) -> CommitAction:
    """XOR-compose per-cluster CommitActions into one window action.

    ``commit_window_edges`` applies edges independently and XORs their effects,
    so this is exact: observable masks XOR, carry flips cancel in pairs, and
    committed counts add.
    """
    if not actions:
        return CommitAction.empty()
    if len(actions) == 1:
        return actions[0]
    obs = 0
    n_committed = 0
    parity: dict = {}
    for act in actions:
        obs ^= act.obs_xor
        n_committed += act.n_committed
        for g in act.carry_globals:
            parity[g] = parity.get(g, 0) ^ 1
    carry = tuple(sorted(g for g, bit in parity.items() if bit))
    return CommitAction(obs_xor=obs, carry_globals=carry, n_committed=n_committed)


def route_clusters(
    fired: np.ndarray,
    tab: ClusterTable,
    lut: Any,
    *,
    margin: float = DEFAULT_MARGIN,
    stats: Optional[ClusterStats] = None,
) -> Optional[CommitAction]:
    """Numpy reference: certified per-cluster answer, or None for blossom."""
    split = pair_up(fired, tab)
    if split is None:
        if stats is not None:
            stats.bail_shape += 1
        return None
    pairs, singles = split
    if not certify(pairs, singles, tab, margin=margin):
        if stats is not None:
            stats.bail_cert += 1
        return None
    actions: List[CommitAction] = []
    for a, b in pairs.tolist():
        act = None
        if FlashCommitLUT.pair_key(a, b) not in tab.bad_pair:
            act = lut.get_pair(a, b)
        if act is None:
            if stats is not None:
                stats.bail_lut += 1
            return None
        actions.append(act)
    for a in singles.tolist():
        act = None if tab.bad_node[a] else lut.get_boundary(a)
        if act is None:
            if stats is not None:
                stats.bail_lut += 1
            return None
        actions.append(act)
    if stats is not None:
        stats.routed += 1
    return compose_actions(actions)


# ---------------------------------------------------------------------------
# Compiled hot path
# ---------------------------------------------------------------------------

def _count_bail(stats: Optional[ClusterStats], status: int) -> None:
    if stats is None:
        return
    if status == _BAIL_SHAPE:
        stats.bail_shape += 1
    elif status == _BAIL_CERT:
        stats.bail_cert += 1
    else:
        stats.bail_lut += 1


def fused_available(tab: Optional[ClusterTable]) -> bool:
    return bool(_HAS_NUMBA and tab is not None and tab.fused_ok)


def route_clusters_fused(
    buf: np.ndarray,
    n: int,
    tab: Optional[ClusterTable],
    *,
    margin: float = DEFAULT_MARGIN,
    stats: Optional[ClusterStats] = None,
) -> Optional[Tuple[int, int, int]]:
    """Compiled route. Returns (obs_xor, n_committed, n_flips) or None.

    On success the carry flips sit in ``tab.flips[:n_flips]``; apply them with
    ``apply_cluster_flips`` in the post stage. Nothing is written to the carry
    buffer here, so a bail leaves no trace.
    """
    if tab is None or tab.D is None:
        if stats is not None:
            stats.no_table += 1
        return None
    status, obs, nc, nf = _nb_route_fused(
        buf, n, tab.indptr, tab.edge_i, tab.bnd_i, tab.cpool, tab.D, tab.B,
        float(margin), tab.ws, tab.y, tab.flips,
    )
    if status != _OK:
        _count_bail(stats, status)
        return None
    if stats is not None:
        stats.routed += 1
    return int(obs), int(nc), int(nf)


def apply_cluster_flips(carry: np.ndarray, tab: ClusterTable, n_flips: int) -> None:
    if n_flips:
        _nb_apply_flips(carry, tab.flips, n_flips)


def warm_kernels(wm: Any, tab: Optional[ClusterTable], n_detectors: int) -> None:
    """Resolve both compiled kernels before any timed shot.

    Numba compiles (or loads from its cache) on the first call with a given
    argument signature. Without this, that first call happens inside
    timers.match() or timers.post() of whichever measured shot first routes a
    cluster, which at low p and small d can fall after the warm-up shots. One
    call each with the real argument types is enough; the branch taken does not
    matter.
    """
    if not fused_available(tab):
        return
    zeros = np.zeros_like(wm.buf)
    _nb_route_fused(
        zeros, int(tab.n_local), tab.indptr, tab.edge_i, tab.bnd_i, tab.cpool, tab.D, tab.B,
        float(DEFAULT_MARGIN), tab.ws, tab.y, tab.flips,
    )
    _nb_apply_flips(np.zeros(int(n_detectors) + 1, dtype=np.uint8), tab.flips, 0)


def route_clusters_buf(
    buf: np.ndarray,
    n: int,
    tab: Optional[ClusterTable],
    lut: Any,
    *,
    margin: float = DEFAULT_MARGIN,
    stats: Optional[ClusterStats] = None,
    prefer_numba: bool = True,
) -> Optional[CommitAction]:
    """Route straight from the window buffer and return a CommitAction.

    With numba this runs the compiled kernel and wraps its result; without it,
    the numpy reference. The FLASH hot loop calls ``route_clusters_fused``
    directly to skip building the CommitAction.
    """
    if tab is None or tab.D is None:
        if stats is not None:
            stats.no_table += 1
        return None
    if prefer_numba and fused_available(tab):
        res = route_clusters_fused(buf, n, tab, margin=margin, stats=stats)
        if res is None:
            return None
        obs, nc, nf = res
        # CommitAction promises sorted, parity-reduced carry globals; the raw
        # flip list can name one global twice when two clusters both flip it.
        if nf:
            vals, cnt = np.unique(tab.flips[:nf], return_counts=True)
            carry = tuple(int(v) for v in vals[cnt % 2 == 1])
        else:
            carry = ()
        return CommitAction(obs_xor=obs, carry_globals=carry, n_committed=nc)
    return route_clusters(np.flatnonzero(buf[:n]), tab, lut, margin=margin, stats=stats)


__all__ = [
    "ClusterStats",
    "ClusterTable",
    "DEFAULT_MARGIN",
    "DEFAULT_MAX_NODES",
    "all_pairs_distances",
    "apply_cluster_flips",
    "boundary_distances",
    "build_weighted_csr",
    "certify",
    "compose_actions",
    "degenerate_flags",
    "fused_available",
    "get_cluster_table",
    "has_numba",
    "pair_up",
    "route_clusters",
    "route_clusters_buf",
    "route_clusters_fused",
]
