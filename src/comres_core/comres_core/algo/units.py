"""
M1 of COMRES: the unknown-space decomposition into zones ("units") and their pieces
(docs/comres_spec.md).

In plain words: every opening (a gap where explored free space meets unknown space) leads into
some unexplored space. A unit is the unknown space closest to one opening: every unknown cell
goes to the opening it can be reached from most cheaply, walking through unknown space, where a
step next to a seen wall is dearer than a step in the open (walls tend to continue, so unknown
space hugging a wall is probably more wall). It is a guess of what lies behind that opening;
small units are merged into a neighbour, so a few units have 2-3 openings. The VLM sees units
as "zones" Z1, Z2, ... (the benchmark's nested_units hints use "unit" for a block of rooms). A
unit bigger than one sensor view is cut into pieces; the pieces are what the planner assigns to
robots and where it places target hypotheses.

    openings(occ)                  O_j = 8-connected groups of frontier cells (free cells
                                   with an unknown 4-neighbour), indices 1..n in raster
                                   order of each group's first cell
    wall_distance, skeleton_of,    D = distance to the nearest SEEN wall; w = median of D
    passage_width, step_cost       along the Lee skeleton of the known free space; the price
                                   of stepping onto a cell, 1 + w / D. The skeleton also gives
                                   the corridor ends of corridor_lanes.
    spread(occ, olabel)            all openings at once, octile steps (1 and sqrt 2) priced by
                                   step_cost (unknown cells and the map border are not walls);
                                   each unknown cell goes to the opening with the cheapest
                                   path. Exact cost ties go to whichever opening skimage's
                                   MCP_Geometric reaches first (heap order), as in the
                                   reference; this is not always the smaller index. No
                                   barriers: every unknown cell the spread reaches is labelled.
    merge_map(label, A_min)        |U_j| < A_min -> the neighbour sharing the longest
                                   (4-adjacency) border; dropped if it has none
    Tracker.track(label)           spec rule: U' inherits U's id if |U' n U| > |U'| / 2;
                                   if several U' qualify for one id, the largest overlap
                                   keeps it
    split_pieces(...)              a unit bigger than one sensor view (A_VIEW = pi *
                                   SENSOR_RANGE^2 = 1257 cells) is cut into pieces: the
                                   corridor of the seen map continued straight into the unknown
                                   with its own width (corridor_strips; only its part joined to
                                   the way in, >= A_min cells), then ceil(|rest| / A_VIEW)
                                   sensor-view pieces (first centre = the way in, then
                                   farthest-point centres, plus one in any part no seed reaches;
                                   cells to the nearest centre by the spread's walking cost,
                                   never across a corridor); a leftover thinner than a corridor
                                   it touches joins it; safety net: a piece in several
                                   8-connected parts is split, parts < A_min join the touching
                                   piece of the same unit with the longest border (a part
                                   < A_min touching no piece is dropped, as merge_map drops a
                                   unit); naming rank: the piece at the way in, then the
                                   continued corridors, then the others
    piece_beyond(...)              robot steps (the metric of d_geo) from the zone's way in,
                                   through the zone's unknown cells, to each piece's edge
    PieceTracker.track(grid)       the spec's tracker rule on the piece grid (region ids only;
                                   names are given anew for every map by the naming rank)
    display_names(...)             "<zone><letter>" ("a" = the piece at the way in), or
                                   "<zone>" for a zone that is not cut
    decompose(occ, A_min,          all of the above -> Decomposition. The split runs on the
              tracker, piece_...)  merged, untracked units (the pieces are a function of the
                                   map alone); the pieces carry the tracked zone ids
    descriptors(occ, labels,       per zone: size, way-in width, its openings, d_geo from each
                olabel, ...)       robot's goal to the zone's nearest opening through known
                                   free space
    piece_descriptors(dec)         per piece: name, zone, letter, size, beyond, corridor,
                                   centre line, way in
    local_frontier_units(occ,      the spec's between-merges rule on a robot's own map: its
                  grid, clusters)  local frontiers are the openings of ITS map, cut at its
                                   frontier clusters' borders, each labelled with the majority
                                   label of its adjacent unknown cells in the plan's piece grid
    assigned_parts(clusters, mask) which of the robot's frontier clusters hold a chosen local
                                   frontier, and a representative on it
    piece_walk / frontier_walks    steering: a robot sent to piece k prefers, among its local
                                   frontiers into k's zone, those nearest to k (steps through
                                   the zone's cells; a frontier into k itself is at 0)

A port of an offline reference decomposition (it gave identical grids and names on 949
recorded snapshots), except that a unit's 27th and later pieces are lettered
"aa", "ab", ... where the reference wrote non-letters, and that a part < A_min touching no piece
is dropped. A_min is the only number of the decomposition.

Choices the reference makes (kept exactly, so the partition is identical):
  * D is inf everywhere when no occupied cell has been seen (the spread is then the plain
    octile nearest-opening rule); w = 1 when the skeleton is empty.
  * The spread starts from the openings' own frontier cells and moves only through
    unknown cells (skimage MCP_Geometric prices a step as its length times the mean cost
    of its two cells); each cell takes the opening its cheapest path starts from. On an
    exact cost tie that is the opening MCP_Geometric's heap settles first (on 229
    recorded snapshots, 314 tied cells, 149 of them to the larger index); every labelled
    cell is a true minimum-cost cell.
  * merge: smallest first (ties: smaller label); border = number of 4-adjacent cell
    pairs; border ties -> smaller label; the absorbing unit keeps its label and gains the
    other's openings.
  * An opening whose own unit is empty (all its adjacent unknown cells went to another
    opening) belongs to the unit given by the majority label of its adjacent unknown
    cells (the spec's rule for labelling a local frontier).
  * Tracker: ties in the largest overlap go to the smaller current label; ids no unit
    inherits are retired; new units get fresh ids in order of their raw label.

Grid: (row, col); -1 unknown, 0 free, 1 occupied.
"""

import math
import time
from dataclasses import dataclass
from typing import Dict, List, Optional, Sequence, Tuple

import numpy as np
from scipy import ndimage

import experiment_config
from grid_map import frontier_mask, geodesic

UNKNOWN, FREE, OCC = -1, 0, 1
A_MIN_DEFAULT = 20
# One sensor view: a unit bigger than the area a robot sees from one spot is cut into pieces.
# 20 = experiment_config.SENSOR_RANGE, so A_VIEW = pi * 20^2 = 1256.6 cells.
A_VIEW = math.pi * experiment_config.SENSOR_RANGE ** 2
TANGENT_K = 6            # skeleton steps behind a skeleton end that give its direction
# What the VLM (prompt, table, image, reply schema) and the plan records call a unit:
# "zone" with ids Z1, Z2, ... Every nested_units hint uses "unit" for a block of rooms
# ("the second unit south of the corridor"), and 36 hints use "pocket" for the target's
# spot; no hint uses "zone".
ID_PREFIX = "Z"
WORD = "zone"
SKELETON_METHOD = "lee"
_EIGHT = np.ones((3, 3), bool)
_NB4 = ((1, 0), (-1, 0), (0, 1), (0, -1))

Cell = Tuple[int, int]


# ---------------------------------------------------------------------------
# Openings
# ---------------------------------------------------------------------------

def openings(occ: np.ndarray) -> Tuple[np.ndarray, List[Dict]]:
    """O_j = 8-connected components of frontier cells (free cells with a 4-neighbour in
    G_un). Indices 1..n in raster order of each component's first cell (ndimage.label).
    Returns (olabel int32[H,W], 0 = no opening; [{"id": j, "cells": [(r, c), ...]}])."""
    occ = np.asarray(occ)
    fm = frontier_mask(occ)
    olabel, n = ndimage.label(fm, structure=_EIGHT)
    olabel = olabel.astype(np.int32)
    ops = []
    if n:
        flat = np.flatnonzero(olabel)
        lab = olabel.ravel()[flat]
        order = np.argsort(lab, kind="stable")
        flat, lab = flat[order], lab[order]
        rows, cols = np.divmod(flat, occ.shape[1])
        starts = np.searchsorted(lab, np.arange(1, n + 1))
        ends = np.append(starts[1:], lab.size)
        for j in range(n):
            s, e = int(starts[j]), int(ends[j])
            ops.append({"id": j + 1,
                        "cells": list(zip(rows[s:e].tolist(), cols[s:e].tolist()))})
    return olabel, ops


# ---------------------------------------------------------------------------
# Passage width and step cost
# ---------------------------------------------------------------------------

def skeleton_of(occ: np.ndarray) -> np.ndarray:
    """Skeleton of the known free space (Lee thinning)."""
    from skimage.morphology import skeletonize
    return skeletonize(np.asarray(occ) == FREE, method=SKELETON_METHOD).astype(bool)


def wall_distance(occ: np.ndarray) -> np.ndarray:
    """Euclidean distance of every cell to the nearest SEEN occupied cell (inf if none).
    Unknown cells and the grid border are not walls."""
    occ = np.asarray(occ)
    if not (occ == OCC).any():
        return np.full(occ.shape, np.inf)
    return ndimage.distance_transform_edt(occ != OCC)


def passage_width(occ: np.ndarray, D: Optional[np.ndarray] = None, sk: Optional[np.ndarray] = None) -> float:
    """w: median clearance (D, the distance to the nearest seen wall) along the skeleton `sk` of
    the known free space; 1.0 when there is no finite value to take the median of."""
    if D is None:
        D = wall_distance(occ)
    v = D[skeleton_of(occ) if sk is None else sk]
    v = v[np.isfinite(v)]
    return float(np.median(v)) if v.size else 1.0


def step_cost(occ: np.ndarray, D: Optional[np.ndarray] = None,
              w: Optional[float] = None) -> np.ndarray:
    """Per-cell price of stepping onto a cell: 1 + w / D."""
    if D is None:
        D = wall_distance(occ)
    if w is None:
        w = passage_width(occ, D)
    with np.errstate(divide="ignore"):
        return 1.0 + w / D


# ---------------------------------------------------------------------------
# The spread
# ---------------------------------------------------------------------------

def mcp_labels(cost: np.ndarray, olabel: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
    """Multi-source weighted shortest paths from every opening cell (olabel > 0) with
    skimage's MCP_Geometric (8-connected) over `cost` (inf = impassable); each reached cell
    takes the opening its cheapest path starts from. Returns (label int32, cumulative cost)."""
    from skimage.graph import MCP_Geometric
    H, W = olabel.shape
    starts = np.argwhere(olabel > 0)
    if len(starts) == 0:
        return np.zeros((H, W), np.int32), np.full((H, W), np.inf)
    m = MCP_Geometric(cost, fully_connected=True)
    cum, tb = m.find_costs([tuple(x) for x in starts.tolist()])
    offs = np.asarray(m.offsets)
    idx = np.arange(H * W).reshape(H, W)
    pred = idx.copy()
    ok = tb >= 0
    k = tb[ok]
    rr, cc = np.nonzero(ok)
    pred[rr, cc] = (rr - offs[k, 0]) * W + (cc - offs[k, 1])
    pred = pred.ravel()
    while True:                                     # pointer jumping to the start cell
        nxt = pred[pred]
        if np.array_equal(nxt, pred):
            break
        pred = nxt
    lab = olabel.ravel()[pred].reshape(H, W).astype(np.int32)
    lab[tb == -2] = 0
    return lab, cum


def spread(occ: np.ndarray, olabel: np.ndarray, cost_all: np.ndarray) -> np.ndarray:
    """Opening index per unknown cell (0 = unreached: no opening leads there through unknown
    space); cost_all = step_cost(occ)."""
    occ = np.asarray(occ)
    olabel = np.asarray(olabel, np.int32)
    unk = occ == UNKNOWN
    cost = np.where(unk | (olabel > 0), cost_all, np.inf)
    lab, _ = mcp_labels(cost, olabel)
    return np.where(unk, lab, 0).astype(np.int32)


# ---------------------------------------------------------------------------
# Merge
# ---------------------------------------------------------------------------

def _borders(label: np.ndarray) -> Dict[Tuple[int, int], int]:
    """{(a, b): number of 4-adjacent cell pairs with labels a < b, both > 0}."""
    out: Dict[Tuple[int, int], int] = {}
    for a, b in ((label[1:], label[:-1]), (label[:, 1:], label[:, :-1])):
        m = (a > 0) & (b > 0) & (a != b)
        if not m.any():
            continue
        x, y = a[m].astype(np.int64), b[m].astype(np.int64)
        lo, hi = np.minimum(x, y), np.maximum(x, y)
        keys, cnt = np.unique(lo * (int(label.max()) + 1) + hi, return_counts=True)
        base = int(label.max()) + 1
        for k, n in zip(keys.tolist(), cnt.tolist()):
            key = (k // base, k % base)
            out[key] = out.get(key, 0) + n
    return out


def merge_map(label: np.ndarray, A_min: int) -> Tuple[np.ndarray, Dict[int, int]]:
    """A unit with |U_j| < A_min merges into the neighbouring unit sharing its longest
    (4-adjacency) border; with no neighbour its label is dropped. Repeats smallest first
    until every unit has >= A_min cells. Returns (the merged labels, {label: label it ended
    up in (0 = dropped)} for every label present in the input)."""
    lab = np.asarray(label, np.int32).copy()
    ids, cnt = np.unique(lab[lab > 0], return_counts=True)
    size = dict(zip(ids.tolist(), cnt.tolist()))
    parent = {k: k for k in size}
    border = _borders(lab)
    while True:
        small = sorted((n, k) for k, n in size.items() if n < A_min)
        if not small:
            break
        _, s = small[0]
        nbs = {}
        for (a, b), n in border.items():
            if a == s:
                nbs[b] = n
            elif b == s:
                nbs[a] = n
        t = min(nbs, key=lambda k: (-nbs[k], k)) if nbs else 0
        lab[lab == s] = t
        parent[s] = t
        if t:
            size[t] += size[s]
        del size[s]
        nb2: Dict[Tuple[int, int], int] = {}
        for (a, b), n in border.items():                 # rewire s's borders to t
            if s not in (a, b):
                nb2[(a, b)] = nb2.get((a, b), 0) + n
                continue
            o = b if a == s else a
            if t and o != t:
                key = (min(o, t), max(o, t))
                nb2[key] = nb2.get(key, 0) + n
        border = nb2

    def root(k):
        while parent[k] not in (0, k):
            k = parent[k]
        return parent[k]

    return lab, {k: root(k) for k in parent}


# ---------------------------------------------------------------------------
# Tracking
# ---------------------------------------------------------------------------

class Tracker:
    """Persistent unit ids across a run's replans. track(label) takes the (merged) raw
    label grid and returns the same partition with tracked ids: U' inherits U's id if
    |U' n U| > |U'| / 2; when several U' qualify for the same id the one with the largest
    overlap keeps it (tie: smaller raw label); every other U' gets a fresh id."""

    def __init__(self):
        self.prev: Optional[np.ndarray] = None
        self.next_id = 1
        self.last_map: Dict[int, int] = {}     # raw label -> tracked id of the last call

    def track(self, label: np.ndarray) -> np.ndarray:
        label = np.asarray(label, np.int32)
        raw = [int(k) for k in np.unique(label[label > 0])]
        prev = self.prev if (self.prev is not None and self.prev.shape == label.shape) else None
        claims: Dict[int, List[Tuple[int, int]]] = {}    # old id -> [(overlap, raw)]
        if prev is not None:
            m = (label > 0) & (prev > 0)
            if m.any():
                a, b = label[m].astype(np.int64), prev[m].astype(np.int64)
                base = int(prev.max()) + 1
                keys, cnt = np.unique(a * base + b, return_counts=True)
                sizes = dict(zip(*[x.tolist() for x in np.unique(label[label > 0], return_counts=True)]))
                for k, n in zip(keys.tolist(), cnt.tolist()):
                    u, U = k // base, k % base
                    if 2 * n > sizes[u]:                 # |U' n U| > |U'| / 2
                        claims.setdefault(U, []).append((n, u))
        mp: Dict[int, int] = {}
        for U, lst in claims.items():
            n, u = min(lst, key=lambda x: (-x[0], x[1]))
            mp[u] = U
        for u in raw:
            if u not in mp:
                mp[u] = self.next_id
                self.next_id += 1
        self.next_id = max([self.next_id] + [v + 1 for v in mp.values()])
        lut = np.zeros(int(label.max()) + 1 if label.size else 1, np.int32)
        for u, v in mp.items():
            lut[u] = v
        out = lut[label]
        self.prev = out
        self.last_map = mp
        return out


# ---------------------------------------------------------------------------
# Frontier -> unit
# ---------------------------------------------------------------------------

def frontier_label(label: np.ndarray, cells: Sequence[Cell]) -> int:
    """The unit of a set of frontier cells: majority label of their adjacent (labelled)
    4-neighbours, counting each adjacency; tie -> smaller label; 0 if none (_majority). Labels
    live only on unknown cells, so these are the adjacent unknown cells."""
    cells = np.asarray(list(cells), np.int64).reshape(-1, 2)
    return int(_majority(cells, np.zeros(len(cells), np.int64), 1, label)[0])


def _majority(cells: np.ndarray, owner: np.ndarray, n: int, grid: np.ndarray,
              unknown: Optional[np.ndarray] = None) -> np.ndarray:
    """int64[n]: for each owner (group of frontier cells) the majority label of the cells'
    labelled 4-neighbours in `grid` (only those in `unknown` if given), counting each
    adjacency; ties -> smaller label; 0 if none. cells (N, 2), owner (N,) in 0..n-1."""
    out = np.zeros(n, np.int64)
    if n == 0 or len(cells) == 0:
        return out
    grid = np.asarray(grid)
    H, W = grid.shape
    own, lab = [], []
    for dr, dc in _NB4:
        r, c = cells[:, 0] + dr, cells[:, 1] + dc
        ok = (r >= 0) & (r < H) & (c >= 0) & (c < W)
        r, c, o = r[ok], c[ok], owner[ok]
        v = grid[r, c]
        m = v > 0
        if unknown is not None:
            m &= np.asarray(unknown)[r, c]
        own.append(o[m])
        lab.append(v[m].astype(np.int64))
    own_a, lab_a = np.concatenate(own), np.concatenate(lab)
    if own_a.size == 0:
        return out
    base = int(lab_a.max()) + 1
    keys, cnt = np.unique(own_a * base + lab_a, return_counts=True)
    ko, kl = keys // base, keys % base
    order = np.lexsort((kl, -cnt, ko))                    # per owner: most, then smaller label
    ko, kl = ko[order], kl[order]
    first = np.ones(len(ko), bool)
    first[1:] = ko[1:] != ko[:-1]
    out[ko[first]] = kl[first]
    return out


def local_frontier_units(occ: np.ndarray, grid: np.ndarray, clusters: Sequence[Dict]
                         ) -> Tuple[np.ndarray, int, np.ndarray]:
    """
    The spec's rule between merges ("a robot labels a local frontier by the majority label
    of its adjacent unknown cells") on a robot's own map `occ`. Its local frontiers are the
    openings of its own map (openings(occ): 8-connected groups of its frontier cells), cut
    where its frontier clusters (the selector's candidates, `clusters`; [] = not cut) meet,
    so every local frontier is one candidate's share of one opening. Each gets the majority
    label (frontier_label's rule) in the plan's piece grid `grid`, counting only cells unknown
    on the robot's map. Right after a merge the robot's map is the merged map, so the local
    frontiers are parts of the plan's own openings.

    Returns (cell_unit int64[H,W]: the label of every frontier cell, 0 elsewhere or when its
    local frontier has no labelled unknown neighbour; the number of local frontiers with no
    label; part int64[H,W]: the index 1..n of every frontier cell's local frontier, 0 elsewhere).
    """
    occ = np.asarray(occ)
    olabel, _ = openings(occ)
    unk = occ == UNKNOWN
    rr, cc = np.nonzero(olabel > 0)
    cell_unit = np.zeros(occ.shape, np.int64)
    part = np.zeros(occ.shape, np.int64)
    if rr.size == 0:
        return cell_unit, 0, part
    key = olabel[rr, cc].astype(np.int64)
    if clusters:
        cid = np.zeros(occ.shape, np.int64)
        for i, c in enumerate(clusters):
            if c["cells"]:
                x = np.asarray(c["cells"], np.int64).reshape(-1, 2)
                cid[x[:, 0], x[:, 1]] = i + 1
        key = key * (len(clusters) + 1) + cid[rr, cc]
    _, piece = np.unique(key, return_inverse=True)
    n = int(piece.max()) + 1
    lab = _majority(np.stack([rr, cc], 1).astype(np.int64), piece.ravel(), n, grid, unk)
    cell_unit[rr, cc] = lab[piece.ravel()]
    part[rr, cc] = piece.ravel() + 1
    return cell_unit, int(np.count_nonzero(lab == 0)), part


def assigned_parts(clusters: Sequence[Dict], mask: np.ndarray) -> Tuple[np.ndarray, List[Optional[Cell]]]:
    """
    The share of a robot's frontier clusters (grid_map.fast_frontier_clusters, the selector's
    candidates; a cluster is a radius-10 ball of frontier cells and can hold the openings of
    several zones) in `mask`, a bool grid of the chosen frontier cells (the strategy: the local
    frontiers nearest to the robot's piece).

    Returns (flags bool[n]: the cluster has at least one cell in the mask; reps: for a flagged
    cluster the cell to drive to, i.e. its own representative when that cell is in the mask,
    else the representative of its masked part by fast_frontier_clusters' rule (the part's cell
    nearest the part's centroid, first in the cluster's cell order on ties); None for an
    unflagged cluster).
    """
    n = len(clusters)
    flags = np.zeros(n, bool)
    reps: List[Optional[Cell]] = [None] * n
    if n == 0:
        return flags, reps
    m = np.asarray(mask, bool)
    sizes = np.fromiter((len(c["cells"]) for c in clusters), dtype=np.int64, count=n)
    if sizes.sum() == 0:
        return flags, reps
    cells = np.array([p for c in clusters for p in c["cells"]], dtype=np.int64).reshape(-1, 2)
    owner = np.repeat(np.arange(n), sizes)
    hit = m[cells[:, 0], cells[:, 1]]
    flags = np.bincount(owner[hit], minlength=n) > 0
    starts = np.concatenate(([0], np.cumsum(sizes)[:-1]))
    for i in np.flatnonzero(flags).tolist():
        r0, c0 = (int(v) for v in clusters[i]["rep"])
        if m[r0, c0]:
            reps[i] = (r0, c0)
            continue
        s = int(starts[i])
        part = cells[s:s + int(sizes[i])][hit[s:s + int(sizes[i])]]
        d2 = (part[:, 0] - part[:, 0].mean()) ** 2 + (part[:, 1] - part[:, 1].mean()) ** 2
        j = int(np.argmin(d2))
        reps[i] = (int(part[j, 0]), int(part[j, 1]))
    return flags, reps


# ---------------------------------------------------------------------------
# Pieces: corridor continued + sensor-view pieces
# ---------------------------------------------------------------------------
#
# A port of the offline reference (split_corridor2, corridor_strips, _pieces_connected, with its
# lanes and skeleton helpers), with the two exceptions in the module docstring: pieces after the
# 26th of one unit are named "aa", "ab", ... (letter_of), and a part < A_min touching no piece is
# dropped.

_NB8_ORDER = ((-1, 0), (-1, 1), (0, 1), (1, 1), (1, 0), (1, -1), (0, -1), (-1, -1))


def letter_of(i: int) -> str:
    """0 -> "a", 25 -> "z", 26 -> "aa", 27 -> "ab", ... (bijective base 26)."""
    s = ""
    i = int(i) + 1
    while i > 0:
        i, r = divmod(i - 1, 26)
        s = chr(ord("a") + r) + s
    return s


def letter_index(s: str) -> int:
    """Inverse of letter_of."""
    n = 0
    for ch in s:
        n = n * 26 + (ord(ch) - ord("a") + 1)
    return n - 1


def _sk_degree(sk: np.ndarray) -> np.ndarray:
    p = np.pad(sk, 1).astype(np.int8)
    H, W = sk.shape
    return sum(p[1 + dr:H + 1 + dr, 1 + dc:W + 1 + dc] for dr, dc in _NB8_ORDER)


def _sk_nbrs(sk: np.ndarray, cell: Cell) -> List[Cell]:
    H, W = sk.shape
    r, c = cell
    return [(r + dr, c + dc) for dr, dc in _NB8_ORDER
            if 0 <= r + dr < H and 0 <= c + dc < W and sk[r + dr, c + dc]]


def _adjacent8(a: Cell, b: Cell) -> bool:
    return max(abs(a[0] - b[0]), abs(a[1] - b[1])) == 1


def _clusters8(cells: List[Cell]) -> List[List[Cell]]:
    """8-connected clusters of a small set of cells."""
    out: List[List[Cell]] = []
    for c in cells:
        hit = [g for g in out if any(_adjacent8(c, x) for x in g)]
        merged = [c] + [x for g in hit for x in g]
        out = [g for g in out if g not in hit] + [merged]
    return out


def skeleton_ends(sk: np.ndarray) -> List[Cell]:
    """Skeleton ends in raster order: degree-1 pixels, and degree-2 pixels whose two
    neighbours are 4-adjacent (the tip of a small triangle)."""
    deg = _sk_degree(sk)
    out = []
    for r, c in np.argwhere(sk & ((deg == 1) | (deg == 2))):
        r, c = int(r), int(c)
        if deg[r, c] == 1:
            out.append((r, c))
        else:
            a, b = _sk_nbrs(sk, (r, c))
            if abs(a[0] - b[0]) + abs(a[1] - b[1]) == 1:
                out.append((r, c))
    return out


def walk_back(sk: np.ndarray, e: Cell, k: int = TANGENT_K) -> List[Cell]:
    """Branch pixels from a skeleton end e: [e, s_-1, ..., s_-m], m <= k (stops at a junction)."""
    path = [e]
    seen = {e}
    cur = e
    for _ in range(k):
        cand = [n for n in _sk_nbrs(sk, cur) if n not in seen]
        if not cand:
            break
        groups = _clusters8(cand)
        if len(groups) > 1:                    # junction: stop here
            break
        g = groups[0]
        if len(g) > 3:                         # a blob of skeleton pixels: also a junction
            break
        ref = path[-2] if len(path) > 1 else cur
        nxt = max(g, key=lambda x: ((x[0] - ref[0]) ** 2 + (x[1] - ref[1]) ** 2, -x[0], -x[1]))
        if len(g) > 1:
            ext = [x for x in g if any(n not in seen and n not in g and n != cur
                                       for n in _sk_nbrs(sk, x))]
            if len(ext) > 1 and not all(_adjacent8(ext[0], x) for x in ext[1:]):
                break
            if ext:
                nxt = max(ext, key=lambda x: ((x[0] - ref[0]) ** 2 + (x[1] - ref[1]) ** 2,
                                              -x[0], -x[1]))
        seen.update(g)
        path.append(nxt)
        cur = nxt
    return path


def traverse(start: Cell, t: Tuple[float, float], shape):
    """Cells hit by the half-line from the centre of `start` along t (row, col), in order,
    `start` excluded; 4-connected (Amanatides & Woo). Stops at the grid border (and after
    100000 cells, as the reference)."""
    H, W = shape
    r, c = start
    tr, tc = float(t[0]), float(t[1])
    sr = 1 if tr > 0 else -1
    sc = 1 if tc > 0 else -1
    inf = math.inf
    t_max_r = 0.5 / abs(tr) if tr != 0 else inf
    t_max_c = 0.5 / abs(tc) if tc != 0 else inf
    dt_r = 1.0 / abs(tr) if tr != 0 else inf
    dt_c = 1.0 / abs(tc) if tc != 0 else inf
    n = 0
    while n < 100000:
        if t_max_r <= t_max_c + 1e-12:         # corner: row step first
            r += sr
            t_max_r += dt_r
        else:
            c += sc
            t_max_c += dt_c
        if not (0 <= r < H and 0 <= c < W):
            return
        n += 1
        yield (r, c)


def corridor_lanes(occ: np.ndarray, olabel: np.ndarray, sk: np.ndarray) -> List[Dict]:
    """Every corridor end of the seen map that runs into an opening, continued straight:
    a skeleton end (sk: skeleton_of(occ), the Lee skeleton of the known free space) whose own
    clearance circle touches an opening casts its direction (from TANGENT_K skeleton steps
    back) as a straight line; the line crosses known free space, enters the unknown through a
    frontier cell of opening j and runs until the unknown ends (a known cell or the map edge).
    [{"opening": j, "cells": [unknown cells from the entry], "e": end}]"""
    occ = np.asarray(occ)
    olabel = np.asarray(olabel, np.int32)
    if not (olabel > 0).any():
        return []
    H, W = occ.shape
    dist_o = ndimage.distance_transform_edt(olabel == 0)
    clr = ndimage.distance_transform_edt(np.pad(occ == FREE, 1))[1:-1, 1:-1]
    out = []
    for e in skeleton_ends(sk):
        if dist_o[e] > clr[e]:
            continue
        path = walk_back(sk, e, TANGENT_K)
        s = path[-1]
        d = (e[0] - s[0], e[1] - s[1])
        nrm = math.hypot(*d)
        if nrm == 0:
            continue
        t = (d[0] / nrm, d[1] / nrm)
        prev = e
        cells: List[Cell] = []
        j = 0
        for cell in traverse(e, t, (H, W)):
            v = occ[cell]
            if not cells:
                if v == FREE:
                    prev = cell
                    continue
                if v == OCC:
                    break
                j = int(olabel[prev])          # prev: free, 4-adjacent to this unknown cell
                if j == 0:
                    break
            if v != UNKNOWN:
                break
            cells.append(cell)
        if cells:
            out.append({"opening": j, "cells": cells, "e": e})
    return out


def corridor_strips(occ: np.ndarray, olabel: np.ndarray, sk: np.ndarray
                    ) -> Dict[int, Tuple[np.ndarray, List[Cell], float]]:
    """For every opening a corridor of the seen map runs into: the corridor continued straight
    into the unknown with its own width, {opening: (unknown cells within the half-width of the
    line, line cells, half-width)}; half-width = the skeleton end's clearance (distance to the
    nearest seen wall; the reference's transform, which is not wall_distance's inf when no wall
    has been seen). One per opening: the widest corridor (first in raster order on ties).
    sk: skeleton_of(occ)."""
    occ = np.asarray(occ)
    clr = ndimage.distance_transform_edt(occ != OCC)
    best: Dict[int, Tuple[List[Cell], float]] = {}
    for ln in corridor_lanes(occ, olabel, sk):
        j = ln["opening"]
        hw = float(clr[ln["e"]])
        if j and (j not in best or hw > best[j][1]):
            best[j] = (ln["cells"], hw)
    out = {}
    unk = occ == UNKNOWN
    for j, (cells, hw) in best.items():
        line = np.zeros(occ.shape, bool)
        rc = np.asarray(cells)
        line[rc[:, 0], rc[:, 1]] = True
        near = ndimage.distance_transform_edt(~line) <= hw
        out[j] = (near & unk, cells, hw)
    return out


@dataclass
class PieceSplit:
    """split_pieces() output: raw piece ids (1..), before tracking."""
    grid: np.ndarray                    # piece id per unknown cell, 0 = none
    parent: Dict[int, int]              # piece id -> unit label (of the grid split)
    order: Dict[int, int]               # piece id -> naming rank in its unit (0 = "a")
    lines: Dict[int, List[Cell]]        # continued-corridor piece -> its centre line (unknown cells)
    op_piece: Dict[int, int]            # opening -> piece id at that way in


def split_pieces(occ: np.ndarray, olabel: np.ndarray, units: np.ndarray, op_unit: Dict[int, int],
                 A_min: int, cost_all: np.ndarray, sk: np.ndarray) -> PieceSplit:
    """
    The user's rule (2026-09-25, "corridor continued + sensor-view pieces"; reference
    skelx/pieces.split_corridor2). A unit bigger than one sensor view (A_VIEW cells) is split:
      1. corridor: if a corridor of the seen map runs into one of its ways in
         (corridor_strips), the corridor continued straight ahead with its own width is ONE
         piece -- only its part joined to the way in, and only if it has >= A_min cells;
      2. the rest of the unit is cut into sensor-view pieces: k = ceil(|rest| / A_VIEW)
         centres, the first is the way in, each next one the rest's cell farthest (straight
         line) from the ways in, the corridor and the centres chosen so far, plus one in any
         part of the rest that no seed reaches; every cell goes to its nearest centre by the
         spread's walking cost (octile steps, 1 + w / D per cell), never across a corridor;
      3. a leftover piece thinner than a corridor piece it touches joins it; cells nothing
         reached join the nearest piece of the unit;
      4. safety net: a piece in several 8-connected parts becomes several pieces; parts under
         A_min join the touching piece of the same unit with the longest border (any touching
         piece if none of the same unit touches); a part under A_min that touches no piece is
         dropped, like a unit under A_min with no neighbour in merge_map (the reference keeps
         it as a piece of its own: 7 of 7935 pieces of 348 review syncs, 1-14 cells each);
      5. naming rank (_pieces_connected): "a" = the piece at the way in, then the continued
         corridor(s), then the others (by piece id); the names themselves are display_names.
    `units` / `op_unit`: the merged units and the opening -> unit map (decompose() before
    tracking); cost_all = step_cost(occ), sk = skeleton_of(occ) (decompose computes them once).
    Deterministic; reads only the map.
    """
    occ = np.asarray(occ)
    olabel = np.asarray(olabel, np.int32)
    strips = corridor_strips(occ, olabel, sk)
    out = np.zeros(np.shape(units), dtype=np.int32)
    parent: Dict[int, int] = {}
    op_piece: Dict[int, int] = {}
    lines: Dict[int, List[Cell]] = {}
    nid = 0
    for u in np.unique(units[units > 0]).tolist():
        m = units == u
        n = int(m.sum())
        ops_u = [j for j, v in op_unit.items() if v == u]
        src = np.isin(olabel, ops_u) & (olabel > 0)
        if n <= A_VIEW or not src.any():
            nid += 1
            out[m] = nid
            parent[nid] = u
            for j in ops_u:
                op_piece[j] = nid
            continue
        near_src = ndimage.binary_dilation(src, structure=_EIGHT)
        st = np.zeros_like(m)
        mine = []
        for j in ops_u:
            if j not in strips:
                continue
            s = strips[j][0] & m & ~st
            comp, _ = ndimage.label(s, structure=_EIGHT)
            touch = np.unique(comp[near_src & (comp > 0)])
            if touch.size == 0:
                continue
            s = np.isin(comp, touch)                     # only the part joined to the way in
            if s.sum() >= A_min:
                mine.append((j, s, strips[j][1]))
                st |= s
        rest = m & ~st
        cost = np.where(rest | src, cost_all, np.inf)    # the rest never grows across a corridor
        seeds = np.zeros(occ.shape, np.int32)
        seeds[src & ndimage.binary_dilation(rest, structure=_EIGHT)] = 1
        k = int(math.ceil(rest.sum() / A_VIEW)) if rest.any() else 0
        dmin = ndimage.distance_transform_edt(~(src | st))
        cs: List[Cell] = []
        for _ in range(max(k - 1, 0)):
            d = np.where(rest, dmin, -1.0)
            c = np.unravel_index(int(np.argmax(d)), d.shape)
            if d[c] <= 0:
                break
            cs.append((int(c[0]), int(c[1])))
            one = np.ones_like(m)
            one[c] = False
            dmin = np.minimum(dmin, ndimage.distance_transform_edt(one))
        for i, (r, c) in enumerate(cs):
            seeds[r, c] = i + 2
        comp, nc = ndimage.label(rest, structure=_EIGHT)
        for c in range(1, nc + 1):
            part = comp == c
            if not (seeds[part] > 0).any() and \
                    not (seeds[ndimage.binary_dilation(part, structure=_EIGHT)] == 1).any():
                dd = ndimage.distance_transform_edt(part)
                cc = np.unravel_index(int(np.argmax(dd)), dd.shape)
                cs.append((int(cc[0]), int(cc[1])))
                seeds[cc] = len(cs) + 1
        lab, _ = mcp_labels(cost, seeds)
        first = None
        for v in range(1, len(cs) + 2):
            cells = rest & (lab == v)
            if not cells.any():
                continue
            nid += 1
            out[cells] = nid
            parent[nid] = u
            if v == 1:
                first = nid
        strip_ids = []
        for j, s, line in mine:
            nid += 1
            out[s] = nid
            parent[nid] = u
            lines[nid] = line
            strip_ids.append((nid, strips[j][2]))
        # slivers: a leftover piece thinner than a corridor piece it touches joins it
        sid_set = dict(strip_ids)
        for p in [q for q in list(parent) if parent[q] == u and q not in sid_set]:
            cells = out == p
            if not cells.any():
                continue
            thick = float(ndimage.distance_transform_edt(np.pad(cells, 1))[1:-1, 1:-1].max())
            ring = ndimage.binary_dilation(cells, structure=_EIGHT) & ~cells
            for sid, hw in strip_ids:
                if (out[ring] == sid).any() and thick < hw:
                    out[cells] = sid
                    del parent[p]
                    if first == p:
                        first = None
                    break
        left = m & (out == 0)
        if left.any():                                   # unreached leftovers: nearest piece of the unit
            has = (out > 0) & m
            if has.any():
                _, (ir, ic) = ndimage.distance_transform_edt(~has, return_indices=True)
                out[left] = out[ir[left], ic[left]]
            else:
                nid += 1
                out[left] = nid
                parent[nid] = u
        if first:
            tgt = first
        else:                                            # the first corridor piece of the unit
            corr = [q for q in lines if parent[q] == u]
            tgt = corr[0] if corr and mine else nid
        for j in ops_u:
            op_piece[j] = tgt
    order = _pieces_connected(out, parent, lines, op_piece, A_min)
    return PieceSplit(grid=out, parent=parent, order=order, lines=lines, op_piece=op_piece)


def _pieces_connected(out: np.ndarray, parent: Dict[int, int], lines: Dict[int, List[Cell]],
                      op_piece: Dict[int, int], A_min: int) -> Dict[int, int]:
    """Safety net and naming rank (in place): a piece in several 8-connected parts becomes
    several pieces (the largest part keeps the id); parts under A_min join the touching piece of
    the same unit with the longest (4-adjacency) border, or any touching piece if none of the
    same unit touches (smallest first, ties -> smaller id); a part under A_min that touches no
    piece is dropped (its cells belong to no piece, as merge_map drops a unit under A_min with no
    neighbour); then every unit's pieces are ranked: the piece at the way in, the continued
    corridor(s), the others (each by id); a unit with one piece has rank 0 (and is shown by its
    plain number, display_names). Returns {piece: naming rank}."""
    nid = int(out.max())
    for p in list(parent):
        comp, n = ndimage.label(out == p, structure=_EIGHT)
        if n <= 1:
            continue
        sizes = np.bincount(comp.ravel())[1:]
        keep = int(np.argmax(sizes)) + 1
        for c in range(1, n + 1):
            if c == keep:
                continue
            nid += 1
            out[comp == c] = nid
            parent[nid] = parent[p]
    changed = True
    while changed:
        changed = False
        ids, cnt = np.unique(out[out > 0], return_counts=True)
        size = dict(zip(ids.tolist(), cnt.tolist()))
        border = _borders(out)
        for p in sorted(size, key=lambda k: size[k]):
            if size[p] >= A_min or p not in parent:
                continue
            nbs = {}
            for (a, b), nb in border.items():
                if a == p:
                    nbs[b] = nb
                elif b == p:
                    nbs[a] = nb
            if not nbs:
                out[out == p] = 0                        # touches no piece: dropped
                for j in [j for j, v in op_piece.items() if v == p]:
                    del op_piece[j]
                del parent[p]
                lines.pop(p, None)
                changed = True
                break
            same = {k: v for k, v in nbs.items() if parent.get(k) == parent[p]}
            pool = same or nbs
            t = min(pool, key=lambda k: (-pool[k], k))
            out[out == p] = t
            for j in [j for j, v in op_piece.items() if v == p]:
                op_piece[j] = t
            del parent[p]
            lines.pop(p, None)
            changed = True
            break
    order: Dict[int, int] = {}
    way_in = set(op_piece.values())
    for u in set(parent.values()):
        ps = [p for p in parent if parent[p] == u]
        if len(ps) == 1:
            order[ps[0]] = 0
            continue
        at_way_in = sorted(p for p in ps if p in way_in and p not in lines)
        corridors = sorted(p for p in ps if p in lines and p not in at_way_in)
        others = sorted(p for p in ps if p not in at_way_in and p not in corridors)
        for i, p in enumerate(at_way_in + corridors + others):
            order[p] = i
    return order


def piece_beyond(occ: np.ndarray, olabel: np.ndarray, grid: np.ndarray, parent: Dict[int, int],
                 op_piece: Dict[int, int]) -> Dict[int, Optional[float]]:
    """How far each piece lies beyond its zone's way in, in robot steps (the metric of d_geo,
    grid_map.geodesic: 8-connected, a diagonal step counts 1): the shortest walk from the zone's
    openings through the zone's own unknown cells to the piece's edge = the nearest cell just
    outside the piece (an opening cell or a cell of another piece of the zone). So d_geo to the
    way in + beyond is about a robot's walk to the piece. 0 for a piece at a way in and for a zone
    that is not split; None if the piece cannot be reached inside its zone. (The 1 + w / D price
    of the spread cuts the pieces; it is not used here.)"""
    out: Dict[int, Optional[float]] = {}
    by_unit: Dict[int, List[int]] = {}
    for p, u in parent.items():
        by_unit.setdefault(u, []).append(p)
    H, W = grid.shape
    for u, ps in by_unit.items():
        ways = [j for j, p in op_piece.items() if p in ps]
        if len(ps) == 1:
            out[ps[0]] = 0.0
            continue
        zone = np.isin(grid, ps)
        src = np.isin(olabel, ways) & (olabel > 0) if ways else np.zeros(grid.shape, bool)
        if not src.any():
            for p in ps:
                out[p] = None
            continue
        walk = zone | src
        rr, cc = np.nonzero(walk)
        r0, r1 = max(int(rr.min()) - 1, 0), min(int(rr.max()) + 2, H)
        c0, c1 = max(int(cc.min()) - 1, 0), min(int(cc.max()) + 2, W)
        wk = walk[r0:r1, c0:c1]
        C = geodesic(wk.astype(np.int8), [tuple(x) for x in np.argwhere(src[r0:r1, c0:c1]).tolist()],
                     passable=wk)
        sub_grid = grid[r0:r1, c0:c1]
        for p in ps:
            P = sub_grid == p
            ring = ndimage.binary_dilation(P, structure=_EIGHT) & ~P & wk
            v = C[ring]
            v = v[np.isfinite(v)]
            out[p] = float(v.min()) if v.size else None
    return out


def piece_walk(grid: np.ndarray, members: Sequence[int], k: int) -> np.ndarray:
    """Robot steps (8-connected, a diagonal counts 1) from every cell of a zone to the nearest cell
    of its piece k, walking only through the zone's cells (`members` = the zone's pieces in the
    plan's piece grid `grid`); 0 on k, inf outside the zone. What steers a robot sent to k."""
    grid = np.asarray(grid)
    H, W = grid.shape
    zone = np.isin(grid, list(members))
    out = np.full((H, W), np.inf)
    if not (grid == k).any():
        return out
    rr, cc = np.nonzero(zone)
    r0, r1, c0, c1 = int(rr.min()), int(rr.max()) + 1, int(cc.min()), int(cc.max()) + 1
    z = zone[r0:r1, c0:c1]
    out[r0:r1, c0:c1] = geodesic(z.astype(np.int8), [tuple(x) for x in np.argwhere(grid[r0:r1, c0:c1] == k).tolist()],
                                 passable=z)
    return out


def frontier_walks(part: np.ndarray, cell_unit: np.ndarray, members: Sequence[int], walk: np.ndarray,
                   unknown: np.ndarray) -> np.ndarray:
    """
    How near each of a robot's local frontiers is to piece k: float[n] over the local frontiers
    (part: the index 1..n of every frontier cell's local frontier, local_frontier_units) = the
    least `walk` (piece_walk: steps through k's zone to k) over the frontier's cells' 4-neighbours
    that are unknown on the robot's map; 0 for a frontier into k itself; inf for a frontier not
    labelled one of `members` (the pieces of k's zone, by the majority rule) or not reaching k.
    """
    part = np.asarray(part)
    rr, cc = np.nonzero(part > 0)
    n = int(part.max()) if part.size else 0
    dist = np.full(n, np.inf)
    if rr.size == 0:
        return dist
    owner = part[rr, cc].astype(np.int64) - 1
    lab = np.zeros(n, np.int64)
    lab[owner] = np.asarray(cell_unit)[rr, cc]
    H, W = part.shape
    unk = np.asarray(unknown, bool)
    for dr, dc in _NB4:
        r, c = rr + dr, cc + dc
        ok = (r >= 0) & (r < H) & (c >= 0) & (c < W)
        r, c, o = r[ok], c[ok], owner[ok]
        v = np.where(unk[r, c], walk[r, c], np.inf)
        np.minimum.at(dist, o, v)
    dist[~np.isin(lab, list(members))] = np.inf
    return dist


class PieceTracker(Tracker):
    """
    Persistent region ids of the pieces across a run's replans: the spec's tracker rule (Tracker)
    on the piece grid, whatever the zone: a new piece keeps an old piece's region id if more than
    half of the NEW piece lies in it (largest overlap wins; ties: smaller raw id); every other
    piece gets a fresh id; ids are never reused. Only ids are tracked, not names: names are given
    anew at every plan (split_pieces' naming rank, display_names), so "a" is always the piece at
    the way in. The ids let the next plan's "current pieces" show a robot's piece under its name
    of that plan (or as gone).
    """


def display_names(piece_unit: Dict[int, int], letters: Dict[int, str]) -> Dict[int, str]:
    """{region id: name without prefix}: "<zone><letter>", or "<zone>" for a zone with one piece
    (letters: split_pieces' naming rank of the piece, as a letter). The one place piece names
    are made."""
    count: Dict[int, int] = {}
    for u in piece_unit.values():
        count[u] = count.get(u, 0) + 1
    return {p: (str(u) if count[u] == 1 else f"{u}{letters[p]}") for p, u in piece_unit.items()}


# ---------------------------------------------------------------------------
# Decompose
# ---------------------------------------------------------------------------

@dataclass
class Decomposition:
    """One decomposition of a merged map (see decompose()). `pieces` holds region ids (tracked
    if a PieceTracker was used, else the raw piece ids of split_pieces); `labels` is the zone of
    each cell's piece (0 where no piece), and `opening_unit` the zone of the piece at each
    opening. Names are given anew for every map (display_names); only the region ids persist."""
    labels: np.ndarray                  # zone id per unknown cell (tracked if a Tracker was used), 0 = none
    olabel: np.ndarray                  # opening index per frontier cell, 0 = none
    opening_unit: Dict[int, int]        # opening index -> zone id in `labels` (0 = no zone)
    w: float                            # passage width used by the spread
    label_raw: np.ndarray               # the merged units (opening indices), before tracking and pieces
    pieces: np.ndarray                  # region id per unknown cell, 0 = none
    piece_unit: Dict[int, int]          # region id -> zone id
    piece_name: Dict[int, str]          # region id -> "9a" / "9" (no prefix)
    piece_letter: Dict[int, str]        # region id -> letter of its naming rank (also for "9")
    piece_line: Dict[int, List[Cell]]   # continued corridor -> centre line
    piece_beyond: Dict[int, Optional[float]]   # robot steps past the way in
    op_piece: Dict[int, int]            # opening -> region id at that way in
    seconds: float = 0.0


def decompose(occ: np.ndarray, A_min: int = A_MIN_DEFAULT, tracker: Optional[Tracker] = None,
              piece_tracker: Optional[PieceTracker] = None) -> Decomposition:
    """Openings, the spread, the A_min merge, (with a tracker) persistent zone ids and the
    pieces: split_pieces() on the merged, untracked units, so the pieces are a function of the
    map alone; their zones are the tracked unit ids, their names display_names of this map's
    naming rank ("<zone><letter>", "a" at the way in), and with a piece_tracker their region ids
    persist (PieceTracker). The wall distance, the skeleton, the passage width and the step
    cost are computed once and shared by the spread and the split."""
    t0 = time.perf_counter()
    occ = np.asarray(occ).astype(np.int8)
    olabel, ops = openings(occ)
    D = wall_distance(occ)
    sk = skeleton_of(occ)
    w = passage_width(occ, D, sk)
    cost_all = step_cost(occ, D, w)
    raw, parent = merge_map(spread(occ, olabel, cost_all), int(A_min))
    op_raw: Dict[int, int] = {}
    for o in ops:
        j = o["id"]
        if j in parent:
            op_raw[j] = parent[j]
        else:                                  # U_j empty before the merge: majority rule
            op_raw[j] = frontier_label(raw, o["cells"])
    mp = None
    if tracker is not None:
        tracker.track(raw)
        mp = tracker.last_map
    sp = split_pieces(occ, olabel, raw, op_raw, int(A_min), cost_all, sk)
    unit_of = (lambda u: int(mp.get(u, 0))) if mp is not None else int
    pu_raw = {p: unit_of(u) for p, u in sp.parent.items()}
    beyond_raw = piece_beyond(occ, olabel, sp.grid, sp.parent, sp.op_piece)
    if piece_tracker is not None:
        grid = piece_tracker.track(sp.grid)
        pm = piece_tracker.last_map
    else:
        grid = sp.grid
        pm = {p: p for p in sp.parent}
    # Names are display_names of this map's naming rank (sp.order), with the tracked zone ids.
    letters = {pm[p]: letter_of(sp.order[p]) for p in sp.parent}
    piece_unit = {pm[p]: pu_raw[p] for p in sp.parent}
    # The zone of every cell / opening follows its piece (a tiny part may have joined a piece
    # of a neighbouring unit in the safety net).
    lut = np.zeros(int(grid.max()) + 1 if grid.size else 1, np.int32)
    for r, u in piece_unit.items():
        lut[r] = u
    op_unit = {j: (piece_unit[pm[p]] if p in pm else 0) for j, p in sp.op_piece.items()}
    op_unit.update({o["id"]: 0 for o in ops if o["id"] not in op_unit})
    return Decomposition(labels=lut[grid], olabel=olabel, opening_unit=op_unit, w=float(w), label_raw=raw,
                         pieces=grid, piece_unit=piece_unit, piece_name=display_names(piece_unit, letters),
                         piece_letter=letters, piece_line={pm[p]: v for p, v in sp.lines.items() if p in pm},
                         piece_beyond={pm[p]: v for p, v in beyond_raw.items() if p in pm},
                         op_piece={j: pm[p] for j, p in sp.op_piece.items() if p in pm},
                         seconds=time.perf_counter() - t0)


# ---------------------------------------------------------------------------
# Descriptors
# ---------------------------------------------------------------------------

def descriptors(occ: np.ndarray, labels: np.ndarray, olabel: np.ndarray,
                opening_unit: Dict[int, int], goals: Sequence[Optional[Cell]]) -> Dict[int, Dict]:
    """
    Per zone id k of `labels`: {"size": |U_k|, "width": |O_k| (cells of the openings that
    belong to k), "openings": [opening indices], "d_geo": [steps from each goal to the nearest
    O_k cell through known free space, 8-connected as the robots move (grid_map.geodesic); None
    for a None goal, inf if unreachable]}. goals = each robot's committed waypoint (else its
    position).
    """
    occ = np.asarray(occ)
    labels = np.asarray(labels, np.int32)
    olabel = np.asarray(olabel, np.int32)
    ids, cnt = np.unique(labels[labels > 0], return_counts=True)
    out: Dict[int, Dict] = {int(k): {"size": int(n), "width": 0, "openings": [], "d_geo": []}
                            for k, n in zip(ids.tolist(), cnt.tolist())}
    ocnt = np.bincount(olabel.ravel(), minlength=int(olabel.max()) + 1)
    for j, k in sorted(opening_unit.items()):
        if k in out:
            out[k]["openings"].append(int(j))
            out[k]["width"] += int(ocnt[j]) if j < len(ocnt) else 0
    # Distances: one geodesic per goal, reduced over each unit's opening cells at once.
    if out:
        lut = np.zeros(int(olabel.max()) + 1, np.int32)
        for j, k in opening_unit.items():
            if k in out and j < len(lut):
                lut[j] = k
        ogrid = lut[olabel]                     # unit id on its openings' cells
        keys = sorted(out)
    for g in goals:
        if g is None:
            for d in out.values():
                d["d_geo"].append(None)
            continue
        if not out:
            continue
        dist = geodesic(occ, [tuple(int(v) for v in g)])
        mins = ndimage.minimum(dist, ogrid, index=keys)
        for k, v in zip(keys, np.atleast_1d(mins)):
            v = float(v)
            out[k]["d_geo"].append(v if np.isfinite(v) else math.inf)
    return out


def piece_descriptors(dec: "Decomposition") -> Dict[int, Dict]:
    """
    Per piece (region id r of dec.pieces): {"name": "9a" (no prefix; "9" for a zone with one
    piece), "unit": zone id, "letter", "size": unknown cells, "beyond": robot steps from the
    zone's way in to the piece (piece_beyond; 0 at a way in; None if not reachable inside the zone),
    "corridor": a continued corridor, "line": its centre line (continued corridors),
    "way_in": the piece behind one of the zone's openings}. The zone's own descriptors (way-in
    width, d_geo) are descriptors() of dec.labels.
    """
    grid = np.asarray(dec.pieces, np.int32)
    ids, cnt = np.unique(grid[grid > 0], return_counts=True)
    way_in = set(dec.op_piece.values())
    out: Dict[int, Dict] = {}
    for r, n in zip(ids.tolist(), cnt.tolist()):
        line = dec.piece_line.get(r)
        out[r] = {"name": dec.piece_name[r], "unit": int(dec.piece_unit[r]), "letter": dec.piece_letter[r],
                  "size": int(n), "beyond": dec.piece_beyond.get(r), "corridor": line is not None,
                  "line": [[int(a), int(b)] for a, b in line] if line is not None else None,
                  "way_in": r in way_in}
    return out


def piece_anchor(mask: np.ndarray) -> Cell:
    """Where a piece's id label goes: among the cells deepest inside `mask` (max distance to its
    outside), the one nearest the piece's centroid (ties: first in raster order). On a strip (a
    continued corridor) that is the middle of the strip, not one of its ends."""
    rr, cc = np.nonzero(mask)
    if rr.size == 0:
        return (-1, -1)
    r0, r1, c0, c1 = rr.min(), rr.max() + 1, cc.min(), cc.max() + 1
    sub = np.pad(mask[r0:r1, c0:c1], 1)
    edt = ndimage.distance_transform_edt(sub)[1:-1, 1:-1]
    dr, dc = np.nonzero(edt == edt.max())
    d2 = (dr + r0 - rr.mean()) ** 2 + (dc + c0 - cc.mean()) ** 2
    j = int(np.argmin(d2))
    return int(dr[j] + r0), int(dc[j] + c0)
