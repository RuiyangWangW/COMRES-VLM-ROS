"""
Shared grid-map utilities for COMRES: frontier mask, frontier clusters,
geodesic distances, map fusion.

The selector re-reads frontiers every time a robot needs a target, and the planner
and the zone descriptors need geodesic distances at every sync, so these run
thousands of times per simulated run. robot.extract_frontier_clusters is the
reference definition of a frontier cluster (the baselines use it), but it is pure
Python over every cell; `fast_frontier_clusters` returns exactly the same list --
order, ids, reps, sizes and cell order -- at C speed, so COMRES and the baselines see
identical frontiers and only the speed differs.

Grid convention: (row, col); -1 unknown, 0 free, 1 occupied.
"""

from typing import Dict, Iterable, List, Optional, Sequence, Tuple

import numpy as np
from scipy import ndimage
from skimage.graph import MCP

UNKNOWN, FREE = -1, 0
_CROSS = ndimage.generate_binary_structure(2, 1)      # 4-connectivity


def fuse(maps: Sequence[np.ndarray]) -> np.ndarray:
    """Merged map: elementwise max, so occupied > free > unknown (as the baselines fuse)."""
    maps = list(maps)
    out = maps[0].copy()
    for m in maps[1:]:
        np.maximum(out, m, out=out)
    return out


def frontier_mask(occ: np.ndarray) -> np.ndarray:
    """Free cells with an unknown 4-neighbour (cells off the grid are not unknown)."""
    unk = occ == UNKNOWN
    nb = np.zeros_like(unk)
    nb[1:] |= unk[:-1]
    nb[:-1] |= unk[1:]
    nb[:, 1:] |= unk[:, :-1]
    nb[:, :-1] |= unk[:, 1:]
    return nb & (occ == FREE)


def fast_frontier_clusters(occ: np.ndarray, max_cluster_distance: int = 10) -> List[Dict]:
    """
    Same output as robot.extract_frontier_clusters(occ, -1, 0, max_cluster_distance).

    The reference seeds a cluster at each still-unclaimed frontier cell in raster order
    and runs a FIFO BFS through free cells (4-connected) up to depth
    max_cluster_distance, claiming every unclaimed frontier cell it reaches; frontier
    cells claimed earlier are passed through but not re-claimed. A FIFO BFS with unit
    steps visits exactly the free cells whose 4-connected free-space distance from the
    seed is <= the limit, so the claimed set only depends on that geodesic ball, which
    is computed here with a masked binary dilation inside the (2d+1)^2 window around the
    seed (the ball cannot leave it). The seed is its cluster's first cell in raster
    order, so cluster ids follow raster order of seeds and the reference's dict
    insertion order equals id order.
    """
    H, W = occ.shape
    D = int(max_cluster_distance)
    fr = frontier_mask(occ)
    flat = np.flatnonzero(fr)
    if flat.size == 0:
        return []
    free = occ == FREE
    comp = np.full((H, W), -1, dtype=np.int32)
    cid = 0
    if D < 1:
        comp[fr] = np.arange(flat.size, dtype=np.int32)   # every cell its own cluster
        cid = flat.size
    else:
        comp_flat = comp.ravel()
        for idx in flat.tolist():
            if comp_flat[idx] != -1:
                continue
            r, c = divmod(idx, W)
            r0, r1 = max(r - D, 0), min(r + D + 1, H)
            c0, c1 = max(c - D, 0), min(c + D + 1, W)
            seed = np.zeros((r1 - r0, c1 - c0), dtype=bool)
            seed[r - r0, c - c0] = True
            ball = ndimage.binary_dilation(seed, _CROSS, iterations=D,
                                           mask=free[r0:r1, c0:c1])
            sub = comp[r0:r1, c0:c1]
            sub[ball & fr[r0:r1, c0:c1] & (sub == -1)] = cid
            cid += 1

    # Group cells by cluster, raster order kept within each cluster (stable sort).
    lab = comp.ravel()[flat]
    order = np.argsort(lab, kind="stable")
    flat_s, lab_s = flat[order], lab[order]
    rows, cols = np.divmod(flat_s, W)
    counts = np.bincount(lab_s, minlength=cid)
    starts = np.concatenate(([0], np.cumsum(counts)[:-1]))
    # Integer coordinate sums are exact in float64, so sum / n equals np.mean(xs).
    cr = np.bincount(lab_s, weights=rows, minlength=cid) / counts
    cc = np.bincount(lab_s, weights=cols, minlength=cid) / counts
    d2 = (rows - cr[lab_s]) ** 2 + (cols - cc[lab_s]) ** 2
    dmin = np.minimum.reduceat(d2, starts)
    # Candidates for the rep: cells at (or within rounding of) the minimum. The exact
    # tie-break is redone below with the reference's own Python float expression.
    near = d2 <= dmin[lab_s] * (1 + 1e-9) + 1e-12

    rows_l, cols_l = rows.tolist(), cols.tolist()
    near_l = near.tolist()
    out = []
    for k in range(cid):
        s, n = int(starts[k]), int(counts[k])
        cells = list(zip(rows_l[s:s + n], cols_l[s:s + n]))
        cands = [cells[j] for j in range(n) if near_l[s + j]]
        if len(cands) == 1:
            rep = cands[0]
        else:
            cx, cy = float(cr[k]), float(cc[k])
            rep = min(cands, key=lambda p: (p[0] - cx) ** 2 + (p[1] - cy) ** 2)
        out.append({"id": k, "rep": rep, "size": n, "cells": cells})
    return out


def geodesic(occ: np.ndarray, sources: Iterable[Tuple[int, int]],
             passable: Optional[np.ndarray] = None) -> np.ndarray:
    """
    Steps from the nearest source over passable cells (default: free), 8-connected, with
    a diagonal move costing 1 like the robot's step. float64, inf where unreachable.

    skimage's MCP charges the cost of every cell on the path including the start, so
    with unit costs it returns steps + 1; that 1 is subtracted. Source cells are made
    passable, so a robot standing on a cell its map does not call free still gets
    distances; sources off the grid are ignored.
    """
    H, W = occ.shape
    if passable is None:
        passable = occ == FREE
    costs = np.where(passable, 1.0, np.inf)
    starts = [(int(r), int(c)) for r, c in sources if 0 <= r < H and 0 <= c < W]
    if not starts:
        return np.full((H, W), np.inf)
    for r, c in starts:
        costs[r, c] = 1.0
    cum, _ = MCP(costs, fully_connected=True).find_costs(starts)
    cum -= 1.0
    return cum
