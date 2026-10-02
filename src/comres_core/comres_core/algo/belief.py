"""
M2 of COMRES: the target belief b_t over unknown cells.

Exploration is search with an uninformative prior, so both tasks keep a belief: it
is uniform over the unknown cells until the planner supplies hypotheses
{(U_j, p_j)}, and then

    b(g)  proportional to  eps + sum_j p_j 1[g in U_j] / |U_j|,  g in G_un,

where eps is a per-cell floor, compared against the
per-cell hypothesis density p_j / |U_j|. It keeps mass outside the hypothesised
pieces, so a wrong guess only slows the team down instead of making the rest of the
map worth nothing. The p_j are normalised to sum to 1 (they are a distribution).

The support of a hypothesis is whatever mask the plan carries for it: the piece itself (a
piece of a zone, the unknown space behind one way in; units.py). Below, Omega_j (the support)
is that piece.

The per-step update ("newly observed cells are set to zero and b is renormalised")
needs no state beyond the prior: b_t = prior * 1[unknown now], renormalised, because
cells only ever go from unknown to known. `dist(occ)` computes exactly that, so a
robot calls it with its own local map whenever it selects a frontier, and every
robot's belief derives from the same plan prior.

`visible_mass` is B(f) of the selector: belief mass within the sensor range of a
frontier representative, with exact Euclidean-disc semantics ||g - p|| <= d_det.

Grid convention: (row, col); -1 unknown, 0 free, 1 occupied.
"""

from typing import Iterable, Optional, Sequence, Tuple

import numpy as np

UNKNOWN = -1


class Belief:
    """One robot's (or the planner's) belief. Holds only the prior of the last reset."""

    def __init__(self, params):
        self.params = params
        self.prior: Optional[np.ndarray] = None   # None -> uniform over current unknown

    def reset(self, occ: np.ndarray, hypotheses: Optional[Sequence[Tuple[np.ndarray, float]]] = None):
        """
        New prior over the current unknown cells of `occ`.

        Each Omega_j is intersected with the current unknown cells first, so a
        hypothesis carries exactly its share p_j even when this robot has explored
        part of Omega_j since the plan was made. Hypotheses whose Omega has no unknown
        cell left, or whose p_j is <= 0 or not finite, carry no mass and the rest are
        renormalised; if no Omega has an unknown cell left the prior is uniform, and if
        every remaining p is 0 the remaining hypotheses share the mass evenly.
        """
        unk = occ == UNKNOWN
        if not hypotheses:
            self.prior = None
            return
        n_un = int(np.count_nonzero(unk))
        if n_un == 0:
            self.prior = None
            return
        kept = []
        for mask, p in hypotheses:
            if mask is None:
                continue
            m = np.asarray(mask, dtype=bool) & unk
            size = int(np.count_nonzero(m))
            if size:
                p = float(p)
                kept.append((m, size, p if np.isfinite(p) and p > 0 else 0.0))
        if not kept:
            self.prior = None
            return
        total = sum(p for _, _, p in kept)
        if total <= 0:                      # hypotheses named but no weights: split evenly
            kept = [(m, s, 1.0) for m, s, _ in kept]
            total = float(len(kept))
        eps = float(self.params.eps)
        prior = np.zeros(occ.shape, dtype=np.float64)
        prior[unk] = eps
        for m, size, p in kept:
            if p > 0:
                prior[m] += (p / total) / size
        self.prior = prior

    def dist(self, occ: np.ndarray) -> np.ndarray:
        """b_t = prior * 1[occ == -1], normalised; all zeros if nothing is unknown.

        If the prior has no mass left on the unknown cells (eps = 0 and every Omega
        explored), the belief falls back to uniform rather than to all zeros.
        """
        unk = occ == UNKNOWN
        if self.prior is not None and self.prior.shape == occ.shape:
            b = np.where(unk, self.prior, 0.0)
            s = b.sum()
            if s > 0:
                b /= s
                return b
        n = np.count_nonzero(unk)
        if n == 0:
            return np.zeros(occ.shape, dtype=np.float64)
        return unk * (1.0 / n)


_DISC_CACHE = {}


def _disc(d_det) -> np.ndarray:
    """0/1 float (2R+1)^2 mask of integer offsets with dr^2 + dc^2 <= d_det^2 (exact)."""
    key = float(d_det)
    k = _DISC_CACHE.get(key)
    if k is None:
        R = int(np.floor(key))
        o = np.arange(-R, R + 1)
        d2 = o[:, None] ** 2 + o[None, :] ** 2
        k = (d2 <= key * key).astype(np.float64)   # 0/1 weights for the einsum
        _DISC_CACHE[key] = k
    return k


def visible_mass(bmap: np.ndarray, points: Iterable[Tuple[int, int]], d_det) -> np.ndarray:
    """
    B(p) = sum of bmap over the cells g with ||g - p|| <= d_det, for each point.

    Only the windows around the points are touched (a (2R+1)^2 view per point of a
    zero-padded copy, then one einsum with the disc mask), which is cheaper than a
    full-map convolution for the few dozen reps a selection needs. Points off the
    grid are fine: cells outside the grid simply contribute nothing.
    """
    pts = np.asarray(list(points) if not isinstance(points, np.ndarray) else points,
                     dtype=np.int64).reshape(-1, 2)
    out = np.zeros(len(pts), dtype=np.float64)
    if len(pts) == 0 or d_det < 0:
        return out
    disc = _disc(d_det)
    R = disc.shape[0] // 2
    H, W = bmap.shape
    pad = np.zeros((H + 2 * R, W + 2 * R), dtype=np.float64)
    pad[R:R + H, R:R + W] = bmap
    # For an in-grid point (r, c), the window's top-left in padded coordinates is (r, c).
    on = (pts[:, 0] >= 0) & (pts[:, 0] < H) & (pts[:, 1] >= 0) & (pts[:, 1] < W)
    if on.any():
        win = np.lib.stride_tricks.sliding_window_view(pad, disc.shape)
        p = pts[on]
        out[on] = np.einsum("nij,ij->n", win[p[:, 0], p[:, 1]], disc)
    # Rare path (points off the grid): slice each window.
    for n in np.flatnonzero(~on).tolist():
        r, c = pts[n].tolist()
        r0, r1 = r - R, r + R + 1
        c0, c1 = c - R, c + R + 1
        rr0, rr1 = max(r0, 0), min(r1, H)
        cc0, cc1 = max(c0, 0), min(c1, W)
        if rr0 >= rr1 or cc0 >= cc1:
            continue
        out[n] = float((bmap[rr0:rr1, cc0:cc1]
                        * disc[rr0 - r0:rr1 - r0, cc0 - c0:cc1 - c0]).sum())
    return out
