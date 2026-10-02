"""
M4 of COMRES: the frontier selector.

The VLM never outputs waypoints. When a robot needs a target it scores every frontier
cluster f of its own map with

    r(f) = alpha * 1[f in R*] + beta * B~(f) - lam * d~(f)

and commits to the best one's representative. 1[f in R*] is the robot's plan term: the
strategy (comres_strategy._piece_flags) flags the clusters holding the robot's local
frontiers nearest to its assigned piece (the openings of the robot's own map cut at its
clusters' borders, each labelled by the majority of its adjacent unknown cells in the plan's
piece grid) and moves their representatives onto those frontiers, so it passes one flag per
cluster, `assigned_clusters`, and the clusters with their reps. B(f) is the belief mass
within sensor range of the rep (M2), d(f) the geodesic steps from the robot; B~ and d~ are
divided by their maxima over the remaining candidates so the three terms share a [0, 1] scale
and the weights mean the same thing on every map. A stale plan therefore only biases the
choice; with no plan (alpha term zero) and a uniform belief this is the greedy
information-gain-minus-distance baseline. The no_decomp ablation arm replaces 1[f in R*] by
the closeness of the rep to the robot's goal point (`goal`).

Candidates are pruned before scoring: reps the robot cannot reach over the free
cells of its own map, reps within r_C of another robot's committed waypoint (the set
C_i, as of the last sync), and reps within 2 cells of a point this robot recently
failed to reach. If the exclusions would leave nothing, they are relaxed rather than
leaving the robot idle: C first (sharing a frontier with a teammate still makes
progress; driving back to a point A* just gave up on does not), then all of them.
If nothing is reachable, Euclidean distance stands in for geodesic for every cluster
(the map may be about to open a passage the robot cannot see yet; A* may still find
a way through unknown space).

`diag` reports every term per candidate and the choice the robot would have made with
alpha = 0 (same candidates, same normalisation), so the strategy can count how often
the plan changed a decision.

Grid convention: (row, col); -1 unknown, 0 free, 1 occupied.
"""

from typing import Dict, Iterable, Optional, Sequence, Tuple

import numpy as np

from belief import visible_mass
from grid_map import geodesic

BLOCKED_RADIUS = 2.0     # reps this close (Euclidean) to a recently unreachable point are skipped
_TIE_TOL = 1e-12         # scores closer than this count as tied


def _near_any(pts: np.ndarray, others, radius: float) -> np.ndarray:
    """bool[n]: point within `radius` (Euclidean, inclusive) of any of `others`."""
    others = [o for o in others if o is not None]
    if not others or len(pts) == 0:
        return np.zeros(len(pts), dtype=bool)
    o = np.asarray(others, dtype=np.float64).reshape(-1, 2)
    d2 = ((pts[:, None, :] - o[None, :, :]) ** 2).sum(-1)
    return (d2 <= radius * radius + 1e-9).any(axis=1)


def _argmax(score: np.ndarray, d: np.ndarray) -> int:
    """Index of the best score; ties -> smaller d, then lower index."""
    best = score.max()
    tied = np.flatnonzero(score >= best - _TIE_TOL)
    if len(tied) == 1:
        return int(tied[0])
    dmin = d[tied].min()
    return int(tied[d[tied] <= dmin + _TIE_TOL][0])


def select(robot_pos: Tuple[int, int], occ: np.ndarray, clusters: Sequence[Dict],
           bmap: Optional[np.ndarray], params, *, assigned_clusters: Optional[Sequence[bool]] = None,
           committed_others: Iterable = (), blocked: Iterable = (),
           goal: Optional[Tuple[int, int]] = None) -> Tuple[Optional[int], Dict]:
    """
    Pick the frontier cluster robot `robot_pos` should go to next.

    clusters: grid_map.fast_frontier_clusters(occ) (dicts with "rep" and "cells").
    bmap: belief over the grid (Belief.dist(occ)); None means no belief term (B = 0).
    assigned_clusters: one bool per cluster: True = the cluster is in R_i* (None -> the
        alpha term is zero).
    goal: the no_decomp ablation's goal cell instead of the flags: the alpha term is the goal
        closeness A = 1 - dist(rep, goal) / max over the remaining candidates (straight line;
        0 for all when that max is 0). At most one of assigned_clusters and goal.
    committed_others: other robots' committed waypoints (None entries ignored).
    blocked: points this robot recently gave up on as unreachable.

    Returns (index into `clusters`, or None when there are no clusters; diag).
    Per-candidate arrays in diag are aligned with diag["candidates"] (indices into
    `clusters`, ascending).
    """
    n = len(clusters)
    empty_f = np.zeros(0)
    diag = {"n_clusters": n, "n_candidates": 0, "chosen": None, "chosen_no_alpha": None,
            "assign_changed": False,
            "has_assigned": assigned_clusters is not None, "has_goal": goal is not None,
            "chosen_in_assigned": False, "n_in_R": 0,
            "n_unreachable": 0, "used_euclid": False,
            "n_excluded": 0, "n_excluded_C": 0, "n_excluded_blocked": 0,
            "exclusions_ignored": False, "relaxed": "",
            "candidates": np.zeros(0, dtype=np.int64), "in_R": np.zeros(0, dtype=bool),
            "B": empty_f, "B_tilde": empty_f, "d": empty_f, "d_tilde": empty_f,
            "score": empty_f, "score_no_alpha": empty_f, "A": empty_f}
    if n == 0:
        return None, diag

    reps = np.array([c["rep"] for c in clusters], dtype=np.int64).reshape(-1, 2)
    pos = np.asarray(robot_pos, dtype=np.float64)

    # Distances: geodesic over this robot's free cells; unreachable reps dropped.
    dist = geodesic(occ, [tuple(int(v) for v in robot_pos)])
    d_all = dist[reps[:, 0], reps[:, 1]]
    reach = np.isfinite(d_all)
    diag["n_unreachable"] = int(n - reach.sum())
    if not reach.any():
        d_all = np.sqrt(((reps - pos) ** 2).sum(axis=1))
        reach[:] = True
        diag["used_euclid"] = True

    # Exclusions (counted among reachable clusters): other robots' commitments C_i and
    # points this robot recently failed to reach. Relaxed in that order if nothing is left.
    repf = reps.astype(np.float64)
    ex_c = _near_any(repf, committed_others, float(params.r_C)) & reach
    ex_b = _near_any(repf, blocked, BLOCKED_RADIUS) & reach
    diag["n_excluded_C"] = int(ex_c.sum())
    diag["n_excluded_blocked"] = int(ex_b.sum())
    diag["n_excluded"] = int((ex_c | ex_b).sum())
    cand_mask = reach & ~(ex_c | ex_b)
    if not cand_mask.any():
        diag["exclusions_ignored"] = True
        cand_mask = reach & ~ex_b
        diag["relaxed"] = "C"
        if not cand_mask.any():
            cand_mask = reach
            diag["relaxed"] = "all"
    cand = np.flatnonzero(cand_mask)

    # Terms, normalised over the remaining candidates.
    d = d_all[cand]
    if assigned_clusters is not None:
        flags = np.asarray(assigned_clusters, dtype=bool).reshape(-1)
        if len(flags) != n:
            raise ValueError(f"assigned_clusters has {len(flags)} flags for {n} clusters")
        in_R = flags[cand]
    else:
        in_R = np.zeros(len(cand), dtype=bool)
    if bmap is not None:
        B = visible_mass(bmap, reps[cand], params.d_det)
    else:
        B = np.zeros(len(cand))
    Bmax, dmax = B.max(), d.max()
    Bt = B / Bmax if Bmax > 0 else np.zeros_like(B)
    dt = d / dmax if dmax > 0 else np.zeros_like(d)
    base = float(params.beta) * Bt - float(params.lam) * dt
    if goal is None:
        A = empty_f
        score = float(params.alpha) * in_R + base
    else:
        if assigned_clusters is not None:
            raise ValueError("select: pass assigned_clusters or goal, not both")
        dg = np.sqrt(((repf[cand] - np.asarray(goal, dtype=np.float64)) ** 2).sum(axis=1))
        gmax = dg.max()
        A = 1.0 - dg / gmax if gmax > 0 else np.zeros(len(cand))
        score = float(params.alpha) * A + base

    k = _argmax(score, d)
    k0 = _argmax(base, d)               # the counterfactual alpha = 0 choice
    chosen, chosen0 = int(cand[k]), int(cand[k0])
    diag.update({
        "n_candidates": int(len(cand)),
        "candidates": cand, "in_R": in_R, "B": B, "B_tilde": Bt, "d": d, "d_tilde": dt,
        "score": score, "score_no_alpha": base, "A": A, "chosen": chosen, "chosen_no_alpha": chosen0,
        "assign_changed": chosen != chosen0, "n_in_R": int(in_R.sum()),
        "chosen_in_assigned": bool(in_R[k]),
    })
    return chosen, diag
