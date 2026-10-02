"""
COMRES in the simulator: a slow planner every T_H steps, a fast frontier selector inside
each robot's own step (docs/comres_spec.md).

The strategy never hands out waypoints (`assign_waypoints` returns {}). Each robot gets its
targets only from the selector (M4), installed as `robot.target_select_fn`, so it runs right
after the robot senses whenever the robot has no target (arrival, or its target was given up
as unreachable). The planner only biases that choice, through the robot's piece (alpha term)
and its belief (beta term); a stale plan can never steer a robot directly.

Communication: the planner sees the merged map, a robot only its own. At a
sync step (step % T_H == 0, the hungarian baseline's cadence) the maps are fused and written
back into every robot, every robot's committed target becomes the shared C, and the planner is
queried (at most one query in flight). Nothing else is exchanged, except the plan when it is
released. T_H comes from Params (run --set T_H=n), the same interval the hungarian
baseline fuses and solves at in that run.

Decomposition: every sync decomposes the merged map into zones (units.py: the unknown space
closest to each opening, tracked: a new zone keeps an old id if more than half of the NEW zone
overlaps it) and cuts every zone bigger than one sensor view into pieces (units.split_pieces:
the corridor continued from a way in plus sensor-view pieces), named anew by the rule ("a" at the way in, then continued corridors, then the
others), with tracked region ids (units.PieceTracker). The snapshot carries zones and pieces;
plans name pieces ("Z9a"; the plan maps names to region ids); lists, hypotheses and the
robots' grid are over region ids. The belief's hypothesis support is the piece masks.

Steering: between merges a robot labels each of its local frontiers -- the openings of its OWN
map (8-connected groups of its frontier cells), cut where its frontier clusters meet, so each
part is one candidate's share of one opening -- with the piece holding most of the part's
adjacent unknown cells in the delivered plan's grid (units.local_frontier_units). A robot assigned piece P of zone U, among its local frontiers
labelled any piece of U, prefers the ones nearest to P: their unknown neighbours' walk to P
through U's cells as the plan saw them (units.piece_walk, units.frontier_walks; a frontier into
P itself is at 0), among the frontiers it may take (not near another robot's goal, not recently
unreachable: the selector's exclusions; if every nearest one is excluded, the next nearest). A
frontier cluster (the selector's candidate) holding such a frontier is in R_i*, and its
representative moves onto that frontier (units.assigned_parts); the selector's assignment term
is 1[cluster in R_i*]. So a robot whose map shows no frontier into P is led toward P, not just
anywhere into U, and a second robot sent to P takes the next nearest frontier
(n_rstar_via_zone counts the checks where no chosen frontier is labelled P). No frontier into U:
no assignment term (n_list_exhausted). Each robot holds one piece (a one-piece list). The
current pieces shown to the next plan name a robot's piece that
still exists by its name of that sync, and a gone one as gone (comres_planner.GONE), never
by an old name.

Ablation variants (`variant`, comres_planner.VARIANTS; search with a hint only): the planner's
question changes (comres_planner, comres_points); here no_history keeps the robots' current
pieces and the belief out of the snapshot, and no_decomp skips the decomposition, plans goal
points and scores frontiers by their closeness to the robot's goal while its goal cell is unknown
on the robot's own map (selector.select(goal=...); counters n_with_goal, n_goal_active,
n_goal_seen).

Planners (`planner_kind`):
  vlm               M3 (comres_planner.VLMPlanner)
  hungarian_region  classical baseline: Hungarian assignment of robots to pieces with
                    d = d_geo(w_i, the zone's nearest way in) + beyond (robot steps from the way
                    in to the piece, the same metric) and U = the piece's size
  none              greedy: the selector alone
  oracle            diagnostic, not a baseline and not a ceiling (search): p = 1 on the piece
                    holding the target, listed for every robot; it shows what a perfect label
                    does with these pieces, NOT an upper bound (all robots go to one piece, and
                    with a small eps a p = 1 guess on a large piece barely moves the belief)

Latency: a plan's compute is timed -- the decomposition and tracking,
the belief map, the planner's own CPU part (table, d_geo, rendering, prompt, parsing) and the
VLM call inside its lock (the lock wait is never charged), or the Hungarian solve -- and the
plan is released ceil(t / dt) steps later (at least one step if t > 0). Map fusion
(communication) and the robots' own selections are not charged: selection is robot-local,
like the baselines' frontier fallback. Its compute is still measured
(`counts["select_seconds"]`) so a run can show it is negligible.

Order within a simulator step: every robot senses, selects if it has no target, and moves;
then `assign_waypoints` delivers a plan whose release step has come and, at a sync step,
syncs and queries. So at step 0 the robots select with an empty plan and the first query runs
after their first move. `attach(robots)` must be called right after the robots are spawned
(run does) so step 0 already uses the selector.
"""

import math
import time
import traceback
from typing import Dict, List, Optional, Sequence, Tuple

import numpy as np
from scipy import ndimage
from scipy.optimize import linear_sum_assignment

from units import ID_PREFIX
from grid_map import fast_frontier_clusters, fuse
from params import Params
from waypoint_strategy import WaypointStrategy

PLANNER_KINDS = ("vlm", "hungarian_region", "none", "oracle")
UNREACHABLE_MEMORY = 50        # steps a given-up target stays excluded from selection
_REC_DROP = ("table",)          # the planner's table: too big for a result row


# --- small helpers ---------------------------------------------------------------
def _jsonable(v):
    if isinstance(v, dict):
        return {str(k): _jsonable(x) for k, x in v.items()}
    if isinstance(v, (list, tuple)):
        return [_jsonable(x) for x in v]
    if isinstance(v, np.generic):
        return v.item()
    if isinstance(v, np.ndarray):
        return None if v.size > 64 else v.tolist()
    if isinstance(v, (str, int, float, bool)) or v is None:
        return v
    return str(v)


def hungarian_region_lists(D: np.ndarray, U: np.ndarray, ids: Sequence, params: Params) -> Dict[int, List]:
    """
    Classical assignment of robots to pieces: cost_ik = lam * d~_ik - beta * U~_k, with d~ =
    d over the largest finite d of the pieces (per robot) and U~ = U over the largest U; an
    unreachable pair costs as if twice the farthest piece. The Hungarian solution (columns
    repeated when there are fewer pieces than robots) gives each robot its piece, as a one-piece
    list. ids: one per column (region ids or piece names).
    """
    n_r, n_k = D.shape
    if n_k == 0:
        return {}
    Dn = np.empty_like(D)
    for i in range(n_r):
        fin = D[i][np.isfinite(D[i])]
        m = fin.max() if fin.size else 0.0
        Dn[i] = np.where(np.isfinite(D[i]), D[i] / m if m > 0 else 0.0, 2.0)
    Un = U / U.max() if U.size and U.max() > 0 else np.zeros_like(U)
    cost = params.lam * Dn - params.beta * Un[None, :]
    reps = max(1, math.ceil(n_r / n_k))
    rows, cols = linear_sum_assignment(np.tile(cost, (1, reps)))
    first = {int(i): int(j) % n_k for i, j in zip(rows, cols)}
    return {i: [ids[first[i]]] for i in range(n_r)}


# --- the strategy ----------------------------------------------------------------
class COMRESStrategy(WaypointStrategy):
    """planner_kind in PLANNER_KINDS. Robots select through `target_select_fn`."""

    def __init__(self, planner_kind: str = "vlm", task: str = "search", hint: Optional[str] = None,
                 params: Optional[Params] = None,
                 n_robots: Optional[int] = None, latency_enabled: bool = True, planner=None,
                 target: Optional[Tuple[int, int]] = None, variant: str = "full"):
        super().__init__()                            # consulted every step; gates on T_H itself
        from comres_planner import VARIANTS
        if planner_kind not in PLANNER_KINDS:
            raise ValueError(f"planner_kind must be one of {PLANNER_KINDS}, got {planner_kind!r}")
        if task not in ("explore", "search"):
            raise ValueError(f"task must be explore or search, got {task!r}")
        if variant not in VARIANTS:
            raise ValueError(f"variant must be one of {VARIANTS}, got {variant!r}")
        if variant != "full" and (planner_kind != "vlm" or task != "search" or not hint):
            raise ValueError(f"the {variant} variant is a VLM arm for search with a hint only")
        self.planner_kind = planner_kind
        # "full" (the method) or an ablation variant (comres_planner.VARIANTS): no_history keeps the
        # previous plan out of the snapshot; no_decomp plans goal points (comres_points.py).
        self.variant = variant
        self.task = task
        self.hint = hint if task == "search" else None
        self.params = (params if params is not None else Params()).for_task(task)
        if planner_kind == "oracle" and task != "search":
            raise ValueError("the oracle planner is a search diagnostic (it needs a target)")
        # oracle only: the true target cell (never shown to anyone else).
        self.target = tuple(int(v) for v in target) if target is not None else None
        self.n_robots = n_robots
        self.latency_enabled = latency_enabled
        self.dt = float(self.params.dt)
        self.T_H = int(self.params.T_H)
        self.T_coord = self.T_H                       # the simulator times steps % T_coord == 0
        self.planner = planner                        # injectable (tests); else built on attach
        self.meta: Dict = {}                          # map_variant / unit_id / arm, for snapshots
        self._clear()

    def _clear(self) -> None:
        self.robots: List = []
        self._index: Dict[int, int] = {}
        self.team_start: Optional[List[Tuple[int, int]]] = None
        # (release_step, plan, record); run loops test `_pending is not None`.
        self._pending: Optional[Tuple[int, object, Dict]] = None
        self.plan_times: List[float] = []        # charged seconds per query
        self.plan_delays: List[int] = []         # steps each plan was withheld
        self.plan_records: List[Dict] = []
        self.errors: List[str] = []              # tracebacks of assign_waypoints failures
        self.stale_steps = 0
        self.n_syncs = 0
        self.n_skipped_in_flight = 0
        # Current plan, as delivered: region ids per robot, hypotheses, the piece grid, each
        # piece's zone, each zone's pieces and the name the plan showed for each piece.
        self.lists: Dict[int, List[int]] = {}
        self.hyps: List[Tuple[int, float]] = []
        self._plan_grid: Optional[np.ndarray] = None
        self._piece_unit: Dict[int, int] = {}
        self._zone_pieces: Dict[int, List[int]] = {}     # zone -> its pieces (region ids) in the plan
        self._walks: Dict[int, np.ndarray] = {}          # piece -> steps to it through its zone (lazy)
        self._plan_names: Dict[int, str] = {}
        # Decomposition across syncs: zone ids (Tracker) and piece region ids (PieceTracker).
        self.global_tracker = None
        self.piece_tracker = None
        self.global_units = None                      # units.Decomposition of the last sync
        self.C: Dict[int, Optional[Tuple[int, int]]] = {}
        self._merged: Optional[np.ndarray] = None     # merged map of the last sync
        self.beliefs: Dict[int, object] = {}
        self.team_belief = None
        self.goals: Dict[int, Tuple[int, int]] = {}      # no_decomp: each robot's goal cell
        self.counts = {
            "select_calls": 0,           # selector callbacks (robot had no target)
            "n_selections": 0,           # ... that scored candidates (had a frontier)
            "n_no_frontier": 0, "n_select_none": 0,
            "n_with_list": 0,            # robot held a piece
            "n_with_rstar": 0,           # ... whose zone has one of its local frontiers
            "n_list_exhausted": 0,       # piece held, its zone not on the robot's map any more
            "n_shown_ids_gone": 0,       # current lists shown to the planner: listed pieces gone
            "n_chose_assigned": 0,       # chosen frontier inside R_i*
            "n_assign_changed": 0,       # alpha term changed the argmax
            "n_excluded_by_C": 0,        # selections where C removed a candidate
            "n_excluded_blocked": 0,     # ... where a recently unreachable point did
            "n_exclusions_ignored": 0, "n_used_euclid": 0,
            "n_belief_resets": 0,
            # Local frontiers (summed over the checks) whose adjacent unknown cells carry no piece
            # in the plan's grid, and R_i* clusters whose representative was moved onto the
            # chosen frontier.
            "n_pieces_no_unit": 0, "n_rep_moved": 0,
            # Steered list checks where none of the chosen frontiers (those of the piece's zone
            # nearest to the piece) is labelled the piece itself.
            "n_rstar_via_zone": 0,
            # Robot-local compute, not charged (seconds, summed over robots).
            "select_seconds": 0.0, "select_seconds_frontiers": 0.0,
            "select_seconds_belief": 0.0, "select_seconds_selector": 0.0, "select_seconds_units": 0.0,
        }
        if self.variant == "no_decomp":
            # Selections of a robot holding a goal point: its goal cell still unknown on its own
            # map (goal term used) or already seen (no plan term until the next plan).
            self.counts.update(n_with_goal=0, n_goal_active=0, n_goal_seen=0)

    def reset(self) -> None:
        super().reset()
        self._clear()

    # -- interface run loops rely on -----------------------------------------
    def attach(self, robots: List) -> None:
        """Install the selector on every robot. Call right after spawning."""
        from belief import Belief
        self.robots = list(robots)
        self._index = {id(r): i for i, r in enumerate(robots)}
        if self.n_robots is None:
            self.n_robots = len(robots)
        if self.team_start is None:
            self.team_start = [tuple(map(int, r.position)) for r in robots]
        for i, r in enumerate(robots):
            r.target_select_fn = self._select
            self.beliefs[i] = Belief(self.params)         # uniform until a plan's hypotheses
        self.team_belief = Belief(self.params)
        if self.planner is None and self.planner_kind == "vlm":
            from comres_planner import VLMPlanner
            self.planner = VLMPlanner(self.task, self.hint or "", self.n_robots, self.params,
                                      variant=self.variant)

    def stats(self) -> Dict:
        n = len(self.plan_times)
        return {
            "dt": self.dt, "latency_enabled": self.latency_enabled, "n_plans": n,
            "mean_delay_steps": round(sum(self.plan_delays) / len(self.plan_delays), 2)
            if self.plan_delays else 0.0,
            "stale_steps": self.stale_steps, "n_syncs": self.n_syncs,
            "n_skipped_in_flight": self.n_skipped_in_flight,
            "n_plans_valid": sum(1 for r in self.plan_records if r.get("has_plan")),
            # Failed calls or unparseable replies (a dead server turns the arm into greedy).
            "n_plan_errors": sum(1 for r in self.plan_records if r.get("error")),
            # ... of which the call itself failed (no reply at all: no finish_reason).
            "n_call_errors": sum(1 for r in self.plan_records
                                 if r.get("error") and r.get("finish_reason") is None),
            "n_plans_released": sum(1 for r in self.plan_records if r.get("released_step") is not None),
            # VLM search plans with no hypotheses (allowed; the beliefs reset to uniform).
            "n_plans_no_hyps": sum(1 for r in self.plan_records if r.get("has_plan") and not r.get("plan_hyps"))
            if (self.task == "search" and self.planner_kind == "vlm") else 0,
        }

    def selection_counters(self) -> Dict:
        c = {k: (round(v, 4) if isinstance(v, float) else v) for k, v in self.counts.items()}
        c["n_unreachable"] = sum(len(getattr(r, "unreachable", [])) for r in self.robots)
        return c

    # -- selector callback (robot-local, not charged) --------------------------
    def _assigned_flags(self, i: int, occ: np.ndarray, clusters: List[Dict], avoid=None
                        ) -> Tuple[Optional[np.ndarray], List[Dict]]:
        """(bool per frontier cluster of robot i's own map, True = in R_i*, or None for no
        assignment term; the clusters to score, with a flagged cluster's representative moved
        onto the chosen frontier). See _piece_flags."""
        lst = self.lists.get(i)
        if not lst:
            return None, clusters
        c = self.counts
        c["n_with_list"] += 1
        t0 = time.perf_counter()
        try:
            grid = self._plan_grid
            if grid is None or grid.shape != occ.shape:
                c["n_list_exhausted"] += 1
                return None, clusters
            return self._piece_flags(i, occ, clusters, lst[0], grid, avoid)
        finally:
            c["select_seconds_units"] += time.perf_counter() - t0

    def _walk(self, k: int) -> np.ndarray:
        """Steps from every cell of piece k's zone to k through the zone (the delivered plan's
        grid; units.piece_walk), computed once per plan and piece."""
        w = self._walks.get(k)
        if w is None:
            from units import piece_walk
            z = self._piece_unit.get(k)
            w = piece_walk(self._plan_grid, self._zone_pieces.get(z, [k]), k)
            self._walks[k] = w
        return w

    def _piece_flags(self, i: int, occ: np.ndarray, clusters: List[Dict], k: int,
                     grid: np.ndarray, avoid=None) -> Tuple[Optional[np.ndarray], List[Dict]]:
        """Pieces: robot i's local frontiers (the openings of its own map, cut at its clusters'
        borders) are labelled by the spec's majority rule in the plan's piece grid. If the zone
        of the robot's piece k labels one of them, the flagged frontiers are those of k's zone
        nearest to k (steps through the zone's cells as the plan saw them; a frontier into k
        itself is at 0; units.frontier_walks) among those the robot may take: a level whose
        clusters are all excluded by `avoid(rep)` (the selector's exclusions: near another
        robot's goal, or recently unreachable) gives way to the next nearest. So a robot whose
        map shows no frontier into k is led toward k, not just into its zone, and a second robot
        sent to k takes the next nearest frontier. A cluster is flagged when it holds one of the
        chosen frontiers, and its representative moves onto it."""
        from units import assigned_parts, frontier_walks, local_frontier_units
        c = self.counts
        cell_unit, n_none, part = local_frontier_units(occ, grid, clusters)
        c["n_pieces_no_unit"] += n_none
        present = set(np.unique(cell_unit[cell_unit > 0]).tolist())
        z = self._piece_unit.get(k)
        members = self._zone_pieces.get(z, [k]) if z is not None else [k]
        best = None
        if k > 0 and present.intersection(members):
            dist = frontier_walks(part, cell_unit, members, self._walk(k), occ == -1)
            for dv in np.unique(dist[np.isfinite(dist)]).tolist():
                mask = np.isin(part, np.flatnonzero(dist == dv) + 1)
                flags, reps = assigned_parts(clusters, mask)
                if not flags.any():
                    continue
                if best is None:
                    best = (mask, flags, reps)
                if avoid is None or not all(avoid(reps[j]) for j in np.flatnonzero(flags).tolist()):
                    best = (mask, flags, reps)
                    break
        if best is None:
            c["n_list_exhausted"] += 1
            return None, clusters
        mask, flags, reps = best
        c["n_with_rstar"] += 1
        # Steered through another piece of the zone: no chosen frontier is labelled k.
        c["n_rstar_via_zone"] += not bool((cell_unit[mask] == k).any())
        out = list(clusters)
        for j in np.flatnonzero(flags).tolist():
            if tuple(reps[j]) != tuple(int(v) for v in clusters[j]["rep"]):
                out[j] = {**clusters[j], "rep": reps[j]}
                c["n_rep_moved"] += 1
        return flags, out

    def _select(self, robot) -> Optional[Tuple[int, int]]:
        t0 = time.perf_counter()
        try:
            return self._select_inner(robot)
        finally:
            self.counts["select_calls"] += 1
            self.counts["select_seconds"] += time.perf_counter() - t0

    def _select_inner(self, robot) -> Optional[Tuple[int, int]]:
        from selector import select
        c = self.counts
        i = self._index.get(id(robot))
        occ = robot.local_map
        pos = tuple(int(v) for v in robot.position)
        t0 = time.perf_counter()
        clusters = fast_frontier_clusters(occ)
        c["select_seconds_frontiers"] += time.perf_counter() - t0
        if not clusters:
            c["n_no_frontier"] += 1
            return None
        flags = goal = None
        others = [w for j, w in self.C.items() if j != i and w is not None]
        now = getattr(robot, "steps_taken", 0)
        blocked = [p for p, t in getattr(robot, "unreachable", []) if now - t <= UNREACHABLE_MEMORY]
        if i is not None:
            from selector import BLOCKED_RADIUS, _near_any
            r_C = float(self.params.r_C)

            def avoid(rep):
                p = np.asarray([rep], np.float64)
                return bool(_near_any(p, others, r_C)[0] or _near_any(p, blocked, BLOCKED_RADIUS)[0])

            if self.variant == "no_decomp":
                goal = self._active_goal(i, occ)
            else:
                flags, clusters = self._assigned_flags(i, occ, clusters, avoid)
        t1 = time.perf_counter()
        belief = self.beliefs.get(i)
        if belief is not None:
            bmap = belief.dist(occ)
        else:
            unk = occ == -1
            bmap = unk / max(1, int(unk.sum()))
        t2 = time.perf_counter()
        c["select_seconds_belief"] += t2 - t1
        idx, diag = select(pos, occ, clusters, bmap, self.params, assigned_clusters=flags,
                           committed_others=others, blocked=blocked, goal=goal)
        c["select_seconds_selector"] += time.perf_counter() - t2
        c["n_selections"] += 1
        c["n_assign_changed"] += bool(diag.get("assign_changed"))
        c["n_excluded_by_C"] += bool(diag.get("n_excluded_C"))
        c["n_excluded_blocked"] += bool(diag.get("n_excluded_blocked"))
        c["n_exclusions_ignored"] += bool(diag.get("exclusions_ignored"))
        c["n_used_euclid"] += bool(diag.get("used_euclid"))
        if idx is None:
            c["n_select_none"] += 1
            return None
        cand, in_R = diag.get("candidates"), diag.get("in_R")
        if flags is not None and cand is not None and in_R is not None:
            hit = np.flatnonzero(np.asarray(cand) == idx)
            c["n_chose_assigned"] += bool(hit.size and in_R[hit[0]])
        return tuple(int(v) for v in clusters[int(idx)]["rep"])

    def _active_goal(self, i: int, occ: np.ndarray) -> Optional[Tuple[int, int]]:
        """no_decomp: robot i's goal cell while it is still unknown on the robot's own map, else
        None (seen: no plan term until the next plan)."""
        g = self.goals.get(i)
        if g is None:
            return None
        self.counts["n_with_goal"] += 1
        if occ[g] == -1:
            self.counts["n_goal_active"] += 1
            return g
        self.counts["n_goal_seen"] += 1
        return None

    # -- plan delivery -------------------------------------------------------
    def _deliver(self, plan, rec: Dict, step: int) -> None:
        rec["released_step"] = int(step)
        if plan is None:
            return
        if self.variant == "no_decomp":
            self._deliver_points(plan)
            return
        # Piece names ("Z9a") -> the region ids of the plan's grid.
        ids = plan.ids
        self._piece_unit = {int(r): int(z) for r, z in plan.piece_unit.items()}
        self._zone_pieces = {}
        for r, z in sorted(self._piece_unit.items()):
            self._zone_pieces.setdefault(z, []).append(r)
        self._walks = {}
        self._plan_names = {int(r): str(u) for u, r in ids.items()}
        self.lists = {int(k): [ids[u] for u in v] for k, v in plan.lists.items() if v}
        self.hyps = [(ids[u], float(p)) for u, p in plan.hyps]
        omegas = {ids[k]: m for k, m in plan.omegas.items()}
        hyps = [(omegas[k], p) for k, p in self.hyps if k in omegas] or None
        # Every robot resets its belief over its own unknown cells from the same prior.
        for i, r in enumerate(self.robots):
            self.beliefs[i].reset(r.local_map, hyps)
        self.counts["n_belief_resets"] += len(self.robots)
        if self._merged is not None:                 # the planner's view: last merged map
            self.team_belief.reset(self._merged, hyps)
        # The plan's piece grid is what the robots label their frontiers with until the next
        # plan is delivered.
        self._plan_grid = plan.labels

    def _deliver_points(self, plan) -> None:
        """no_decomp: the robots' goal cells, and every belief reset from the point hypotheses
        (support: the unknown cells within d_det of each point, comres_points.point_support)."""
        self.goals = dict(plan.goals)
        self.hyps = list(plan.hyps)
        hyps = [(m, p) for m, (_, p) in zip(plan.omegas, plan.hyps)] or None
        for i, r in enumerate(self.robots):
            self.beliefs[i].reset(r.local_map, hyps)
        self.counts["n_belief_resets"] += len(self.robots)
        if self._merged is not None:
            self.team_belief.reset(self._merged, hyps)

    # -- sync and planning ---------------------------------------------------
    def _sync(self, robots: List, step: int) -> None:
        self.n_syncs += 1
        occ = fuse([r.local_map for r in robots])
        for r in robots:
            r.local_map = occ.copy()
        self._merged = occ
        # C: every robot's committed target as of this sync (its selected frontier).
        self.C = {}
        for i, r in enumerate(robots):
            w = r.waypoints[r.current_waypoint_idx] if r.current_waypoint_idx < len(r.waypoints) \
                else r._fallback_waypoint
            self.C[i] = tuple(map(int, w)) if w is not None else None
        if self.planner_kind == "none":
            return
        query = self.planner_kind != "none" and self._pending is None
        if self.planner_kind != "none" and not query:
            self.n_skipped_in_flight += 1      # at most one query in flight
        if self.variant == "no_decomp":              # no zones or pieces: the plain merged map
            if query:
                t0 = time.perf_counter()
                snap = self._snapshot(occ, robots, step, None)
                self._query(snap, step, time.perf_counter() - t0)
            return
        t0 = time.perf_counter()
        # The trackers follow every sync, in flight or not, so zone and piece ids stay stable
        # across the T_H-step gaps.
        from units import PieceTracker, Tracker, decompose
        if self.global_tracker is None:
            self.global_tracker = Tracker()
        if self.piece_tracker is None:
            self.piece_tracker = PieceTracker()
        dec = decompose(occ, int(self.params.A_min), self.global_tracker, self.piece_tracker)
        self.global_units = dec
        if not query:
            return
        snap = self._snapshot(occ, robots, step, dec)
        t_prep = time.perf_counter() - t0
        self._query(snap, step, t_prep)

    def _snapshot(self, occ: np.ndarray, robots: List, step: int, dec):
        """M3's Snapshot of this sync: what the planner may see (never the target). The ablation
        variants see less: no_history neither the robots' current pieces nor the belief (both
        carry the previous plan); no_decomp (dec None) no zones or pieces."""
        from comres_planner import Snapshot
        from units import descriptors, piece_descriptors
        history = self.variant != "no_history" and dec is not None
        bmap = self.team_belief.dist(occ) if (self.task == "search" and history) else None
        pos = [tuple(map(int, r.position)) for r in robots]
        wps = [self.C.get(i) for i in range(len(robots))]
        common = dict(task=self.task, hint=self.hint or "", step=int(step), occ=occ, robot_pos=pos,
                      waypoints=wps, team_start=list(self.team_start or []), belief_map=bmap,
                      map_variant=str(self.meta.get("map_variant", "")),
                      unit_id=str(self.meta.get("unit_id", "")),
                      meta={k: v for k, v in self.meta.items() if k not in ("map_variant", "unit_id")})
        if dec is None:
            return Snapshot(lists=[], **common)
        goals = [w if w is not None else q for w, q in zip(wps, pos)]
        desc = descriptors(occ, dec.labels, dec.olabel, dec.opening_unit, goals)
        return Snapshot(lists=self._current_lists(len(robots), dec) if history else [],
                        unit_labels=dec.labels, olabel=dec.olabel, opening_unit=dec.opening_unit,
                        unit_desc=desc, passage_width=dec.w, piece_labels=dec.pieces,
                        piece_desc=piece_descriptors(dec), op_piece=dec.op_piece, **common)

    def _current_lists(self, n: int, dec) -> List[List[str]]:
        """
        Each robot's current pieces as the planner reads them. Names are given anew for every
        map, so a listed piece is shown only under its name of THIS sync (its region id is
        tracked, units.PieceTracker); a piece that no longer exists is shown as GONE, never
        under an old name, so every name in a prompt means one place of that prompt's table.
        """
        from comres_planner import GONE
        live = {r: f"{ID_PREFIX}{nm}" for r, nm in dec.piece_name.items()}
        out: List[List[str]] = []
        for i in range(n):
            shown = []
            for r in self.lists.get(i, []):
                if r in live:
                    shown.append(live[r])
                else:
                    self.counts["n_shown_ids_gone"] += 1
                    shown.append(GONE)
            out.append(shown)
        return out

    def _query(self, snap, step: int, t_prep: float) -> None:
        """Run the planner on `snap`, charge its compute, schedule the release."""
        t0 = time.perf_counter()
        if self.planner_kind in ("hungarian_region", "oracle"):
            plan, log = (self._hungarian_region if self.planner_kind == "hungarian_region"
                         else self._oracle)(snap, step)
            planner_s, call_s, lock_wait = time.perf_counter() - t0, 0.0, 0.0
        else:
            plan, log = self.planner.plan(snap)
            log = dict(log)
            call_s = float(log.get("call_seconds") or 0.0)
            lock_wait = float(log.get("lock_wait_seconds") or 0.0)
            planner_s = float(log["cpu_seconds"])   # its CPU part (units, render, prompt, parse)
        charged = t_prep + planner_s + call_s
        delay = math.ceil(charged / self.dt) if (charged > 0 and self.latency_enabled) else 0
        self.plan_times.append(charged)
        self.plan_delays.append(delay)

        rec = {k: _jsonable(v) for k, v in log.items() if k not in _REC_DROP}
        rec.update({
            "query_step": int(step), "release_step": int(step + delay), "released_step": None,
            "compute_seconds": round(charged, 4), "prep_cpu_seconds": round(t_prep, 4),
            "planner_cpu_seconds": round(planner_s, 4), "call_seconds": round(call_s, 4),
            "lock_wait_seconds": round(lock_wait, 4), "delay_steps": delay,
            "n_open": rec.get("n_open"),
            **self._plan_fields(plan),
            "has_plan": plan is not None,
        })
        if snap.has_pieces:
            rec["n_units"] = len(np.unique(snap.unit_labels[snap.unit_labels > 0]))
            rec["passage_width"] = snap.passage_width
            rec["n_pieces"] = len(snap.piece_desc)
            rec["n_corridors"] = sum(bool(d.get("corridor")) for d in snap.piece_desc.values())
            rec["decomp_seconds"] = round(self.global_units.seconds, 4) if self.global_units is not None else None
        else:                                          # no_decomp: no zones or pieces
            rec.update(n_units=None, passage_width=None, n_pieces=None, n_corridors=None, decomp_seconds=None)
        self.plan_records.append(rec)
        if plan is None:
            return
        if delay <= 0:
            self._deliver(plan, rec, step)
        else:
            self._pending = (step + delay, plan, rec)

    def _plan_fields(self, plan) -> Dict:
        """A plan as its record shows it: the pieces per robot and the hypotheses (piece, p); for
        no_decomp the goal cells [row, col] per robot and the hypotheses ([row, col], p)."""
        if self.variant == "no_decomp":
            return {"plan_lists": None,
                    "plan_goals": {str(k): [int(v[0]), int(v[1])] for k, v in plan.goals.items()}
                    if plan is not None else None,
                    "plan_hyps": [[[int(c[0]), int(c[1])], float(p)] for c, p in plan.hyps]
                    if plan is not None else None}
        return {"plan_lists": _jsonable({str(k): v for k, v in plan.lists.items()}) if plan is not None else None,
                "plan_hyps": _jsonable(plan.hyps) if plan is not None else None}

    def _piece_table(self, snap):
        """Pieces of a pieces snapshot: (Units of the planner, region ids, sizes, cost matrix
        robots x pieces). d(i, k) = d_geo from robot i's goal to the nearest way in of piece k's
        zone (through explored space) + beyond_k (robot steps from that way in to the piece);
        inf if either is unknown."""
        from comres_planner import prepare_units
        U_ = prepare_units(snap)
        rids = list(U_.open_labels)
        n = len(snap.robot_pos)
        size = np.array([float(r["size"]) for r in U_.rows])
        D = np.full((n, len(rids)), np.inf)
        for k, r in enumerate(U_.rows):              # dgeo = zone's d_geo + beyond (prepare_units)
            for i, d in enumerate(r["dgeo"][:n]):
                if d is not None:
                    D[i, k] = float(d)
        return U_, rids, size, D

    def _hungarian_region(self, snap, step: int):
        """Classical planner over the pieces (charged like the VLM): cost_ik = lam * d~_ik - beta *
        |P_k|~ with d_ik = d_geo(w_i, way in of k's zone) + beyond_k, Hungarian for each robot's
        piece (hungarian_region_lists), no hypotheses."""
        from comres_planner import Plan
        U_, rids, size, D = self._piece_table(snap)
        if not rids:
            return None, {"planner": "hungarian_region", "skipped": "no_open_unit", "n_open": 0}
        lists = hungarian_region_lists(D, size, rids, self.params)
        name = {r: u for u, r in U_.ids.items()}
        plan = Plan(lists={i: [name[k] for k in l] for i, l in lists.items()}, hyps=[], omegas=U_.masks,
                    labels=U_.labels, query_step=int(step), ids=dict(U_.ids), piece_unit=dict(U_.piece_unit))
        return plan, {"planner": "hungarian_region", "n_open": len(rids)}

    def _oracle(self, snap, step: int):
        """Diagnostic plan with a perfect label (search): p = 1 on the piece holding the target,
        that piece listed for every robot. Not an upper bound on hint use. If the target cell
        carries no piece (already seen this step, or in unknown space no opening leads to), the
        piece nearest to it (Euclidean) is used."""
        from comres_planner import Plan, prepare_units
        if self.target is None:
            raise RuntimeError("the oracle planner needs the target (COMRESStrategy(target=...))")
        U_ = prepare_units(snap)
        grid = U_.labels
        if not U_.open_ids:
            return None, {"planner": "oracle", "skipped": "no_open_unit", "n_open": 0}
        r, c = self.target
        k, how = int(grid[r, c]), "contains"
        if k == 0:
            _, (ir, ic) = ndimage.distance_transform_edt(grid == 0, return_indices=True)
            k, how = int(grid[ir[r, c], ic[r, c]]), "nearest"
        name = {rid: u for u, rid in U_.ids.items()}
        u = name[k]
        mask = U_.masks[u]
        plan = Plan(lists={i: [u] for i in range(len(snap.robot_pos))}, hyps=[(u, 1.0)],
                    omegas={u: mask}, labels=grid,
                    query_step=int(step), ids=dict(U_.ids), piece_unit=dict(U_.piece_unit))
        return plan, {"planner": "oracle", "n_open": len(U_.open_ids), "target_unit": k,
                      "target_piece": u, "target_zone": U_.piece_unit[k],
                      "target_how": how, "target_unit_size": int(mask.sum())}

    # -- strategy interface --------------------------------------------------
    def assign_waypoints(self, robots: List, step_count: int, **kwargs) -> Dict:
        try:
            if not self._index or len(self._index) != len(robots) or \
                    any(id(r) not in self._index for r in robots):
                self.attach(robots)            # late attach: earlier steps used the plain fallback
            if self._pending is not None:
                release, plan, rec = self._pending
                if step_count >= release:
                    self._pending = None
                    self._deliver(plan, rec, step_count)
                else:
                    self.stale_steps += 1
            if step_count % self.T_H == 0:
                self._sync(robots, step_count)
        except Exception:
            # The simulator swallows strategy exceptions; keep them for the run loop.
            self.errors.append(traceback.format_exc())
            raise
        return {}
