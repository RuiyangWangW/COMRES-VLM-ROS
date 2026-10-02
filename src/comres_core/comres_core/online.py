"""
Real-time COMRES: the simulator's COMRESStrategy (algo/comres_strategy.py, unchanged) driven by
real maps, poses and goal outcomes. No ROS here; comres_ros feeds it.

A simulator step s is (COMRES-VLM simulator.py, robot.py): each robot senses, drops targets whose
frontier is gone from its own map (the "vanished = reached" rule, up to 4 a step) and, without a
target, asks the COMRES selector for one; it moves; it drops a target it has arrived at (within
1.5 cells of its new cell); then the team: a finished plan is delivered, and every T_H steps the
robots' maps are merged, the merged map becomes every robot's map and the planner is asked (at most
one question in flight). Tick s here does exactly that, regrouped around the robots' motion,
which happens between ticks: it ends step s-1 (arrival at the cells the robots reached, then the
team of step s-1, on the maps sensed at s-1) and begins step s (the maps and cells just received,
then the robots' checks and choices).
What differs is time and the world, where it must:
  * the planner's question runs in a background thread and is delivered at the first tick after
    the answer arrives, so its latency is real (the simulator charged ceil(seconds / dt) steps);
  * a robot's own map between merges is its live SLAM map, with the last merged map filling the
    cells its SLAM does not know (in the simulator the world never changes, so the two agree);
  * the cells within 2 cells of the robot count as seen (free unless its map says wall), as the
    simulated sensor always observes that disc (SENSOR_ALWAYS_OBSERVED_RADIUS);
  * a robot that, staying where it is, picks again and again the target it has just reached
    (REPEAT_LIMIT times; its frontier never clears from its map: a SLAM map that has not caught
    up, glass, a sensor blind spot) avoids it for `arrival_hold` steps and chooses again, instead
    of standing still. A robot crawling along a frontier edge picks the cell beside its last
    target each step, but those are different targets, so this never acts in the simulator
    (tools/shadow_check.py checks it);
  * goal outcomes come from the robot's navigation (report): "reached", "failed" (unreachable:
    avoided for the selector's UNREACHABLE_MEMORY steps, as the simulator gives up a route) or
    "dropped" (cancelled outside COMRES: chosen again freely).
"""

import concurrent.futures
import math
import threading
import time
import traceback
from typing import Callable, Dict, List, Optional, Sequence, Tuple

import numpy as np

from . import ALGO_DIR  # noqa: F401  (importing the package puts the algorithm on sys.path)
import experiment_config  # noqa: E402
from comres_strategy import UNREACHABLE_MEMORY, COMRESStrategy  # noqa: E402
from grid_map import frontier_mask  # noqa: E402
from params import Params  # noqa: E402

from .grid import FREE, UNKNOWN, free_cell_near

Cell = Tuple[int, int]
VANISH_RADIUS = 2.0          # COMRES-VLM robot.VANISH_RADIUS (cells)
ARRIVAL_RADIUS = 1.5         # COMRES-VLM Robot._check_arrival: distance < 1.5 cells
MAX_VANISH_PER_STEP = 4      # COMRES-VLM robot.MAX_VANISH_PER_STEP
ALWAYS_SEEN_RADIUS = experiment_config.SENSOR_ALWAYS_OBSERVED_RADIUS   # cells (2)
REPEAT_LIMIT = 3             # re-picks of a just-reached target, without leaving it, before it is avoided
STAY_RADIUS = 3.0            # cells: still at a reached target (a Nav2 goal moved off a wall, its tolerance)


def frontier_near(occ: np.ndarray, point: Cell, radius: float = VANISH_RADIUS) -> bool:
    """COMRES-VLM Robot.frontier_near: a frontier cell (free, with an unknown 4-neighbour) of
    `occ` within `radius` (Euclidean) of `point`."""
    H, W = occ.shape
    r, c = int(point[0]), int(point[1])
    k = int(math.floor(radius))
    r0, r1 = max(r - k - 1, 0), min(r + k + 2, H)
    c0, c1 = max(c - k - 1, 0), min(c + k + 2, W)
    if r0 >= r1 or c0 >= c1:
        return False
    rr, cc = np.ogrid[r0:r1, c0:c1]
    disk = (rr - r) ** 2 + (cc - c) ** 2 <= radius * radius + 1e-9
    return bool((frontier_mask(occ[r0:r1, c0:c1]) & disk).any())


class RobotProxy:
    """What COMRESStrategy reads from a simulator Robot: its own map, its cell, its start, its
    committed target (`_fallback_waypoint`: COMRES robots only hold targets they picked
    themselves) and the targets it gave up as unreachable, with the tick it gave them up."""

    def __init__(self, name: str, start: Cell, shape: Tuple[int, int]):
        self.name = name
        self.position = np.array(start, dtype=np.int64)
        self.start_position = tuple(int(v) for v in start)
        self.local_map = np.full(shape, -1, dtype=np.int8)
        self.waypoints: List[Cell] = []
        self.current_waypoint_idx = 0
        self._fallback_waypoint: Optional[Cell] = None
        self.unreachable: List[Tuple[Cell, int]] = []
        self.steps_taken = 0
        self.target_select_fn: Optional[Callable] = None
        self.ended: Optional[Tuple[Cell, Cell]] = None    # Mission: (target, robot cell) when a target was reached
        self.streak: Optional[List] = None           # ... [target, re-picks] of a target picked again there

    @property
    def target(self) -> Optional[Cell]:
        return self._fallback_waypoint


class _Answer:
    """Stands in for the planner while a finished answer is handed to COMRESStrategy._query."""

    def __init__(self, result):
        self.result = result

    def plan(self, snap):
        return self.result


class _DaemonPool:
    """One question at a time on a daemon thread, so an unanswered VLM call never keeps the
    process alive at exit."""

    def submit(self, fn, *args) -> concurrent.futures.Future:
        fut: concurrent.futures.Future = concurrent.futures.Future()

        def work():
            try:
                fut.set_result(fn(*args))
            except BaseException as e:               # noqa: BLE001
                fut.set_exception(e)
        threading.Thread(target=work, name="comres-planner", daemon=True).start()
        return fut

    def shutdown(self, **kw) -> None:
        pass


class OnlineStrategy(COMRESStrategy):
    """COMRESStrategy whose planner question runs in a background thread. `_sync` (unchanged)
    merges, decomposes and builds the snapshot; `_query` submits the question; `poll` hands the
    finished answer to the unchanged `_query`, which records and delivers it. While a question is
    in flight `_pending` is set, so `_sync` skips asking, as in the simulator."""

    def __init__(self, *args, **kwargs):
        kwargs["latency_enabled"] = False          # the latency is real: deliver when answered
        super().__init__(*args, **kwargs)
        self._pool = _DaemonPool()
        self._future = None
        self._inflight = None
        self.answer_seconds: List[float] = []
        self.n_asked = 0

    def _query(self, snap, step: int, t_prep: float) -> None:
        if self._future is not None:               # at most one question in flight
            self.n_skipped_in_flight += 1
            return
        self._inflight = (snap, int(step), float(t_prep), time.monotonic())
        self._pending = (None, None, None)          # in flight (read by _sync only)
        self.n_asked += 1
        self._future = self._pool.submit(self._ask, snap)

    def _ask(self, snap):
        """The planner's answer and, unless it skipped, the messages it sent (in its thread)."""
        plan, log = self.planner.plan(snap)
        msgs = None if (log or {}).get("skipped") else getattr(self.planner, "last_messages", None)
        return (plan, log), msgs

    def poll(self, step: int) -> Optional[Dict]:
        """Deliver a finished answer. Returns its record, {"plan_error": traceback} if the planner
        raised, or None (nothing finished)."""
        if self._future is None or not self._future.done():
            return None
        fut, (snap, qstep, t_prep, t0) = self._future, self._inflight
        self._future = self._inflight = None
        self._pending = None
        try:
            result, msgs = fut.result()
        except Exception:                          # noqa: BLE001
            tb = traceback.format_exc()
            self.errors.append(tb)
            return {"plan_error": tb[-1500:], "query_step": qstep}
        self.answer_seconds.append(time.monotonic() - t0)
        real, self.planner = self.planner, _Answer(result)
        try:
            COMRESStrategy._query(self, snap, qstep, t_prep)
        finally:
            self.planner = real
        rec = self.plan_records[-1]
        rec["released_step"] = int(step)
        rec["answer_seconds"] = round(self.answer_seconds[-1], 3)
        rec["_messages"] = msgs                    # what the VLM was given (not logged as is)
        return rec

    def in_flight(self) -> bool:
        return self._future is not None

    def shutdown(self) -> None:
        self._pool.shutdown()


class Mission:
    """
    COMRES for n robots in real time. Per tick: update_robot() for every robot (its SLAM map on
    the planning grid and its current cell), then tick(step), which returns the robots whose
    target changed ({index: new target cell or None}). report(i, outcome) feeds back a goal's
    outcome from the robot's navigation at any time.
    """

    def __init__(self, names: Sequence[str], shape: Tuple[int, int], task: str = "search", hint: str = "",
                 params: Optional[Params] = None, planner=None, arrival_hold: int = 5):
        self.names = list(names)
        self.shape = tuple(shape)
        self.params = params if params is not None else Params()
        self.strategy = OnlineStrategy(planner_kind="vlm", task=task, hint=hint, params=self.params,
                                       n_robots=len(self.names), planner=planner)
        self.robots: List[RobotProxy] = []
        self.slam: List[Optional[np.ndarray]] = []
        self.merged: Optional[np.ndarray] = None
        self.started = False
        self.step = -1
        self.events: List[Dict] = []
        self.arrival_hold = int(arrival_hold)

    @property
    def T_H(self) -> int:
        return self.strategy.T_H

    def start(self, start_cells: Sequence[Cell]) -> None:
        """Begin at the robots' start cells (the team start, as the hints mean it)."""
        self.robots = [RobotProxy(n, c, self.shape) for n, c in zip(self.names, start_cells)]
        self.slam = [None] * len(self.robots)
        self._sensed = [None] * len(self.robots)
        self.strategy.attach(self.robots)
        self.started = True

    def update_robot(self, i: int, slam_grid: np.ndarray, cell: Cell) -> None:
        """Robot i's SLAM map on the planning grid and its current cell, as sensed now; used by the
        next tick (it ends the previous step at this cell, then begins the new one with this map)."""
        g = np.asarray(slam_grid, dtype=np.int8).copy()
        cell = free_cell_near(g, tuple(int(v) for v in cell))
        self._sensed[i] = (g, cell)

    def _apply_sensed(self, i: int) -> None:
        g, cell = self._sensed[i]
        k = ALWAYS_SEEN_RADIUS                     # the always-observed disc around the robot: free
        r0, c0 = max(cell[0] - k, 0), max(cell[1] - k, 0)   # where neither its SLAM nor the merge knows
        rr, cc = np.ogrid[r0:min(cell[0] + k + 1, g.shape[0]), c0:min(cell[1] + k + 1, g.shape[1])]
        disc = np.zeros(g.shape, bool)
        disc[r0:r0 + rr.shape[0], c0:c0 + cc.shape[1]] = (rr - cell[0]) ** 2 + (cc - cell[1]) ** 2 <= k * k
        unk = (g == UNKNOWN) if self.merged is None else ((g == UNKNOWN) & (self.merged == UNKNOWN))
        g = g.copy()
        g[disc & unk] = FREE
        self.slam[i] = g
        r = self.robots[i]
        r.position = np.array(cell, dtype=np.int64)
        r.local_map = g if self.merged is None else np.where(g != UNKNOWN, g, self.merged).astype(np.int8)

    def _event(self, kind: str, i: Optional[int] = None, **kw) -> None:
        self.events.append({"t": round(time.time(), 3), "step": self.step, "kind": kind,
                            **({"robot": self.names[i]} if i is not None else {}), **kw})

    def _hold(self, r: RobotProxy, cell: Cell) -> None:
        """Avoid `cell` for arrival_hold steps (the selector's exclusion list, aged so that it
        expires after arrival_hold of its UNREACHABLE_MEMORY steps)."""
        if self.arrival_hold > 0:
            r.unreachable.append((tuple(cell), r.steps_taken - UNREACHABLE_MEMORY + self.arrival_hold))

    def report(self, i: int, outcome: str, target: Optional[Cell] = None) -> None:
        """A goal outcome from robot i's navigation: "reached", "failed" (unreachable) or
        "dropped" (the goal ended for a reason outside COMRES, e.g. cancelled by an operator).
        `target`: the target the goal was for; an outcome for another target is ignored."""
        r = self.robots[i]
        if r.target is None or (target is not None and tuple(target) != tuple(r.target)):
            return
        cell = self._sensed[i][1] if self._sensed[i] is not None else tuple(int(v) for v in r.position)
        if outcome == "failed" and math.dist(r.target, cell) < ARRIVAL_RADIUS:
            outcome = "reached"                        # the simulator never gives up a target it is at
        if outcome == "failed":
            r.unreachable.append((r.target, r.steps_taken))
        if outcome == "reached":
            r.ended = (tuple(r.target), cell)
        self._event(outcome, i, target=list(r.target))
        r._fallback_waypoint = None

    def _select(self, i: int, r: RobotProxy) -> None:
        r._fallback_waypoint = r.target_select_fn(r)
        t, pos = r.target, tuple(int(v) for v in r.position)
        if t is not None and r.ended is not None and math.dist(t, r.ended[0]) < ARRIVAL_RADIUS \
                and math.dist(pos, r.ended[1]) <= STAY_RADIUS:          # the target just reached, again
            if r.streak is not None and math.dist(t, r.streak[0]) < ARRIVAL_RADIUS:
                r.streak[1] += 1
            else:
                r.streak = [t, 1]
            if r.streak[1] >= REPEAT_LIMIT:            # see the module docstring
                self._event("stuck_frontier", i, target=list(t))
                self._hold(r, t)
                r.streak = None
                r._fallback_waypoint = r.target_select_fn(r)
        else:
            r.streak = None
        if r.target is not None:
            self._event("target", i, target=list(r.target))

    def arrivals(self) -> None:
        """End of a step, after the motion: drop targets the robots have arrived at (within 1.5
        cells of the cells just received); they choose at their next step, as in the simulator."""
        for i, r in enumerate(self.robots):
            old = r.target
            if old is not None and math.dist(old, self._sensed[i][1]) < ARRIVAL_RADIUS:
                self._event("arrived", i, target=list(old))
                r.ended = (tuple(old), self._sensed[i][1])
                r._fallback_waypoint = None

    def robots_phase(self, step: int) -> None:
        """Each robot's step after sensing: drop vanished targets, choose new ones."""
        for i, r in enumerate(self.robots):
            r.steps_taken = int(step) + 1                 # Robot.step_towards_waypoint counts first
            r.unreachable = [(p, t) for p, t in r.unreachable if r.steps_taken - t <= UNREACHABLE_MEMORY]
            if r.target is None:
                self._select(i, r)
            n = 0
            while r.target is not None and n < MAX_VANISH_PER_STEP and not frontier_near(r.local_map, r.target):
                self._event("seen", i, target=list(r.target))
                r._fallback_waypoint = None
                n += 1
                self._select(i, r)

    def team_phase(self, step: int) -> None:
        """Step 2 of a tick (the strategy, after the robots)."""
        st = self.strategy
        rec = st.poll(step)
        if rec is not None and "plan_error" in rec:
            self._event("plan_error", query_step=rec["query_step"], error=rec["plan_error"])
        elif rec is not None:
            self._event("plan", query_step=rec.get("query_step"), plan=rec.get("plan_lists"), hyps=rec.get("plan_hyps"),
                        answer_seconds=rec.get("answer_seconds"), valid=rec.get("valid"), error=rec.get("error"))
        if step % st.T_H == 0:
            for r, g in zip(self.robots, self.slam):
                r.local_map = g                        # the merge is of what the robots know now
            asked = st.n_asked
            st._sync(self.robots, step)                # writes the merged map into every robot
            self.merged = st._merged
            self._event("merge", asked=st.n_asked > asked)

    def tick(self, step: int) -> Dict[int, Optional[Cell]]:
        """End step `self.step` (arrivals, then the team), begin step `step` (the maps and cells
        from update_robot, then the robots). Returns {robot: new target} for changed targets."""
        if not self.started or any(s is None for s in self._sensed):
            return {}
        before = [r.target for r in self.robots]
        if self.step >= 0:
            self.arrivals()
            for i, r in enumerate(self.robots):         # the team sees where the robots moved to
                r.position = np.array(self._sensed[i][1], dtype=np.int64)
            self.team_phase(self.step)
        self.step = int(step)
        for i in range(len(self.robots)):
            self._apply_sensed(i)
        self.robots_phase(step)
        return {i: r.target for i, r in enumerate(self.robots) if r.target != before[i]}

    def stop(self) -> None:
        self.strategy.shutdown()
