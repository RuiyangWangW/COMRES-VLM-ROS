"""
Real-time COMRES: the simulator's COMRESStrategy (algo/comres_strategy.py, unchanged) driven by
real maps, poses and goal outcomes. No ROS here; comres_ros feeds it.

The simulator's semantics are kept (COMRES-VLM docs/comres_spec.md):
  * one tick = one simulator step (dt seconds of real time);
  * every T_H ticks the robots' maps are merged; the merged map becomes every robot's map and the
    planner is asked (at most one question in flight);
  * between merges each robot plans on its own map: its live SLAM map on top of the last merged
    map (here: fuse(last merged, SLAM now), so what a robot saw since the merge is added);
  * a robot's target counts as reached once its frontier is gone from its own map (the
    "vanished = reached" rule, VANISH_RADIUS) or when it gets there; a target it cannot reach is
    given up and remembered for a while (the selector avoids it);
  * a robot without a target asks the COMRES selector (the strategy's target_select_fn) for one.
What differs is time: the planner's question runs in a background thread, the robots keep moving,
and the plan is delivered at the first tick after the answer arrives, so its latency is real
(the simulator charged it as ceil(seconds / dt) steps).
"""

import concurrent.futures
import math
import time
import traceback
from typing import Callable, Dict, List, Optional, Sequence, Tuple

import numpy as np

from . import ALGO_DIR  # noqa: F401  (importing the package puts the algorithm on sys.path)
from comres_strategy import UNREACHABLE_MEMORY, COMRESStrategy  # noqa: E402
from grid_map import frontier_mask, fuse  # noqa: E402
from params import Params  # noqa: E402

from .grid import WALL, free_cell_near

Cell = Tuple[int, int]
VANISH_RADIUS = 2.0          # COMRES-VLM robot.VANISH_RADIUS (cells)
ARRIVAL_RADIUS = 1.5         # cells: at the target


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

    @property
    def target(self) -> Optional[Cell]:
        return self._fallback_waypoint


class _Answer:
    """Stands in for the planner while a finished answer is handed to COMRESStrategy._query."""

    def __init__(self, result):
        self.result = result

    def plan(self, snap):
        return self.result


class OnlineStrategy(COMRESStrategy):
    """COMRESStrategy whose planner question runs in a background thread. `_sync` (unchanged)
    merges, decomposes and builds the snapshot; `_query` submits the question; `poll` hands the
    finished answer to the unchanged `_query`, which records and delivers it. While a question is
    in flight `_pending` is set, so `_sync` skips asking, as in the simulator."""

    def __init__(self, *args, **kwargs):
        kwargs["latency_enabled"] = False          # the latency is real: deliver when answered
        super().__init__(*args, **kwargs)
        self._pool = concurrent.futures.ThreadPoolExecutor(max_workers=1, thread_name_prefix="comres-planner")
        self._future = None
        self._inflight = None
        self.answer_seconds: List[float] = []

    def _query(self, snap, step: int, t_prep: float) -> None:
        if self._future is not None:               # at most one question in flight
            self.n_skipped_in_flight += 1
            return
        self._inflight = (snap, int(step), float(t_prep), time.monotonic())
        self._future = self._pool.submit(self.planner.plan, snap)
        self._pending = (None, None, None)          # in flight (read by _sync only)

    def poll(self, step: int) -> Optional[Dict]:
        """Deliver a finished answer; returns its record (or None)."""
        if self._future is None or not self._future.done():
            return None
        fut, (snap, qstep, t_prep, t0) = self._future, self._inflight
        self._future = self._inflight = None
        self._pending = None
        try:
            result = fut.result()
        except Exception:
            self.errors.append(traceback.format_exc())
            return None
        self.answer_seconds.append(time.monotonic() - t0)
        real, self.planner = self.planner, _Answer(result)
        try:
            COMRESStrategy._query(self, snap, qstep, t_prep)
        finally:
            self.planner = real
        rec = self.plan_records[-1]
        rec["released_step"] = int(step)
        rec["answer_seconds"] = round(self.answer_seconds[-1], 3)
        return rec

    def in_flight(self) -> bool:
        return self._future is not None

    def shutdown(self) -> None:
        self._pool.shutdown(wait=False, cancel_futures=True)


class Mission:
    """
    COMRES for n robots in real time. Per tick: update_robot() for every robot (its SLAM map on
    the planning grid and its cell), then tick(step) returns the robots whose target changed
    ({index: new target cell or None}); report(i, "reached" | "failed") feeds back a goal's
    outcome from the robot's navigation.
    """

    def __init__(self, names: Sequence[str], shape: Tuple[int, int], task: str = "search", hint: str = "",
                 params: Optional[Params] = None, planner=None, unreachable_memory: int = UNREACHABLE_MEMORY):
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
        self.unreachable_memory = int(unreachable_memory)

    @property
    def T_H(self) -> int:
        return self.strategy.T_H

    def start(self, start_cells: Sequence[Cell]) -> None:
        """Begin at the robots' start cells (the team start, as the hints mean it)."""
        self.robots = [RobotProxy(n, c, self.shape) for n, c in zip(self.names, start_cells)]
        self.slam = [None] * len(self.robots)
        self.strategy.attach(self.robots)
        self.started = True

    def update_robot(self, i: int, slam_grid: np.ndarray, cell: Cell) -> None:
        """Robot i's SLAM map on the planning grid and its current cell."""
        g = np.asarray(slam_grid, dtype=np.int8).copy()
        cell = free_cell_near(g, tuple(int(v) for v in cell))
        if g[cell] != WALL:
            g[cell] = 0                                # the robot stands on free floor
        self.slam[i] = g
        r = self.robots[i]
        r.position = np.array(cell, dtype=np.int64)
        r.local_map = g if self.merged is None else fuse([self.merged, g])

    def _event(self, kind: str, i: Optional[int] = None, **kw) -> None:
        self.events.append({"t": round(time.time(), 3), "step": self.step, "kind": kind,
                            **({"robot": self.names[i]} if i is not None else {}), **kw})

    def report(self, i: int, outcome: str) -> None:
        """A goal outcome from robot i's navigation: "reached" or "failed" (unreachable)."""
        r = self.robots[i]
        if r.target is None:
            return
        if outcome == "failed":
            r.unreachable.append((r.target, r.steps_taken))
        self._event(outcome, i, target=list(r.target))
        r._fallback_waypoint = None

    def tick(self, step: int) -> Dict[int, Optional[Cell]]:
        """One step: deliver a finished plan, merge and ask every T_H, check every robot's
        target, pick new ones. Returns {robot: new target or None} for the robots whose target
        changed."""
        if not self.started or any(s is None for s in self.slam):
            return {}
        self.step = int(step)
        st = self.strategy
        rec = st.poll(step)
        if rec is not None:
            self._event("plan", plan=rec.get("plan_lists"), hyps=rec.get("plan_hyps"),
                        answer_seconds=rec.get("answer_seconds"), valid=rec.get("valid"), error=rec.get("error"))
        if step % st.T_H == 0:
            for r, g in zip(self.robots, self.slam):
                r.local_map = g                        # the merge is of what the robots know now
            st._sync(self.robots, step)                # writes the merged map into every robot
            self.merged = st._merged
            self._event("merge", asked=st.in_flight())
        changed: Dict[int, Optional[Cell]] = {}
        for i, r in enumerate(self.robots):
            r.steps_taken = int(step)
            r.unreachable = [(p, t) for p, t in r.unreachable if step - t <= self.unreachable_memory]
            old = r.target
            if old is not None:
                at = math.dist(old, tuple(r.position)) <= ARRIVAL_RADIUS
                if at or not frontier_near(r.local_map, old):
                    self._event("arrived" if at else "seen", i, target=list(old))
                    r._fallback_waypoint = None
            if r.target is None:
                r._fallback_waypoint = r.target_select_fn(r)
                if r.target is not None:
                    self._event("target", i, target=list(r.target))
            if r.target != old:
                changed[i] = r.target
        return changed

    def stop(self) -> None:
        self.strategy.shutdown()
