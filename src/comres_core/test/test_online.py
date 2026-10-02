"""Real-time COMRES on a small simulated world: perfect sensing, robots that walk one cell per
tick toward their targets, the planner in fake mode (random valid answers) or a slow stand-in."""
import math
import time
from collections import deque

import numpy as np
import pytest

from comres_core.online import Mission, frontier_near
from params import Params

UNK, FREE, WALL = -1, 0, 1


def world():
    """A corridor with four rooms; the robots start at its west end."""
    m = np.ones((40, 60), np.int8)
    m[18:22, 1:59] = FREE                                 # corridor (east-west)
    for c0 in (8, 22, 36, 48):
        m[4:16, c0:c0 + 8] = FREE                         # rooms south... (rows 4-15)
        m[16:18, c0 + 3:c0 + 5] = FREE                    # doors
        m[24:36, c0:c0 + 8] = FREE                        # ...and north (rows 24-35)
        m[22:24, c0 + 3:c0 + 5] = FREE
    return m


def sense(truth, known, cell, rng=8):
    """Reveal the cells within `rng` that a straight line from `cell` reaches without a wall."""
    from skimage.draw import line
    r0, c0 = cell
    H, W = truth.shape
    for dr in range(-rng, rng + 1):
        for dc in range(-rng, rng + 1):
            if dr * dr + dc * dc > rng * rng:
                continue
            rr, cc = line(r0, c0, r0 + dr, c0 + dc)
            for r, c in zip(rr, cc):
                if not (0 <= r < H and 0 <= c < W):
                    break
                known[r, c] = truth[r, c]
                if truth[r, c] == WALL:
                    break


def step_toward(truth_known, cell, goal):
    """One cell along a shortest path on the robot's known free cells (stay if none)."""
    H, W = truth_known.shape
    prev = {cell: None}
    q = deque([cell])
    while q:
        u = q.popleft()
        if u == goal:
            break
        for d in ((1, 0), (-1, 0), (0, 1), (0, -1)):
            v = (u[0] + d[0], u[1] + d[1])
            if 0 <= v[0] < H and 0 <= v[1] < W and v not in prev and truth_known[v] == FREE:
                prev[v] = u
                q.append(v)
    if goal not in prev:
        return cell, False
    v = goal
    while prev[v] is not None and prev[v] != cell:
        v = prev[v]
    return v, True


def run(mission, truth, starts, target, ticks=400, slow=None):
    known = [np.full(truth.shape, UNK, np.int8) for _ in starts]
    cells = list(starts)
    for i, c in enumerate(cells):
        sense(truth, known[i], c)
    mission.start(cells)
    goals = [None] * len(cells)
    for step in range(ticks):
        for i in range(len(cells)):
            if mission.merged is not None:
                known[i] = np.maximum(known[i], mission.robots[i].local_map)
            mission.update_robot(i, known[i], cells[i])
        for i, g in mission.tick(step).items():
            goals[i] = g
        for i in range(len(cells)):
            if goals[i] is None:
                continue
            nxt, ok = step_toward(known[i], cells[i], goals[i])
            if not ok:
                mission.report(i, "failed")
                goals[i] = None
                continue
            cells[i] = nxt
            if nxt == goals[i]:
                mission.report(i, "reached")
                goals[i] = None
            sense(truth, known[i], cells[i])
            if any(k[target] != UNK for k in known):
                return step
        if slow is not None:
            time.sleep(slow)
    return None


@pytest.fixture
def fake_vlm(monkeypatch, tmp_path):
    monkeypatch.setenv("COMRES_FAKE_VLM", "1")
    monkeypatch.setenv("COMRES_VLM_LOCK", str(tmp_path / "lock"))


def test_two_robots_find_the_target(fake_vlm):
    truth = world()
    m = Mission(["go2_1", "go2_2"], truth.shape, task="search", hint="The target is in the far north-east room.",
                params=Params(T_H=20))
    found = run(m, truth, [(19, 2), (20, 2)], target=(30, 52))
    try:
        assert found is not None and found < 400
        kinds = [e["kind"] for e in m.events]
        assert kinds.count("merge") >= found // 20 and "plan" in kinds and "target" in kinds
        assert any(e["kind"] == "seen" for e in m.events)                 # targets end when seen
        assert not m.strategy.errors
        plans = [e for e in m.events if e["kind"] == "plan"]
        assert all(p["valid"] for p in plans)
    finally:
        m.stop()


class SlowPlanner:
    """Answers like the fake VLM, but only after `delay` seconds (a slow model)."""

    def __init__(self, delay, task, hint, n):
        from comres_planner import VLMPlanner
        self.inner = VLMPlanner(task, hint, n, Params(T_H=5))
        self.delay = delay

    def plan(self, snap):
        time.sleep(self.delay)
        return self.inner.plan(snap)


def test_a_slow_answer_does_not_stop_the_robots(fake_vlm):
    truth = world()
    hint = "North-east room."
    m = Mission(["a", "b"], truth.shape, task="search", hint=hint, params=Params(T_H=5),
                planner=SlowPlanner(0.3, "search", hint, 2))
    try:
        run(m, truth, [(19, 2), (20, 2)], target=(30, 52), ticks=40, slow=0.02)
        st = m.strategy
        assert st.n_skipped_in_flight >= 1                                  # merges while asking skip asking
        recs = [r for r in st.plan_records if r.get("released_step") is not None]
        assert recs and all(r["released_step"] > r["query_step"] for r in recs)   # delivered later
        targets = [e for e in m.events if e["kind"] == "target"]
        assert targets and targets[0]["step"] < recs[0]["released_step"]          # robots moved meanwhile
    finally:
        m.stop()


def test_a_failed_goal_is_avoided(fake_vlm):
    truth = world()
    m = Mission(["a"], truth.shape, task="explore", params=Params(T_H=50))
    known = np.full(truth.shape, UNK, np.int8)
    start = (19, 30)                                   # mid-corridor: several frontiers in view
    sense(truth, known, start)
    m.start([start])
    m.update_robot(0, known, start)
    first = m.tick(0)[0]
    m.report(0, "failed")
    assert m.robots[0].unreachable and m.robots[0].unreachable[0][0] == first
    second = m.tick(1)[0]
    assert second is not None and math.dist(second, first) > 2           # not the same place again
    # (with one frontier only, the selector ignores the exclusion rather than idle, as in COMRES-VLM)
    m.stop()


def test_frontier_near():
    occ = np.full((10, 10), UNK, np.int8)
    occ[:, :5] = FREE
    assert frontier_near(occ, (5, 4)) and frontier_near(occ, (5, 3)) and not frontier_near(occ, (5, 0))
