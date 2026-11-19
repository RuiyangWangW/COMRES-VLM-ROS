"""The transform between two robots' maps (comres_core.map_align), on synthetic floors and a real
Go2 map: it converges where the maps overlap, waits where another transform fits as well, and is
never ready where they do not overlap."""
import math

import numpy as np

from align_worlds import floor_plan, in_frame, real_room, twin_rooms, view
from comres_core.map_align import (AlignmentFilter, GridMap, ScoreField, candidates, compose, map_points, transform,
                                   wrap)


def error(T, truth):
    return math.hypot(T[0] - truth[0], T[1] - truth[1]), abs(math.degrees(float(wrap(T[2] - truth[2]))))


def run(a, stages, seed=0):
    """stages: [(b, path in B's frame)]; the maps grow from one stage to the next. Returns the
    filter, the last estimate and, per stage, whether it was ready after each update."""
    field = ScoreField(a)
    f = AlignmentFilter(seed=seed)
    ready = []
    est = None
    for k, (b, path) in enumerate(stages):
        cands = candidates(a, b)
        f.seed(cands, a, path[0]) if k == 0 else f.inject(cands)
        whole = map_points(b, max_obstacles=1500, max_free=1500)
        flags = []
        for p in path:
            est = f.update(field, map_points(b, center=p, rng=f.rng), whole)
            flags.append(est["ready"])
        ready.append(flags)
    return f, est, ready


def test_grid_geometry():
    m = GridMap(np.zeros((40, 60), np.int8), 0.05, (1.0, -2.0, 0.7))
    m.data[10, 20] = 100
    p = m.points(m.occupied)
    i, j = m.cell_of(p[:, 0], p[:, 1])
    assert (i[0], j[0]) == (20, 10)
    T, U = (1.0, 2.0, 0.5), (-0.3, 0.4, -1.2)
    q = np.array([[0.7, -0.2]])
    assert np.allclose(transform(compose(T, U), q), transform(T, transform(U, q)))


def test_search_has_the_truth_among_its_candidates():
    w = floor_plan(seed=1)
    truth = (3.0, -2.0, math.radians(70))
    a, b = view(w, (0, 0, 10, 12), seed=1), view(w, (6, 0, 16, 12), T=truth, seed=2)
    cands = candidates(a, b)
    assert any(error(T, truth)[0] < 0.3 and error(T, truth)[1] < 3.0 for _, T in cands[:5])


def test_converges_where_the_maps_overlap():
    w = floor_plan(seed=1)
    truth = (3.0, -2.0, math.radians(70))
    a, b = view(w, (0, 0, 10, 12), seed=1), view(w, (6, 0, 16, 12), T=truth, seed=2)
    path = in_frame(truth, [(x, 6.0) for x in np.linspace(7, 15, 12)])  # from the shared 4 m eastwards
    _, est, ready = run(a, [(b, path)])
    assert ready[0][-1] and sum(ready[0]) >= len(path) - 4             # ready early, and it stays
    dx, dyaw = error(est["T"], truth)
    assert dx < 0.1 and dyaw < 1.0


def test_waits_in_a_symmetric_place_then_converges():
    """Robot B has mapped only the west room, which fits both identical rooms of A: no fusion.
    Once its map holds the corridor's pillar, there is one answer."""
    w = twin_rooms()
    truth = (1.5, -0.5, math.radians(-40))
    a = view(w, (0, 0, 12, 9), seed=1)
    room = view(w, (0.45, 2.95, 5.6, 8.1), T=truth, seed=2)
    room_and_corridor = view(w, (0.0, 0.3, 5.6, 8.1), T=truth, seed=2)
    in_room = in_frame(truth, [(2, 4), (3, 5), (4, 6), (3, 7), (2, 6), (1.5, 5)] * 2)
    in_corridor = in_frame(truth, [(2.5, 3.5), (2.5, 2.5), (2.5, 2.0), (3.0, 1.5), (3.5, 1.5), (4.0, 1.5)])
    _, est, ready = run(a, [(room, in_room), (room_and_corridor, in_corridor)])
    assert not any(ready[0])
    assert ready[1][-1]
    dx, dyaw = error(est["T"], truth)
    assert dx < 0.1 and dyaw < 1.0


def test_never_ready_where_the_maps_do_not_overlap():
    for seed in (1, 2, 3):
        w = floor_plan(seed=seed)
        T = (2.0, 1.0, 1.0)
        a, b = view(w, (0, 0, 7, 12), seed=1), view(w, (9, 0, 16, 12), T=T, seed=2)
        _, _, ready = run(a, [(b, in_frame(T, [(x, 6) for x in np.linspace(9.5, 15.5, 12)]))])
        assert not any(ready[0]), f"floor {seed}"


def test_real_go2_map():
    """Two overlapping parts (3.5 m) of a Go2's RTAB-Map grid, the second turned 135 deg."""
    w = real_room()
    height, width = w.shape[0] * 0.05, w.shape[1] * 0.05
    truth = (-1.0, 2.0, math.radians(135))
    a, b = view(w, (0, 0, width, 7.5), seed=1), view(w, (0, 4.0, width, height), T=truth, seed=2)
    free = np.argwhere(w == 0)[:, ::-1] * 0.05 + 0.025
    free = free[(free[:, 1] > 4.2) & (free[:, 1] < 9.0)]
    path = sorted(map(tuple, free[np.random.default_rng(0).choice(len(free), 10, replace=False)]), key=lambda p: p[1])
    _, est, ready = run(a, [(b, in_frame(truth, path))])
    assert ready[0][-1]
    dx, dyaw = error(est["T"], truth)
    assert dx < 0.1 and dyaw < 1.0
