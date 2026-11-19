"""
The map alignment (comres_core.map_align) on many random cases, offline: how often it fuses right,
whether it ever fuses wrong, how accurately, and after how many updates.

    python3 tools/map_align_eval.py [--trials 7] [--seed 0]

  overlap     a random floor (16 x 12 m): robot A's map its west part, robot B's its east part,
              sharing 2, 3 or 5 m (--trials of each); B's frame at a random pose; robot B walks
              from the shared part into its own
  no overlap  the two parts 2 m apart (2 * --trials): every "ready" is a false fusion
  real map    two parts of a Go2's RTAB-Map grid (test/data) sharing 2.5 to 3.5 m (--trials)

The error is where B's map lands against where it should: the RMS displacement of its obstacle
cells, and the yaw. A trial succeeds when the filter is ready at its end within 0.15 m and 2 deg; a
false fusion is any update where it is ready with more (or ready at all, without overlap).
"""

import argparse
import math
import os
import sys
import time

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path[:0] = [os.path.join(HERE, "..", "src", "comres_core"), os.path.join(HERE, "..", "src", "comres_core", "test")]

from align_worlds import floor_plan, in_frame, real_room, view  # noqa: E402
from comres_core.map_align import AlignmentFilter, ScoreField, candidates, map_points, transform, wrap  # noqa: E402

OK_XY, OK_YAW = 0.15, math.radians(2.0)


def error(T, truth, points):
    """(m, deg): the RMS displacement of B's points at T from where they are at truth, and the yaw."""
    d = transform(T, points) - transform(truth, points)
    return float(np.sqrt((d ** 2).sum(axis=1).mean())), math.degrees(abs(float(wrap(T[2] - truth[2]))))


MAX_CONFLICT = 0.05


def trial(a, b, truth, path, seed):
    """(ready at the end, error (m, deg) at the end, first update ready, false fusion, s per update)."""
    field = ScoreField(a)
    f = AlignmentFilter(seed=seed, max_conflict=MAX_CONFLICT)
    f.seed(candidates(a, b), a, path[0])
    whole = map_points(b, max_obstacles=1500, max_free=1500)
    first, false, t0 = None, False, time.perf_counter()
    for k, p in enumerate(path):
        e = f.update(field, map_points(b, center=p, rng=f.rng), whole)
        if e["ready"]:
            wrong = truth is None or (lambda er: er[0] > OK_XY or er[1] > math.degrees(OK_YAW))(error(e["T"], truth, whole[0]))
            false = false or wrong
            first = k if first is None else first
    dt = (time.perf_counter() - t0) / len(path)
    err = error(e["T"], truth, whole[0]) if truth else None
    return e["ready"], err, first, false, dt


def report(name, rows, with_truth=True):
    n = len(rows)
    ok = [r for r in rows if with_truth and r[0] and r[1][0] <= OK_XY and r[1][1] <= math.degrees(OK_YAW)]
    false = sum(r[3] for r in rows)
    line = f"{name:<22} {n:3d} trials  "
    if with_truth:
        errs = np.array([r[1] for r in ok]) if ok else np.zeros((0, 2))
        firsts = [r[2] for r in ok]
        line += (f"fused right {len(ok):3d} ({100 * len(ok) / n:3.0f}%)  false fusions {false}  "
                 + (f"error median {np.median(errs[:, 0]) * 100:.1f} cm {np.median(errs[:, 1]):.2f} deg, "
                    f"max {errs[:, 0].max() * 100:.1f} cm {errs[:, 1].max():.2f} deg  "
                    f"ready after {np.median(firsts) + 1:.0f} updates (median)" if ok else ""))
    else:
        line += f"false fusions {false}"
    print(line + f"  {1000 * np.mean([r[4] for r in rows]):.0f} ms/update", flush=True)


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--trials", type=int, default=7)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--max-conflict", type=float, default=0.05, help="the filter's readiness conflict limit")
    args = ap.parse_args()
    global MAX_CONFLICT
    MAX_CONFLICT = args.max_conflict
    rng = np.random.default_rng(args.seed)
    rand_T = lambda: (float(rng.uniform(-5, 5)), float(rng.uniform(-5, 5)), float(rng.uniform(-math.pi, math.pi)))  # noqa: E731

    for shared in (2.0, 3.0, 5.0):
        rows = []
        for t in range(args.trials):
            w = floor_plan(seed=int(rng.integers(1 << 30)))
            truth = rand_T()
            a = view(w, (0, 0, 8 + shared / 2, 12), seed=t)
            b = view(w, (8 - shared / 2, 0, 16, 12), T=truth, seed=t + 100)
            y = float(rng.uniform(3, 9))
            path = in_frame(truth, [(x, y) for x in np.linspace(8, 15, 12)])
            rows.append(trial(a, b, truth, path, seed=t))
        report(f"overlap {shared:.0f} m", rows)

    rows = []
    for t in range(2 * args.trials):
        w = floor_plan(seed=int(rng.integers(1 << 30)))
        T = rand_T()
        a, b = view(w, (0, 0, 7, 12), seed=t), view(w, (9, 0, 16, 12), T=T, seed=t + 100)
        rows.append(trial(a, b, None, in_frame(T, [(x, 6.0) for x in np.linspace(9.5, 15.5, 12)]), seed=t))
    report("no overlap", rows, with_truth=False)

    w = real_room()
    height, width = w.shape[0] * 0.05, w.shape[1] * 0.05
    free = np.argwhere(w == 0)[:, ::-1] * 0.05 + 0.025
    rows = []
    for t in range(args.trials):
        truth = rand_T()
        split = float(rng.uniform(4.0, 5.0))
        a, b = view(w, (0, 0, width, 7.5), seed=t), view(w, (0, split, width, height), T=truth, seed=t + 100)
        pts = free[(free[:, 1] > split + 0.2) & (free[:, 1] < height - 0.5)]
        path = sorted(map(tuple, pts[rng.choice(len(pts), 10, replace=False)]), key=lambda p: p[1])
        rows.append(trial(a, b, truth, in_frame(truth, path), seed=t))
    report("real Go2 map", rows)


if __name__ == "__main__":
    main()
