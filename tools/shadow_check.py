"""
Shadow check: does comres_core's Mission (COMRES as the ROS node runs it) decide exactly as the
simulator's COMRES?

For each recorded COMRES run of COMRES-VLM (a results/.../comres.jsonl row), the simulator runs it
again as tools/replay_check.py does (every VLM call answered with the reply recorded at that plan,
every plan released after its recorded delay), and a Mission runs alongside in shadow: at every
step it gets exactly what the ROS node would get from a robot -- the map each simulated robot
holds right after sensing and the cell it sensed from -- plus the robots' given-up targets, and
its plans are answered with the same recorded replies and delivered at the same steps. At every
step the Mission's target for each robot must equal the simulated robot's target, its merged map
the simulator's, what it shows the VLM the simulator's (robot cells, committed targets, pieces,
the belief, the map), and its plans the recorded ones. Motion stays the simulator's (the ROS node
leaves it to Nav2), so this checks COMRES's decisions, not driving.

    python tools/shadow_check.py --sim /path/to/COMRES-VLM [--limit N] results/benchmark/search_correct/comres.jsonl ...

Exit 0 when every checked run matches at every step.
"""

import argparse
import concurrent.futures
import filecmp
import hashlib
import importlib.util
import json
import os
import sys
import time
from pathlib import Path

ALGO_FILES = ("comres_strategy.py", "comres_planner.py", "comres_points.py", "units.py", "belief.py",
              "selector.py", "params.py", "grid_map.py", "vlm_client.py", "waypoint_strategy.py",
              "experiment_config.py")


def fingerprint(snap):
    """What the VLM is shown, in comparable form."""
    import numpy as np

    def h(a):
        return None if a is None else hashlib.md5(np.ascontiguousarray(a).tobytes()).hexdigest()
    return {"robot_pos": repr(snap.robot_pos), "waypoints": repr(snap.waypoints), "lists": repr(snap.lists),
            "team_start": repr(snap.team_start), "occ": h(snap.occ), "belief": h(snap.belief_map),
            "pieces": repr(getattr(snap, "piece_desc", None)), "units": repr(getattr(snap, "unit_desc", None))}


class SyncPool:
    """Runs the Mission's planner question at once, in this thread (the check is step-locked)."""

    def submit(self, fn, *args):
        f = concurrent.futures.Future()
        try:
            f.set_result(fn(*args))
        except Exception as e:                           # noqa: BLE001
            f.set_exception(e)
        return f

    def shutdown(self, **kw):
        pass


def shadow_one(row, run, vlm_client, replay, Mission):
    recs = row.get("plan_records") or []
    st = {"k": 0}

    def fake_call(messages, schema, model=None, base_url=None):
        k = st["k"]
        if k >= len(recs):
            return "", dict(call_seconds=0.5, lock_wait_seconds=0.0, finish_reason="replay_missing")
        r = recs[k]
        return r["raw"], dict(call_seconds=float(r.get("call_seconds") or 0.0), lock_wait_seconds=0.0,
                              prompt_tokens=r.get("prompt_tokens"), output_tokens=r.get("output_tokens"),
                              reasoning_tokens=r.get("reasoning_tokens"), finish_reason=r.get("finish_reason"),
                              model=r.get("model"), fake=False)

    vlm_client.call = fake_call
    sense, sim_target, giveup = {}, {}, {}
    sim_snaps, m_snaps = {}, {}
    box = {}
    problems = []

    def configure(strategy, unit):
        # The simulator: recorded replies (index set before each question) and recorded delays.
        orig_query = strategy._query

        def q(snap, step, t_prep):
            sim_snaps[int(step)] = fingerprint(snap)
            k = len(strategy.plan_records)
            st["k"] = k
            orig_query(snap, step, t_prep)
            rec = strategy.plan_records[-1]
            d = int(recs[k]["delay_steps"]) if k < len(recs) else 0
            if strategy._pending is not None and strategy._pending[2] is rec:
                strategy._pending = (step + d, strategy._pending[1], rec)
        strategy._query = q

        # The simulated robots: what each sensed, from where; its target after the vanish check
        # (and selection); the targets it gave up.
        for i, r in enumerate(strategy.robots):
            def upd(gt, r=r, i=i, orig=r.update_local_map):
                orig(gt)
                sense[(r.steps_taken - 1, i)] = (r.local_map.copy(), tuple(int(v) for v in r.position))
            r.update_local_map = upd

            def cur(r=r, i=i, orig=r._current_target):
                w = orig()
                sim_target[(r.steps_taken - 1, i)] = None if w is None else tuple(int(v) for v in w)
                return w
            r._current_target = cur

            def failed(wp, r=r, i=i, orig=r._route_failed):
                n = len(r.unreachable)
                out = orig(wp)
                if len(r.unreachable) > n:
                    giveup[(r.steps_taken - 1, i)] = r.unreachable[-1][0]
                return out
            r._route_failed = failed

        # The shadow Mission: the same task, hint, parameters and team start.
        from comres_planner import VLMPlanner
        names = [f"r{i}" for i in range(len(strategy.robots))]
        shape = strategy.robots[0].local_map.shape

        class ShadowPlanner:
            def __init__(self):
                self.inner = VLMPlanner(strategy.task, strategy.hint or "", strategy.n_robots, strategy.params,
                                        variant=strategy.variant)

            def plan(self, snap):
                st["k"] = len(box["m"].strategy.plan_records)
                return self.inner.plan(snap)

        m = Mission(names, shape, task=strategy.task, hint=strategy.hint or "", params=strategy.params,
                    planner=ShadowPlanner())
        m.strategy._pool = SyncPool()
        orig_poll = m.strategy.poll

        def poll(step):                                   # deliver at the recorded release step
            inf = m.strategy._inflight
            if inf is None:
                return None
            k = len(m.strategy.plan_records)
            d = int(recs[k]["delay_steps"]) if k < len(recs) else 0
            return orig_poll(step) if step >= inf[1] + d else None
        m.strategy.poll = poll
        m_query = m.strategy._query

        def mq(snap, step, t_prep):
            m_snaps[int(step)] = fingerprint(snap)
            return m_query(snap, step, t_prep)
        m.strategy._query = mq
        m.start([tuple(int(v) for v in p) for p in strategy.team_start])
        box["m"] = m

    compared = {"steps": 0, "robot_steps": 0, "questions": 0}

    sim_merged = {}

    def hook(sim, planner):
        m = box["m"]
        s = sim.step_count - 1
        n = len(sim.robots)
        if any((s, i) not in sense for i in range(n)):
            return
        for i in range(n):                                # given up during the previous step
            if (s - 1, i) in giveup:
                m.report(i, "failed")
        for i in range(n):                                # what each robot sensed at step s, and where
            g, cell = sense[(s, i)]
            m.update_robot(i, g, cell)
        m.tick(s)                                         # ends step s-1, begins step s
        compared["steps"] += 1
        for i in range(n):
            compared["robot_steps"] += 1
            want, got = sim_target.get((s, i)), m.robots[i].target
            if want != got and len(problems) < 20:
                problems.append(f"step {s} robot {i}: simulator target {want}, Mission {got}")
        if (s - 1) in sim_merged:                         # the Mission merged step s-1 in this tick
            if m.merged is None or (m.merged != sim_merged[s - 1]).any():
                problems.append(f"step {s - 1}: merged maps differ")
        if s % planner.T_H == 0 and not sim_done(sim) and planner._merged is not None:
            sim_merged[s] = planner._merged.copy()

    def sim_done(sim):
        return box.get("done", False)

    unit = replay._unit_for(row)
    job = {"unit": unit, "arm": row["arm"], "repeat": int(row.get("repeat", 0)), "cwd": os.getcwd(),
           "tag": row.get("tag"), "phase": row["phase"], "latency_enabled": bool(row.get("latency_enabled", True)),
           "quiet": True, "vanish_reached": bool(row.get("vanish_reached", True)), "dt": float(row["dt"]),
           "env": replay._env_for(row)}
    # The simulator stops consulting its strategy in the step the target is found / coverage is
    # reached; mark that before the hook runs.
    orig_step = None
    out = None
    try:
        import simulator
        orig_step = simulator.GridMapSimulator.step

        def step(self, stop_condition=None):
            orig_step(self, stop_condition)
            box["done"] = bool(stop_condition is not None and stop_condition())
        simulator.GridMapSimulator.step = step
        out = run.run_unit(job, step_hook=hook, configure=configure)
    finally:
        if orig_step is not None:
            simulator.GridMapSimulator.step = orig_step
    m = box.get("m")
    if out is None or out.get("error"):
        problems.append(f"simulation error: {None if out is None else out.get('error')}")
    elif out["steps"] != row["steps"]:
        problems.append(f"the simulator itself did not replay: steps {out['steps']} != {row['steps']}")
    if m is not None:
        mp = [(r.get("query_step"), r.get("released_step"), json.dumps(r.get("plan_lists"), sort_keys=True))
              for r in m.strategy.plan_records]
        rp = [(r.get("query_step"), r.get("released_step"), json.dumps(r.get("plan_lists"), sort_keys=True))
              for r in recs]
        if mp != rp[:len(mp)] or len(mp) < len([r for r in recs if r.get("released_step") is not None]):
            problems.append(f"plans differ: Mission {mp[:4]} recorded {rp[:4]}")
        for step in sorted(set(sim_snaps) | set(m_snaps)):
            a, b = sim_snaps.get(step), m_snaps.get(step)
            if a != b:
                what = "asked only by one" if a is None or b is None else [k for k in a if a[k] != b[k]]
                problems.append(f"step {step}: what the VLM is shown differs: {what}")
                break
        compared["questions"] = len(sim_snaps)
        if m.strategy.errors:
            problems.append(f"Mission errors: {m.strategy.errors[0][-400:]}")
        m.stop()
    return {"unit": row["unit_id"], "condition": row["condition"], "size": row["size"], "family": row["family"],
            "repeat": row.get("repeat"), "steps": row["steps"], "compared_steps": compared["steps"],
            "robot_steps": compared["robot_steps"], "plans": len(recs), "questions": compared["questions"],
            "ok": not problems, "problems": problems}


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("files", nargs="+")
    ap.add_argument("--sim", required=True, help="the COMRES-VLM checkout")
    ap.add_argument("--limit", type=int, default=None, help="first N rows per file")
    ap.add_argument("--units", nargs="*", default=None, help="only these unit ids")
    a = ap.parse_args(argv)
    sim_root = Path(a.sim).resolve()
    ws = Path(__file__).resolve().parent.parent
    algo = ws / "src" / "comres_core" / "comres_core" / "algo"
    differ = [f for f in ALGO_FILES if not filecmp.cmp(sim_root / f, algo / f, shallow=False)]
    if differ:
        print(f"the algorithm copy differs from {sim_root}: {differ} (run tools/sync_core.sh)")
        return 2
    os.chdir(sim_root)
    sys.path.insert(0, str(sim_root))
    # The simulator's modules first (comres_core then finds them in sys.modules: same files).
    import run  # noqa: F401
    import experiment_config, comres_strategy, comres_planner, units, selector, belief, grid_map, params  # noqa: F401,E401
    import vlm_client
    spec = importlib.util.spec_from_file_location("replay_check", sim_root / "tools" / "replay_check.py")
    replay = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(replay)
    sys.path.append(str(ws / "src" / "comres_core"))
    from comres_core.online import Mission
    for k in list(os.environ):
        if k.startswith("COMRES_PARAM_") or k == "COMRES_FAKE_VLM":
            del os.environ[k]
    rows = []
    for f in a.files:
        n = 0
        for line in open(f):
            if not line.strip():
                continue
            r = json.loads(line)
            if a.units and str(r["unit_id"]) not in a.units:
                continue
            rows.append(r)
            n += 1
            if a.limit and n >= a.limit:
                break
    t0 = time.time()
    n_ok = 0
    for r in rows:
        res = shadow_one(r, run, vlm_client, replay, Mission)
        n_ok += res["ok"]
        print(f"{res['condition']:15s} {res['size']:6s} {res['family']:16s} unit {res['unit']:>4} r{res['repeat']} "
              f"{res['steps']:5d} steps, {res['plans']} plans, {res['questions']} questions, "
              f"{res['robot_steps']:6d} robot-steps compared: "
              f"{'SAME' if res['ok'] else 'DIFFERENT'}", flush=True)
        for p in res["problems"][:6]:
            print(f"     {p}")
    print(f"{n_ok}/{len(rows)} runs decided exactly the same at every step ({time.time() - t0:.0f} s)")
    return 0 if n_ok == len(rows) else 1


if __name__ == "__main__":
    sys.exit(main())
