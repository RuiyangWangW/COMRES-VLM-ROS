"""
The central computer <-> VLM server link, end to end, without ROS or robots:

    python3 tools/vlm_check.py [--url http://127.0.0.1:18765/v1] [--model qwen3.8-27b] [--plans 3]

It asks the server for its models (what ~/start checks), then runs COMRES on the tests' small
world (two robots standing at its west end) until --plans plans have come back. Each is the
question the node asks, sent by the node's code path (Mission -> planner thread -> vlm_client):
the picture, the text and the reply schema. Per plan it prints the request size, the time from
question to answer, the server's call time, the tokens and whether the plan was usable.
Exit 0 when every plan was usable.
"""

import argparse
import json
import os
import sys
import tempfile
import time
import urllib.request

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.join(HERE, "..", "src", "comres_core"))
sys.path.insert(0, os.path.join(HERE, "..", "src", "comres_core", "test"))


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--url", default="http://127.0.0.1:18765/v1")
    ap.add_argument("--model", default="qwen3.8-27b")
    ap.add_argument("--plans", type=int, default=3)
    ap.add_argument("--timeout", type=float, default=90.0, help="seconds per VLM call (the node's vlm_timeout)")
    a = ap.parse_args()

    os.environ["COMRES_VLM_BASE_URL"] = a.url                # as comres_node sets them
    os.environ["COMRES_VLM_MODEL"] = a.model
    os.environ["COMRES_VLM_TIMEOUT"] = str(a.timeout)
    os.environ.pop("COMRES_VLM_PROVIDER", None)
    os.environ.pop("COMRES_FAKE_VLM", None)
    os.environ["COMRES_VLM_LOCK"] = os.path.join(tempfile.mkdtemp(prefix="vlm_check_"), "vlm.lock")

    t = time.perf_counter()
    try:
        with urllib.request.urlopen(a.url.rstrip("/") + "/models", timeout=5) as resp:
            ids = [m.get("id") for m in json.loads(resp.read()).get("data", [])]
    except Exception as e:                                     # noqa: BLE001
        print(f"FAIL: {a.url} not reachable: {e}", flush=True)
        return 2
    print(f"models: {ids} ({1000 * (time.perf_counter() - t):.0f} ms)", flush=True)
    if a.model not in ids:
        print(f"FAIL: the server does not serve {a.model}", flush=True)
        return 2

    import numpy as np
    from comres_core.online import Mission
    from params import Params
    from test_online import UNK, sense, world

    truth = world()
    cells = [(19, 2), (20, 2)]
    known = [np.full(truth.shape, UNK, np.int8) for _ in cells]
    for k, c in zip(known, cells):
        sense(truth, k, c)
    m = Mission(["go2_1", "go2_2"], truth.shape, task="search", hint="The target is in the far north-east room.",
                params=Params(T_H=1))                          # ask again as soon as an answer is in
    m.start(cells)
    n_seen, results, step = 0, [], 0
    deadline = time.monotonic() + a.plans * (a.timeout + 10)
    try:
        while len(results) < a.plans and time.monotonic() < deadline:
            for i, c in enumerate(cells):
                m.update_robot(i, known[i], c)
            m.tick(step)
            for e in m.events[n_seen:]:
                if e["kind"] == "plan_error":
                    results.append(False)
                    print(f"plan {len(results)}: FAIL (planner raised): {e['error'][-300:]}", flush=True)
                elif e["kind"] == "plan":
                    rec = next(r for r in reversed(m.strategy.plan_records) if r.get("query_step") == e["query_step"])
                    msgs = rec.get("_messages") or []
                    body = len(json.dumps(msgs))
                    png = sum(len(p["image_url"]["url"]) for mm in msgs if isinstance(mm["content"], list)
                              for p in mm["content"] if p["type"] == "image_url")
                    results.append(bool(rec.get("valid")))
                    print(f"plan {len(results)}: {'ok' if rec.get('valid') else 'UNUSABLE'}  "
                          f"request {body / 1024:.0f} KiB (picture {png / 1024:.0f} KiB base64)  "
                          f"answer {rec.get('answer_seconds')} s, call {rec.get('call_seconds')} s  "
                          f"tokens in {rec.get('prompt_tokens')} out {rec.get('output_tokens')} "
                          f"({rec.get('finish_reason')})", flush=True)
                    print(f"        pieces {rec.get('lists')}  guesses {rec.get('hyps')}"
                          + (f"  error {rec.get('error')}" if rec.get("error") else ""), flush=True)
            n_seen = len(m.events)
            step += 1
            time.sleep(0.1)
    finally:
        m.stop()
    if len(results) < a.plans:
        print(f"FAIL: {len(results)} of {a.plans} plans came back in time", flush=True)
        return 1
    print("PASS" if all(results) else "FAIL: some plans were unusable", flush=True)
    return 0 if all(results) else 1


if __name__ == "__main__":
    sys.exit(main())
