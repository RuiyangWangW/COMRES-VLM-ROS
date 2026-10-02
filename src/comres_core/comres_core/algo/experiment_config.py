"""
Single source of truth for the conditions every arm runs under.

Robot counts, horizons, the sensor, dt, the coverage target and the per-run seeds are
defined here and nowhere else, so every arm faces the same runs. protocol builds the
units from them and hashes what a run faced into its `fingerprint` (map bytes, spawns,
target, robot count, horizon, T_H, dt, seed), so rows of different arms pair only when
their conditions were provably identical.
"""

import hashlib
import json
import platform
import subprocess
from pathlib import Path
from typing import Dict, List

import numpy as np

# The benchmark (benchmark/README.md): its manifest and maps live next to this file.
BENCHMARK = Path(__file__).resolve().parent / "benchmark"
MANIFEST = str(BENCHMARK / "benchmark_350.json")
MAPS_DIR = str(BENCHMARK / "maps")

# --- conditions ------------------------------------------------------------
ROBOTS = {"small": 2, "medium": 4, "large": 6}

# The robots' sensor, the same for every robot of every method (simulator.spawn_robots
# builds every robot with it; robot.Robot defaults to it): a 100 degree field of view and a
# range of 20 cells (2 m on the HM3D scans). The disc of radius SENSOR_ALWAYS_OBSERVED_RADIUS
# around the robot is observed regardless of heading. Params.d_det (the radius of B(f)) must
# equal SENSOR_RANGE.
SENSOR_FOV_DEG = 100
SENSOR_RANGE = 20
SENSOR_ALWAYS_OBSERVED_RADIUS = 2


def sensor_record() -> Dict:
    """The sensor, as recorded in every result row."""
    return {"fov_deg": SENSOR_FOV_DEG, "range": SENSOR_RANGE,
            "always_observed_radius": SENSOR_ALWAYS_OBSERVED_RADIUS}

# Search ends when any robot observes the target; explore ends at COVERAGE_TARGET of the
# free cells. The step caps (one per task, every size) are far above the longest runs
# measured when they were set (search about 1200 steps, explore 1395 on a large HM3D scan),
# so no run that makes progress is censored. A cap costs nothing on runs that succeed (they
# stop early); a run at the cap is a failure, so read mean steps together with `success`.
HORIZON = {
    "search":  {"small": 2000, "medium": 2000, "large": 2000},
    "explore": {"small": 5000, "medium": 5000, "large": 5000},
}

COVERAGE_TARGET = 0.95          # explore ends when the team has seen this share of the free cells

# Seconds of robot execution per simulation step. Mission time is steps * DT.
# Planning runs concurrently with execution (see latency_strategy), so a slow
# coordinator costs steps rather than adding a separate term.
#
# One step = one cell per second, a realistic robot speed for ~1 m cells, so a plan is
# released ceil(latency / DT) steps after its query -- a 50 ms Hungarian solve waits 1
# step, a 5 s VLM call 5 steps. DT is part of the fingerprint.
DT = 1.0


def seed_for(scenario_id: str, repeat: int) -> int:
    """
    Deterministic per-(unit, repeat) seed, identical across strategies (protocol.seed
    passes the unit id: a search unit's scenario id, or "<map_id>#s<k>" for explore).

    Holding the seed fixed across strategies makes the comparison PAIRED: every strategy
    faces the same initial robot headings and the same robot random streams (the MPPI
    controller's, seeded from it by GridMapSimulator.spawn_robots) on the same map, so a
    difference between them is the strategy, not the draw. Varying it across repeats is what
    makes repeats informative.
    """
    if isinstance(scenario_id, str) and scenario_id.isdigit():
        base = int(scenario_id)
    else:                                   # explore unit ids ("<map_id>#s<k>") are hashed
        base = int(hashlib.md5(str(scenario_id).encode()).hexdigest()[:6], 16)
    # numpy.random.seed rejects anything >= 2**32, which the hashed map ids
    # exceed once scaled. Numeric scenario ids are far below the wrap point, so
    # their seeds are unaffected.
    return (base * 1_000 + repeat) % (2 ** 32 - 1)


def seed_rngs(seed: int) -> None:
    """Seed every RNG the simulation touches. AEP uses stdlib random."""
    import random
    np.random.seed(seed)
    random.seed(seed)


def load_scenarios(manifest: str = MANIFEST) -> List[Dict]:
    return json.load(open(manifest))["scenarios"]


def map_digest(scenario: Dict, maps_dir: str = MAPS_DIR) -> str:
    p = Path(maps_dir) / scenario["scene_id"] / f"{scenario['map_variant']}.npy"
    return hashlib.md5(p.read_bytes()).hexdigest()[:16]


def provenance() -> Dict:
    """Recorded once per results file, so a mismatch can be diagnosed later."""
    try:
        commit = subprocess.check_output(
            ["git", "rev-parse", "--short", "HEAD"], text=True).strip()
        dirty = bool(subprocess.check_output(
            ["git", "status", "--porcelain"], text=True).strip())
    except Exception:
        commit, dirty = "unknown", None
    return {
        "git_commit": commit,
        "git_dirty": dirty,
        "platform": platform.platform(),
        "python": platform.python_version(),
        "numpy": np.__version__,
        "robots": ROBOTS,
        "horizon": HORIZON,
    }
