"""
Every tunable parameter of COMRES-VLM in one frozen dataclass.

The modules (units, belief, planner, selector, strategy) are written and tuned
separately, so their knobs live here and nowhere else: a run logs its Params (`params` of a
run row, hashed into its resume key), and a tuning sweep changes a value through the
environment (`COMRES_PARAM_<NAME>=...`, read by `from_env`; run --set NAME=VALUE) without
touching code. Frozen so a params object shared by several robots cannot be changed under one
of them; use `.replace(**kw)` for a variant.

Names follow docs/comres_spec.md. Every field also accepts the upper-case form
(`COMRES_PARAM_ALPHA=2`); any other COMRES_PARAM_ name is refused.

The defaults are the settings COMRES runs with (tuned on the tuning maps). alpha depends on the
task: by default it is ALPHA_BY_TASK[task], set by `for_task` before a run starts, so every run
records a number.
"""

import dataclasses
import os
from dataclasses import dataclass
from typing import Dict, Optional

ENV_PREFIX = "COMRES_PARAM_"
# The weight of the assigned piece in a robot's frontier score, per task (tuned on the tuning maps).
ALPHA_BY_TASK = {"search": 0.5, "explore": 0.25}


@dataclass(frozen=True)
class Params:
    # --- M1: decomposition (units.py) ---
    # A_min is the only number of the decomposition: units smaller than this are merged, and it
    # is also the smallest piece and the smallest continued corridor. The piece size (one sensor
    # view, pi * SENSOR_RANGE^2) comes from the sensor, not from a parameter.
    A_min: int = 20
    # --- M2: belief ---
    # eps: per-cell belief floor, b(g) ~ eps + sum_j p_j 1[g in P_j]/|P_j| (the support is the
    # piece P_j).
    eps: float = 1e-3
    # d_det must equal the robots' sensor range (experiment_config.SENSOR_RANGE).
    d_det: int = 20           # radius of B(f), the belief mass visible from a frontier (= sensor range, 20)
    # --- M3: planner ---
    T_H: int = 100            # replanning horizon = map sync interval (steps)
    # --- M4: selector ---
    alpha: Optional[float] = None   # weight of the assigned-piece indicator (None: ALPHA_BY_TASK[task])
    beta: float = 0.5         # weight of normalised visible belief mass
    lam: float = 1.0          # weight of normalised geodesic distance (subtracted)
    r_C: float = 5.0          # candidates within r_C of another robot's committed waypoint are excluded
    # --- time ---
    dt: float = 1.0           # seconds per simulation step (plan release = ceil(latency / dt))

    def replace(self, **kw) -> "Params":
        return dataclasses.replace(self, **kw)

    def asdict(self) -> Dict:
        return dataclasses.asdict(self)

    def for_task(self, task: str) -> "Params":
        """These params with alpha set to the task's value when it is not set explicitly."""
        return self if self.alpha is not None else self.replace(alpha=ALPHA_BY_TASK[task])


def _parse_as(typ, name: str, raw: str):
    if typ == Optional[float]:
        typ = float
    if typ is int:
        v = float(raw)                      # accept "30" and "30.0", reject "30.5"
        if v != int(v):
            raise ValueError(f"{name} must be an integer, got {raw!r}")
        return int(v)
    if typ is float:
        return float(raw)
    return raw.strip()


def env_names(name: str) -> tuple:
    """Environment variables that override field `name`, in lookup order."""
    return (ENV_PREFIX + name,) + ((ENV_PREFIX + name.upper(),) if name.upper() != name else ())


def from_env(base: Optional[Params] = None, environ=None) -> Params:
    """`base` (default Params()) with every COMRES_PARAM_<NAME> override applied, typed. An empty
    value means unset; a name that is no field (e.g. a removed one) is refused."""
    env = os.environ if environ is None else environ
    base = Params() if base is None else base
    fields = dataclasses.fields(Params)
    known = {v for f in fields for v in env_names(f.name)}
    unknown = sorted(k for k, v in env.items() if k.startswith(ENV_PREFIX) and v != "" and k not in known)
    if unknown:
        raise ValueError(f"unknown parameter(s) {unknown}; Params has {[f.name for f in fields]}")
    kw = {}
    for f in fields:
        for var in env_names(f.name):
            if var in env and env[var] != "":
                kw[f.name] = _parse_as(f.type, f.name, env[var])
                break
    return base.replace(**kw) if kw else base
