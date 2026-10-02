"""
The no_decomp ablation arm (comres:no_decomp): COMRES without the zones and pieces.

The VLM sees only the merged map, with a coordinate frame, and answers with a GOAL POINT per robot
and the target guesses as POINTS (search with a hint only). Nothing of units.py runs.

- Picture (render_points_png): the size, scale and colours of comres_planner.render_png: the known
  map in plain grayscale, the robots as numbered discs with arrows to where each is driving, the
  team start S and the north arrow, plus thin dotted gray grid lines every GRID_STEP cells and
  their numbers along the bottom (x) and left (y) edges. No tints, labels, ways in or lines.
- Coordinates: x = the column (cells east from the west edge, 0..W-1), y = the row (cells north
  from the south edge, 0..H-1; the maps are drawn with row 0 at the bottom). A point (x, y) is the
  grid cell (row y, col x).
- Text (build_points_prompt_text): the task and hint, the picture's legend, the robots' positions
  and the spots they are driving to as (x, y), how the plan is used, the search rules for points
  and the reply format. No table and no current pieces (there are none).
- Schema (points_reply_schema): {"hypotheses": [{"x", "y", "p"}] (0..10), "goals": {"1": {"x",
  "y"}, ...}} with x and y integers on the map.
- Plan (PointPlan): each robot's goal cell, the hypotheses as cells with their p (normalised as
  comres_planner.parse_reply does), and each hypothesis's support: the unknown cells of the merged
  map within d_det (the sensor range, Euclidean) of its point. The belief is built from these as
  from pieces (belief.Belief.reset).
- The robots (comres_strategy, selector.select(goal=...)): while a robot's goal cell is unknown on
  its OWN map, the plan term is its goal closeness A(f) = 1 - dist(rep_f, goal) / (max over its
  remaining candidates), straight-line; once the goal cell is seen, no plan term until the next
  plan (as a robot whose piece has no frontier left).

Latency is charged as for the method: the snapshot, this module's CPU part and the VLM call.
"""

import base64
import dataclasses
import io
import json
import time
from dataclasses import dataclass
from typing import Any, Dict, List, Optional, Tuple

import numpy as np

import comres_planner as CP
import vlm_client
from params import Params

Point = Tuple[int, int]
GRID_STEP = 10                  # grid lines and tick numbers every this many cells
GRID_RGB = (150, 150, 150)      # the dotted grid lines
TICK_RGB = (60, 60, 60)         # tick marks and numbers
TICK_PX = 12                    # font size of the numbers
MAX_HYPS = 10


@dataclass
class PointPlan:
    goals: Dict[int, Point]                  # robot index (0-based) -> goal cell (row, col); missing = none
    hyps: List[Tuple[Point, float]]          # (cell (row, col), p), p sums to 1; [] = uniform belief
    omegas: List[np.ndarray]                 # support of each hypothesis (unknown cells within d_det)
    query_step: int


def _xy(cell) -> Tuple[int, int]:
    """Grid cell (row, col) -> the prompt's point (x, y)."""
    return int(cell[1]), int(cell[0])


# ---------------------------------------------------------------------------
# Picture
# ---------------------------------------------------------------------------

def render_points_png(snap: CP.Snapshot) -> bytes:
    """The no_decomp picture as PNG bytes (see the module docstring)."""
    from PIL import Image, ImageDraw
    occ = snap.occ
    H, W = occ.shape
    s = CP._scale(occ)
    rgb = np.empty((H, W, 3), np.uint8)
    rgb[:] = CP.FREE_RGB
    rgb[occ == -1] = CP.UNKNOWN_RGB
    rgb[occ == 1] = CP.OCC_RGB
    canvas = np.full((H * s, W * s + CP.MARGIN_PX, 3), 255, np.uint8)
    canvas[:, :W * s] = np.repeat(np.repeat(rgb[::-1], s, axis=0), s, axis=1)
    im = Image.fromarray(canvas)
    dr = ImageDraw.Draw(im)

    def xy(p):                               # cell centre -> pixel, row 0 at the bottom
        return ((p[1] + 0.5) * s, (H - 1 - p[0] + 0.5) * s)

    # Dotted grid lines through the cells whose x (column) or y (row) is a multiple of GRID_STEP.
    for c in range(GRID_STEP, W, GRID_STEP):
        x = xy((0, c))[0]
        for y in range(0, H * s, 6):
            dr.line([(x, y), (x, min(y + 2, H * s - 1))], fill=GRID_RGB, width=1)
    for r in range(GRID_STEP, H, GRID_STEP):
        y = xy((r, 0))[1]
        for x in range(0, W * s, 6):
            dr.line([(x, y), (min(x + 2, W * s - 1), y)], fill=GRID_RGB, width=1)
    dr.rectangle([0, 0, W * s - 1, H * s - 1], outline=(110, 110, 110), width=1)
    # Tick marks and numbers just inside the bottom (x) and left (y) edges; "0" once, in the corner.
    f = CP._font(TICK_PX)
    for c in range(0, W, GRID_STEP):
        x = xy((0, c))[0]
        dr.line([(x, H * s - 1), (x, H * s - 6)], fill=TICK_RGB, width=1)
        t = str(c)
        l, tp, rt, bt = dr.textbbox((0, 0), t, font=f, stroke_width=2)
        bx = min(max(x - (rt - l) / 2, 2), W * s - (rt - l) - 2)
        dr.text((bx - l, H * s - 8 - (bt - tp) - tp), t, font=f, fill=TICK_RGB, stroke_width=2,
                stroke_fill=(255, 255, 255))
    for r in range(GRID_STEP, H, GRID_STEP):
        y = xy((r, 0))[1]
        dr.line([(0, y), (5, y)], fill=TICK_RGB, width=1)
        t = str(r)
        l, tp, rt, bt = dr.textbbox((0, 0), t, font=f, stroke_width=2)
        by = min(max(y - (bt - tp) / 2, 2), H * s - (bt - tp) - 2)
        dr.text((8 - l, by - tp), t, font=f, fill=TICK_RGB, stroke_width=2, stroke_fill=(255, 255, 255))

    sh = 9
    true_xy, disc_xy, moved = CP.robot_discs(snap)
    starts = CP._points(snap.team_start)
    start_xy = xy(np.mean(np.asarray(starts, float), axis=0)) if starts else None
    CP._draw_waypoint_arrows(dr, snap, true_xy, xy)
    CP._draw_start_robots_north(dr, start_xy, sh, true_xy, disc_xy, moved, W, s)
    buf = io.BytesIO()
    im.save(buf, format="PNG", compress_level=3)
    return buf.getvalue()


# ---------------------------------------------------------------------------
# Prompt and schema
# ---------------------------------------------------------------------------

def _search_rules_points(n: int, W: int, H: int) -> List[str]:
    """The search RULES with a hint, for points (the pieces' rules with pieces made points)."""
    hyp = ("- hypotheses: points (x, y) in unknown space where the target most likely is, given the hint and "
           "what is already explored, each with probability p. Give each p as honestly as you can: a hint can "
           "be misread, and your single top guess is right less than half the time. List your 2 or 3 most "
           "likely points, unless the hint and the map together clearly point to one place.")
    goals = ("- goals: start with your most likely guess: send its nearest robots there, but never more than "
             "half of the robots (rounded up) near one guess. Send the other robots to your next most likely "
             "guesses, and any robots still left to nearby large unexplored areas. Put the robots in at least "
             "two different parts of the building whenever you can: a wrong place costs the most when every "
             "robot is there.")
    ids = (f"- every robot 1..{n} gets exactly 1 goal point; x is a whole number from 0 to {W - 1}, y from 0 "
           f"to {H - 1}.")
    return [hyp, CP.EMPTY_HYPS, goals, ids]


def build_points_prompt_text(snap: CP.Snapshot, params: Optional[Params] = None) -> str:
    """The no_decomp prompt text (search with a hint only)."""
    params = params or Params()
    if snap.task != "search" or not snap.hint:
        raise ValueError("the no_decomp variant asks a search with a hint only")
    n = len(snap.robot_pos)
    H, W = snap.occ.shape
    L = [f"You coordinate a team of {n} robots searching a partly explored building for a target. "
         "Goal: find the target as quickly as possible. It has not been seen yet, so it lies "
         "somewhere in the unknown (white) space.",
         f'Hint about the target: "{snap.hint}"']
    moved = any(CP.robot_discs(snap)[2])
    L += ["", f"IMAGE: the team's merged map, north up. The map is {W} cells east-west by {H} cells "
              f"north-south; 1 cell = {CP._scale(snap.occ)} px in the image.",
          "- light gray = explored free space; black = walls and obstacles; white = unknown.",
          f"- a point (x, y) is a cell: x = its column, counted east from the west edge (0 to {W - 1}); "
          f"y = its row, counted north from the south edge (0 to {H - 1}). The numbers along the bottom edge "
          f"are x and the numbers along the left edge are y, every {GRID_STEP} cells; thin dotted gray lines "
          f"mark every {GRID_STEP} cells.",
          f"- numbered circles = robots 1..{n}; an arrow points from a robot to the spot it was driving to "
          "(chosen on its own map before the maps were merged, so this map may show it already explored)."
          + (" Circles that would cover each other are drawn apart, each joined to its robot's true cell "
             "by a thin line ending in a dot." if moved else ""),
          "- black square S = the team's start."]
    L += ["", "ROBOTS (x, y):"]
    for i, (pos, w) in enumerate(zip(snap.robot_pos, snap.waypoints)):
        x, y = _xy(pos)
        spot = (f"driving to ({_xy(w)[0]}, {_xy(w)[1]})" if w is not None and tuple(w) != tuple(pos)
                else "no spot it was driving to")
        L.append(f"Robot {i + 1}: at ({x}, {y}), {spot}")
    L += ["", "HOW YOUR PLAN IS USED:",
          "Robots always drive to frontiers (boundaries between explored and unknown space), one cell per step.",
          "You give each robot a goal point (x, y). It prefers the frontiers nearest to its goal (straight-line "
          "distance), until it has seen its goal point; then it picks frontiers on its own."]
    if float(params.beta) > 0:
        L.append("Robots also prefer frontiers that look into unknown space near your target guesses.")
    L.append(f"A new plan is made every {int(params.T_H)} steps; yours is used until then.")
    L += ["", "RULES:"] + _search_rules_points(n, W, H)
    g = ", ".join(f'"{i + 1}": {{"x": <x>, "y": <y>}}' for i in range(n))
    L += ["", "Reply with only this JSON object:",
          '{"hypotheses": [{"x": <x>, "y": <y>, "p": <number 0..1>}, ...], "goals": {' + g + "}}"]
    return "\n".join(L)


def points_reply_schema(snap: CP.Snapshot) -> Dict:
    """JSON schema of the no_decomp reply: points on the map, one goal per robot (all required)."""
    H, W = snap.occ.shape
    X = {"type": "integer", "minimum": 0, "maximum": W - 1}
    Y = {"type": "integer", "minimum": 0, "maximum": H - 1}
    point = {"type": "object", "additionalProperties": False, "required": ["x", "y"],
             "properties": {"x": X, "y": Y}}
    keys = [str(i + 1) for i in range(len(snap.robot_pos))]
    return {"type": "object", "additionalProperties": False, "required": ["hypotheses", "goals"],
            "properties": {
                "hypotheses": {"type": "array", "minItems": 0, "maxItems": MAX_HYPS,
                               "items": {"type": "object", "additionalProperties": False,
                                         "required": ["x", "y", "p"],
                                         "properties": {"x": X, "y": Y,
                                                        "p": {"type": "number", "minimum": 0, "maximum": 1}}}},
                "goals": {"type": "object", "additionalProperties": False, "required": keys,
                          "properties": {k: point for k in keys}}}}


def build_points_messages(snap: CP.Snapshot, params: Optional[Params] = None,
                          png: Optional[bytes] = None) -> List[Dict]:
    """One user message: the picture, then the text."""
    png = png if png is not None else render_points_png(snap)
    b64 = base64.b64encode(png).decode("ascii")
    return [{"role": "user", "content": [
        {"type": "image_url", "image_url": {"url": f"data:image/png;base64,{b64}"}},
        {"type": "text", "text": build_points_prompt_text(snap, params)},
    ]}]


# ---------------------------------------------------------------------------
# Reply parsing and the plan
# ---------------------------------------------------------------------------

def _coord(v, hi: int) -> Optional[int]:
    """An integer coordinate in 0..hi, or None."""
    if isinstance(v, bool) or not isinstance(v, (int, float)) or not np.isfinite(v) or float(v) != int(v):
        return None
    v = int(v)
    return v if 0 <= v <= hi else None


def parse_points_reply(text: str, W: int, H: int, n_robots: int
                       ) -> Tuple[Dict[int, Tuple[int, int]], List[Tuple[Tuple[int, int], float]]]:
    """({robot index: goal (x, y)}; normalised hypotheses [((x, y), p)], repeated points summed, as
    comres_planner.parse_reply does for pieces). A robot without a valid point gets no goal; a
    hypothesis off the map is dropped. Raises if the reply is not JSON."""
    start, end = text.find("{"), text.rfind("}") + 1
    if start == -1 or end <= start:
        raise ValueError("no JSON object in the reply")
    data = json.loads(text[start:end])
    goals: Dict[int, Tuple[int, int]] = {}
    for i in range(n_robots):
        g = (data.get("goals") or {}).get(str(i + 1))
        if isinstance(g, dict):
            x, y = _coord(g.get("x"), W - 1), _coord(g.get("y"), H - 1)
            if x is not None and y is not None:
                goals[i] = (x, y)
    acc: Dict[Tuple[int, int], float] = {}
    for h in data.get("hypotheses") or []:
        if not isinstance(h, dict):
            continue
        x, y = _coord(h.get("x"), W - 1), _coord(h.get("y"), H - 1)
        if x is None or y is None:
            continue
        p = h.get("p", 0.0)
        p = float(p) if isinstance(p, (int, float)) and np.isfinite(p) else 0.0
        acc[(x, y)] = acc.get((x, y), 0.0) + max(p, 0.0)
    tot = sum(acc.values())
    hyps = [(pt, (p / tot) if tot > 0 else 1.0 / len(acc)) for pt, p in acc.items()]
    return goals, hyps


def point_support(occ: np.ndarray, cell: Point, radius: float) -> np.ndarray:
    """The unknown cells of `occ` within `radius` (Euclidean, inclusive) of `cell` (row, col)."""
    H, W = occ.shape
    rr, cc = np.ogrid[:H, :W]
    return (occ == -1) & ((rr - cell[0]) ** 2 + (cc - cell[1]) ** 2 <= float(radius) ** 2)


def plan_points(planner, snap: CP.Snapshot) -> Tuple[Optional[PointPlan], Dict]:
    """VLMPlanner.plan for the no_decomp variant: one VLM call, the reply turned into a PointPlan.
    The record keeps the goals and hypotheses as the VLM gave them (x, y) and the timings; the CPU
    part charged is everything but the VLM call."""
    t0 = time.perf_counter()
    snap = dataclasses.replace(snap, task=planner.task, hint=planner.hint)
    H, W = snap.occ.shape
    rec: Dict[str, Any] = dict(step=int(snap.step), n_robots=planner.n)
    t1 = time.perf_counter()
    png = render_points_png(snap)
    t2 = time.perf_counter()
    messages = build_points_messages(snap, planner.params, png)
    schema = points_reply_schema(snap)
    t3 = time.perf_counter()
    planner.last_messages, planner.last_schema = messages, schema
    rec.update(render_seconds=round(t2 - t1, 4), build_seconds=round(t3 - t2, 4))
    text, usage, err = "", {}, None
    try:
        text, usage = vlm_client.call(messages, schema, model=planner.model, base_url=planner.base_url)
    except Exception as e:                      # server down / HTTP error: no plan
        err = f"{type(e).__name__}: {e}"
    t4 = time.perf_counter()
    if err is not None:
        rec["failed_call_seconds"] = round(t4 - t3, 4)
    plan, goals, hyps = None, {}, []
    if err is None:
        try:
            goals, hyps = parse_points_reply(text, W, H, planner.n)
            d_det = float(planner.params.d_det)
            cells = [((y, x), p) for (x, y), p in hyps]
            plan = PointPlan(goals={i: (y, x) for i, (x, y) in goals.items()}, hyps=cells,
                             omegas=[point_support(snap.occ, c, d_det) for c, _ in cells],
                             query_step=int(snap.step))
        except Exception as e:
            err = f"{type(e).__name__}: {e}"
    t5 = time.perf_counter()
    cpu = (t3 - t0) + (t5 - t4)
    rec.update(usage)
    rec.update(parse_seconds=round(t5 - t4, 4), cpu_seconds=round(max(cpu, 0.0), 4),
               raw=text, valid=plan is not None, error=err,
               goals={str(i + 1): [x, y] for i, (x, y) in goals.items()},
               hyps=[[x, y, round(p, 4)] for (x, y), p in hyps])
    rec.setdefault("call_seconds", 0.0)
    rec.setdefault("lock_wait_seconds", 0.0)
    return plan, rec
