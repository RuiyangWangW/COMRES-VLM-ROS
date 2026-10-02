"""
M3 of COMRES-VLM: the VLM planner.

At a sync step the strategy freezes what the planner may see into a `Snapshot`: the merged
map, the zones and their pieces (units.py), the search belief, the robots, their committed
waypoints and their current pieces. The planner draws it, writes the table, asks the VLM for
one piece per robot (and, in search, target hypotheses over pieces) and turns the reply into
a `Plan`. The VLM never outputs waypoints: the selector (M4) picks frontiers, and the plan
only biases its reward, so a stale plan degrades gracefully. IDs are constrained by the JSON
schema to the pieces of the table, so there is no validation or repair step; a reply that
does not parse (or an empty fake reply) means no plan.

Zones and pieces: a zone is the unexplored space behind one way in (units.decompose: the
unknown space closest to one opening, small ones merged); a zone bigger than one sensor view is
cut into pieces (units.split_pieces: the corridor continued from a way in plus sensor-view
pieces). The snapshot carries the piece grid (`piece_labels`, region ids), `piece_desc` (name
"9a" / "9", zone, size, beyond, corridor, centre line, way in) and `op_piece` (the piece behind
each opening); `unit_labels` is each cell's zone. The VLM sees zones Z<k> and pieces
Z<k><letter>, named anew at every plan ("a" = the piece at the way in, then the continued
corridors, then the others; a zone that is not cut is one piece named Z<k>).

Image: the known map in plain grayscale (unknown white, free light gray, walls black) with the
map's size and scale in the prompt; each piece tinted with its colour and labelled with its
name (labels placed smallest piece first; one that does not fit goes beside its piece in a
smaller font, to the nearest spot that covers little of other pieces or white space), a thin
dark line between zones, each way in bold in the colour of the piece behind it with its zone's
id beside it (not for a zone that is not cut when its label is close), a thin dashed centre
line along every continued corridor, robots numbered with arrows to their goals (discs spread
until none overlap), the team start and a north arrow.
Table: one line per zone (way-in width, d1..dM to its nearest way in), and under a cut zone one
indented line per piece (size, beyond = robot steps from the way in to the piece, corridor, m);
a zone that is not cut is one line with both parts. Current pieces: each robot's piece of the
previous plan under its name in this table, or "gone" (never an old name). Reply: one piece
per robot, hypotheses [{"piece", "p"}] (0 or more). Plans carry `ids` (name -> region id) and
`piece_unit` (region id -> zone).

Hypotheses (search only) are uncapped and may be empty: when the map does not yet tell where the
target is, the robots keep a uniform belief. Each robot gets exactly one piece.

Instructions: explore asks to spread the robots over large pieces, weighing distance. A search
with a hint adds the search RULES (_search_rules_pieces: honest probabilities -- a hint can be
misread; the top guess is right less than half the time; 2 or 3 pieces unless the hint and the
map clearly point to one -- never more than half of the robots on one piece, the others to the
next guesses and then the nearest large pieces, and at least two zones when possible). A search
without a hint is asked exactly as an explore query (query_view). Thinking is off.

Ablation arms (VARIANTS; search with a hint only): "no_table" asks without the table,
"no_history" without the robots' current pieces and the m column, "no_picture" with the text
only (no picture drawn; the IMAGE legend becomes TERMS: its definitions without the drawing); "no_decomp" asks for goal points on the plain map
(comres_points.py). The picture, schema and every other sentence stay the method's.

`Snapshot.save/load` round-trips a snapshot (tools/warmup.py sends two saved ones to warm a
fresh server up). `target` is stored with it only and is never drawn or written into the prompt.

Grid convention: (row, col); -1 unknown, 0 free, 1 occupied. The image is drawn with
row 0 at the bottom (origin="lower"), so row index grows north.
"""

import base64
import dataclasses
import io
import json
import math
import os
import time
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Sequence, Tuple

import numpy as np
from scipy import ndimage

import units as unitlib
import vlm_client
from units import ID_PREFIX, WORD as UNIT_WORD
from params import Params

PIECE_WORD = "piece"      # what the VLM assigns (hint #279 said "pieces" for shelf blocks; reworded)
# A listed piece of the previous plan that no longer exists: shown so, never under an old name
# (names are given anew for every map, so an old name may name another place now).
GONE = "(gone)"
TASKS = ("explore", "search")
# The method ("full") and its ablation variants (search with a hint only; docs/comres_spec.md,
# "Ablation arms"): no_table, no_history and no_picture leave one part out of this module's
# question; no_decomp (comres_points.py) asks for goal points on the plain map instead of pieces.
VARIANTS = ("full", "no_table", "no_history", "no_picture", "no_decomp")


Point = Tuple[int, int]


# ---------------------------------------------------------------------------
# Snapshot
# ---------------------------------------------------------------------------

@dataclass
class Snapshot:
    """
    What the planner may see at one sync step. Built by the strategy (or loaded from a
    file). Inputs are normalised in __post_init__: cells become int tuples, `lists` one list
    of piece names per robot ("Z9a", or comres_planner.GONE), `team_start` one (r, c) or a
    list of cells. The strategy always fills in the zones and the piece grid.
    """
    task: str                                   # "explore" | "search"
    hint: str                                   # search hint, verbatim ("" for explore)
    step: int                                   # sync step of the query
    occ: np.ndarray                             # merged map (-1/0/1)
    robot_pos: List[Point]                      # robot i position (row, col), 0-based i
    waypoints: List[Optional[Point]]            # committed waypoint per robot (None: none)
    lists: List[List[str]]                      # each robot's current pieces ("Z9a", ...)
    team_start: List[Point]                     # the team's start cells (marker at their mean)
    belief_map: Optional[np.ndarray] = None     # search belief over unknown cells, sums to 1
    target: Optional[Point] = None              # offline scoring only, NEVER rendered
    map_variant: str = ""
    unit_id: str = ""
    meta: Dict = field(default_factory=dict)    # free-form JSON-able extras (arm, seed, ...)
    # Zones (units.py): each unknown cell's zone, the openings and the zone descriptors.
    unit_labels: Optional[np.ndarray] = None    # zone id per unknown cell, 0 = none
    olabel: Optional[np.ndarray] = None         # opening index per frontier cell, 0 = none
    opening_unit: Dict[int, int] = field(default_factory=dict)   # opening index -> zone id (0 = none)
    unit_desc: Dict[int, Dict] = field(default_factory=dict)     # zone id -> units.descriptors() row
    passage_width: Optional[float] = None       # w of the spread (for the record)
    # Pieces: what the VLM assigns and places hypotheses on.
    piece_labels: Optional[np.ndarray] = None   # region id per unknown cell, 0 = none
    piece_desc: Dict[int, Dict] = field(default_factory=dict)    # region id -> units.piece_descriptors() row
    op_piece: Dict[int, int] = field(default_factory=dict)       # opening index -> region id at that way in

    def __post_init__(self):
        self.occ = np.asarray(self.occ)
        self.step = int(self.step)
        self.hint = self.hint or ""
        self.robot_pos = [_cell(p) for p in self.robot_pos]
        n = len(self.robot_pos)
        wps = list(self.waypoints) if self.waypoints is not None else []
        self.waypoints = [None if (i >= len(wps) or wps[i] is None) else _cell(wps[i]) for i in range(n)]
        self.team_start = [_cell(p) for p in _points(self.team_start)]
        if self.target is not None:
            self.target = _cell(self.target)
        if self.unit_labels is not None:
            self.unit_labels = np.asarray(self.unit_labels, np.int32)
            if self.olabel is None:
                raise ValueError("a snapshot with zones needs olabel (the opening grid) too")
            self.olabel = np.asarray(self.olabel, np.int32)
            self.opening_unit = {int(j): int(k) for j, k in (self.opening_unit or {}).items()}
            self.unit_desc = {int(k): dict(v) for k, v in (self.unit_desc or {}).items()}
        if self.piece_labels is not None:
            if self.unit_labels is None:
                raise ValueError("a pieces snapshot needs unit_labels and olabel (the zones) too")
            self.piece_labels = np.asarray(self.piece_labels, np.int32)
            self.piece_desc = {int(k): dict(v) for k, v in (self.piece_desc or {}).items()}
            self.op_piece = {int(j): int(k) for j, k in (self.op_piece or {}).items()}
        lists = [[str(u) for u in l] for l in (self.lists or [])]
        self.lists = [lists[i] if i < len(lists) else [] for i in range(n)]

    @property
    def has_pieces(self) -> bool:
        """True when the snapshot carries the piece grid (every snapshot the strategy makes)."""
        return self.piece_labels is not None

    @property
    def n_robots(self) -> int:
        return len(self.robot_pos)

    def save(self, path: str) -> str:
        """One compressed .npz: arrays as entries, everything else as a JSON string. Only a pieces
        snapshot (every snapshot the strategy makes) can be saved."""
        if not self.has_pieces:
            raise ValueError("only a pieces snapshot can be saved")
        arrays: Dict[str, np.ndarray] = {"occ": np.asarray(self.occ, dtype=np.int8),
                                         "unit_labels": np.asarray(self.unit_labels, np.int32),
                                         "olabel": np.asarray(self.olabel, np.int32),
                                         "piece_labels": np.asarray(self.piece_labels, np.int32)}
        if self.belief_map is not None:
            arrays["belief_map"] = np.asarray(self.belief_map, dtype=np.float64)
        meta = dict(task=self.task, hint=self.hint, step=int(self.step),
                    robot_pos=[_pt(p) for p in self.robot_pos],
                    waypoints=[None if w is None else _pt(w) for w in self.waypoints],
                    lists=[list(map(str, l)) for l in (self.lists or [])],
                    team_start=[_pt(p) for p in _points(self.team_start)],
                    target=None if self.target is None else _pt(self.target),
                    map_variant=self.map_variant, unit_id=self.unit_id, meta=_jsonable(self.meta),
                    opening_unit=[[int(j), int(k)] for j, k in sorted(self.opening_unit.items())],
                    unit_desc=[[int(k), _desc_json(v)] for k, v in sorted(self.unit_desc.items())],
                    passage_width=self.passage_width,
                    piece_desc=[[int(k), _jsonable(v)] for k, v in sorted(self.piece_desc.items())],
                    op_piece=[[int(j), int(k)] for j, k in sorted(self.op_piece.items())])
        arrays["meta_json"] = np.array(json.dumps(meta))
        if not path.endswith(".npz"):
            path += ".npz"
        os.makedirs(os.path.dirname(os.path.abspath(path)), exist_ok=True)
        np.savez_compressed(path, **arrays)
        return path

    @classmethod
    def load(cls, path: str) -> "Snapshot":
        """A snapshot written by `save`; a file without the piece grid is refused."""
        with np.load(path, allow_pickle=False) as z:
            arrays = {k: z[k] for k in z.files}
        if "piece_labels" not in arrays:
            raise ValueError(f"{path}: not a pieces snapshot (no piece grid)")
        meta = json.loads(str(arrays.pop("meta_json")))
        tup = lambda p: None if p is None else (int(p[0]), int(p[1]))
        return cls(task=meta["task"], hint=meta["hint"], step=int(meta["step"]),
                   occ=arrays["occ"], robot_pos=[tup(p) for p in meta["robot_pos"]],
                   waypoints=[tup(w) for w in meta["waypoints"]], lists=[list(l) for l in meta["lists"]],
                   team_start=[tup(p) for p in meta["team_start"]], belief_map=arrays.get("belief_map"),
                   target=tup(meta["target"]), map_variant=meta["map_variant"],
                   unit_id=meta["unit_id"], meta=meta.get("meta", {}),
                   unit_labels=arrays["unit_labels"], olabel=arrays["olabel"],
                   opening_unit={int(j): int(k) for j, k in meta["opening_unit"]},
                   unit_desc={int(k): _desc_from_json(v) for k, v in meta["unit_desc"]},
                   passage_width=meta.get("passage_width"), piece_labels=arrays["piece_labels"],
                   piece_desc={int(k): dict(v) for k, v in meta["piece_desc"]},
                   op_piece={int(j): int(k) for j, k in meta["op_piece"]})


def _desc_json(d: Dict) -> Dict:
    """A descriptor row as JSON (inf -> "inf")."""
    out = {}
    for k, v in d.items():
        if k == "d_geo":
            out[k] = [None if x is None else ("inf" if not np.isfinite(x) else float(x)) for x in v]
        else:
            out[k] = _jsonable(v)
    return out


def _desc_from_json(d: Dict) -> Dict:
    out = dict(d)
    if "d_geo" in out:
        out["d_geo"] = [None if x is None else (float("inf") if x == "inf" else float(x)) for x in out["d_geo"]]
    return out


def _pt(p) -> List[int]:
    return [int(p[0]), int(p[1])]


def _cell(p) -> Point:
    return int(p[0]), int(p[1])


def _points(ps) -> List:
    """team_start may be one (r, c) or a list of them."""
    if ps is None:
        return []
    if isinstance(ps, np.ndarray):
        ps = ps.tolist()
    ps = list(ps)
    if len(ps) == 2 and np.isscalar(ps[0]):
        return [ps]
    return ps


def _id_key(uid: str):
    return (uid[:1], int(uid[1:])) if uid[1:].isdigit() else (uid, 0)


def _jsonable(v):
    if isinstance(v, dict):
        return {str(k): _jsonable(x) for k, x in v.items()}
    if isinstance(v, (list, tuple)):
        return [_jsonable(x) for x in v]
    if isinstance(v, np.ndarray):
        return v.tolist()
    if isinstance(v, np.generic):
        return v.item()
    return v


# ---------------------------------------------------------------------------
# Plan
# ---------------------------------------------------------------------------

@dataclass
class Plan:
    lists: Dict[int, List[str]]          # robot index (0-based) -> [its piece name] ([] = none)
    hyps: List[Tuple[str, float]]        # (piece name, p), p sums to 1; [] = uniform belief
    omegas: Dict[str, np.ndarray]        # piece name -> hypothesis support (the piece's cells)
    labels: np.ndarray                   # plan-time piece grid (region ids)
    query_step: int
    # the name shown ("Z9a") -> region id in `labels`, and each region id's zone
    ids: Dict[str, int] = field(default_factory=dict)
    piece_unit: Dict[int, int] = field(default_factory=dict)


# ---------------------------------------------------------------------------
# Pieces and the table
# ---------------------------------------------------------------------------

@dataclass
class Units:
    """What one query is about: the pieces, their masks and table rows, and the zones."""
    open_ids: List[str]                 # piece names ("Z9a"), grouped by zone, in table order
    labels: np.ndarray                  # the piece grid (region ids)
    open_labels: List[int]              # region id of each name in open_ids
    anchors: Dict[str, Point]           # where each piece's label goes
    masks: Dict[str, np.ndarray]        # name -> bool mask of the piece's cells
    rows: List[Dict]                    # one per piece: id, size, beyond, corridor, m, dgeo
    belief: Optional[np.ndarray]        # belief over unknown actually used (None: none)
    prep_seconds: float
    olabel: np.ndarray                  # the opening grid
    opening_unit: Dict[int, int]        # opening -> zone
    op_piece: Dict[int, int]            # opening -> region id behind it
    zone_grid: np.ndarray               # zone id per unknown cell (the zone of its piece)
    zones: List[Dict]                   # zone rows: id, unit, width, dgeo, pieces
    lines: Dict[str, List]              # continued corridor -> centre line
    ids: Dict[str, int]                 # name -> region id
    piece_unit: Dict[int, int]          # region id -> zone


def uses_belief(task: str) -> bool:
    """Search keeps a belief: the m column in the table and the hypotheses in the reply."""
    return task == "search"


def query_view(snap: Snapshot) -> Snapshot:
    """The snapshot as the VLM query sees it. A search without a hint is asked exactly as an explore
    query: the explore image, table, instructions and schema, so no hypotheses and a uniform
    belief. Everything else is unchanged."""
    if snap.task == "search" and not snap.hint:
        return dataclasses.replace(snap, task="explore")
    return snap


def _snapshot_belief(snap: Snapshot) -> Optional[np.ndarray]:
    """The belief the query uses: the snapshot's over the unknown cells, normalised (search)."""
    if not uses_belief(snap.task):
        return None
    unk = snap.occ == -1
    if snap.belief_map is not None:
        belief = np.where(unk, snap.belief_map, 0.0)
    else:
        belief = unk.astype(np.float64)
    s = belief.sum()
    return belief / s if s > 0 else belief


def prepare_units(snap: Snapshot) -> Units:
    """The pieces of a snapshot as the query sees them (query_view): every piece is assignable
    and its own hypothesis support. Rows (one per piece, grouped by zone: zones by id, pieces by
    letter): size, beyond (robot steps past the way in), corridor, m, dgeo (the path from each
    robot's goal to the zone's way in plus beyond; hungarian_region's cost). Zone rows: way-in
    width (all its openings) and d_geo from each robot's goal to its nearest opening (the
    snapshot's unit_desc: units.descriptors of the zone grid, which every recorded pieces
    snapshot carries)."""
    if not snap.has_pieces:
        raise ValueError("not a pieces snapshot (no piece grid)")
    t0 = time.perf_counter()
    snap = query_view(snap)
    grid = np.asarray(snap.piece_labels, np.int32)
    zlab = np.asarray(snap.unit_labels, np.int32)
    belief = _snapshot_belief(snap)
    n = len(snap.robot_pos)
    pdesc = snap.piece_desc
    rids = [int(k) for k in np.unique(grid[grid > 0])]
    if any(r not in pdesc for r in rids):
        raise ValueError("a pieces snapshot needs piece_desc for every piece")
    zdesc = snap.unit_desc
    zids = sorted({int(pdesc[r]["unit"]) for r in rids})
    if any(k not in zdesc or len(zdesc[k]["d_geo"]) != n for k in zids):
        raise ValueError("a pieces snapshot needs unit_desc (units.descriptors, d_geo per robot) for every zone")
    by_zone: Dict[int, List[int]] = {}
    for r in rids:
        by_zone.setdefault(int(pdesc[r]["unit"]), []).append(r)
    rows, zones, open_ids, open_labels = [], [], [], []
    masks, lines, ids = {}, {}, {}
    for k in zids:
        ps = sorted(by_zone[k], key=lambda r: (unitlib.letter_index(pdesc[r]["letter"]), r))
        d = zdesc[k]
        zones.append(dict(id=f"{ID_PREFIX}{k}", unit=k, width=int(d["width"]),
                          dgeo=[None if (x is None or not np.isfinite(x)) else int(round(x)) for x in d["d_geo"]],
                          pieces=[f"{ID_PREFIX}{pdesc[r]['name']}" for r in ps]))
        for r in ps:
            pd = pdesc[r]
            u = f"{ID_PREFIX}{pd['name']}"
            masks[u] = grid == r
            b = pd.get("beyond")
            b = None if b is None else int(round(float(b)))
            # dgeo of a piece (hungarian_region's cost): the path to the
            # zone's way in plus the walk beyond it; None if either is unknown.
            rows.append(dict(id=u, size=int(pd["size"]), beyond=b, corridor=bool(pd.get("corridor")),
                             m=None if belief is None else float(belief[masks[u]].sum()),
                             dgeo=[None if (d is None or b is None) else d + b for d in zones[-1]["dgeo"]]))
            if pd.get("line"):
                lines[u] = [tuple(c) for c in pd["line"]]
            open_ids.append(u)
            open_labels.append(r)
            ids[u] = r
    anchors = {u: unitlib.piece_anchor(masks[u]) for u in open_ids}
    return Units(open_ids=open_ids, labels=grid, open_labels=open_labels, anchors=anchors, masks=masks,
                 rows=rows, belief=belief, prep_seconds=time.perf_counter() - t0,
                 olabel=np.asarray(snap.olabel, np.int32), opening_unit=dict(snap.opening_unit),
                 op_piece=dict(snap.op_piece), zone_grid=zlab, zones=zones, lines=lines, ids=ids,
                 piece_unit={r: int(pdesc[r]["unit"]) for r in rids})


# ---------------------------------------------------------------------------
# Image
# ---------------------------------------------------------------------------

TARGET_PX = 768
UNKNOWN_RGB = (255, 255, 255)
FREE_RGB = (205, 205, 205)
OCC_RGB = (0, 0, 0)
ROBOT_COLORS = [(230, 25, 75), (60, 180, 75), (67, 99, 216), (245, 130, 49),
                (145, 30, 180), (66, 212, 244), (240, 50, 230), (191, 239, 69)]
MARGIN_PX = 56

_fonts: Dict[int, Any] = {}


def _font(size: int):
    if size not in _fonts:
        from PIL import ImageFont
        try:
            import matplotlib
            path = os.path.join(matplotlib.get_data_path(), "fonts", "ttf", "DejaVuSans-Bold.ttf")
            _fonts[size] = ImageFont.truetype(path, size)
        except Exception:
            _fonts[size] = ImageFont.load_default(size=size)
    return _fonts[size]


class _LabelPlacer:
    """Greedy label placement: the first candidate spot overlapping no placed box."""

    def __init__(self, width: float, height: float):
        self.w, self.h = width, height
        self.boxes: List[Tuple[float, float, float, float]] = []

    def add(self, box):
        self.boxes.append(tuple(box))

    def _free(self, b) -> bool:
        return all(b[2] <= o[0] or b[0] >= o[2] or b[3] <= o[1] or b[1] >= o[3] for o in self.boxes)

    def place(self, x: float, y: float, w: float, h: float, gap: float):
        """
        Top-left (bx, by) of a w x h label beside a marker of radius `gap` > 0 at (x, y) (the
        zone tags beside the ways in; never on top of it), and whether it was moved off its
        first choice.
        """
        dx = w / 2 + gap + 1
        dy = h / 2 + gap + 1
        cands = []
        for k in range(1, 7):                # rings of 8 spots, farther out each time
            for ox, oy in ((0, -1), (0, 1), (1, 0), (-1, 0), (1, -1), (-1, -1), (1, 1), (-1, 1)):
                cands.append((ox * dx * k, oy * dy * k))
        first = None
        for ox, oy in cands:
            bx = min(max(x + ox - w / 2, 1), self.w - w - 1)
            by = min(max(y + oy - h / 2, 1), self.h - h - 1)
            box = (bx - 1, by - 1, bx + w + 1, by + h + 1)
            if first is None:
                first = (bx, by, box)
            if self._free(box):
                self.add(box)
                return bx, by, (ox, oy) != cands[0]
        bx, by, box = first
        self.add(box)
        return bx, by, False


# --- the pieces image --------------------------------------------------------------
UNIT_ALPHA = 0.5           # piece tint over the white unknown
OPENING_DARK = 0.78        # an opening is drawn in the colour of the piece behind it times this ...
OPENING_EDGE_DARK = 0.35   # ... with a 1 px outline in its colour times this
LABEL_DARK = 0.45          # labels, tags and corridor lines: the piece's colour times this
OPENING_PX = 7             # width of an opening's line in pixels, whatever the map's scale
NO_UNIT_OPENING_RGB = (95, 95, 95)    # dotted: an opening into a patch too small to be a zone
UNIT_COLOR_RADIUS = 20     # pieces closer than this (cells) must get clearly different colours
UNIT_MIN_DE = 20.0         # ... tints at least this far apart (CIE76 Delta E) when the palette allows
LABEL_OWN_MIN = 0.6        # a piece label sits on its piece only if this share of its box lies there
TAG_PX = 13                # font size of the zone id tag beside every opening
TAG_DEDUP_PX = 40          # an opening gets no tag if the same id's tag is already this close
ROBOT_R = 11               # robot disc radius (pixels)
# Piece colours: 16, none red, pink or orange (robots 1 and 7), none
# gray (openings without a zone); picked for large Lab distances between their tints at
# UNIT_ALPHA (every pair >= 21 Delta E).
UNIT_PALETTE = [
    (36, 119, 178), (17, 102, 0), (170, 0, 255), (255, 227, 89), (0, 255, 191), (0, 64, 255),
    (191, 255, 0), (213, 128, 255), (0, 128, 106), (26, 0, 102), (103, 255, 89), (89, 241, 255),
    (128, 106, 0), (89, 0, 178), (0, 0, 255), (0, 255, 0),
]


def _lab(rgb) -> np.ndarray:
    """CIE Lab (D65) of sRGB colours in 0..255, shape (..., 3)."""
    c = np.asarray(rgb, float) / 255.0
    c = np.where(c > 0.04045, ((c + 0.055) / 1.055) ** 2.4, c / 12.92)
    M = np.array([[0.4124, 0.3576, 0.1805], [0.2126, 0.7152, 0.0722], [0.0193, 0.1192, 0.9505]])
    xyz = c @ M.T / np.array([0.95047, 1.0, 1.08883])
    f = np.where(xyz > 216 / 24389, np.cbrt(xyz), (24389 / 27 * xyz + 16) / 116)
    return np.stack([116 * f[..., 1] - 16, 500 * (f[..., 0] - f[..., 1]),
                     200 * (f[..., 1] - f[..., 2])], -1)


def unit_tint(col) -> np.ndarray:
    """A piece colour as drawn over the white unknown space."""
    return 255.0 * (1 - UNIT_ALPHA) + np.asarray(col, float) * UNIT_ALPHA


def _dark(col, f: float = LABEL_DARK) -> Tuple[int, ...]:
    """A colour darkened to f of its value (labels, tags, corridor lines, openings)."""
    return tuple(int(v * f) for v in col)


def _near_units(labels: np.ndarray, ids: Sequence[int], radius: int) -> Dict[int, set]:
    """{label: the labels (pieces) within `radius` cells (Euclidean) of it}."""
    H, W = labels.shape
    near: Dict[int, set] = {k: set() for k in ids}
    for k in ids:
        rr, cc = np.nonzero(labels == k)
        if rr.size == 0:
            continue
        r0, r1 = max(int(rr.min()) - radius, 0), min(int(rr.max()) + radius + 1, H)
        c0, c1 = max(int(cc.min()) - radius, 0), min(int(cc.max()) + radius + 1, W)
        sub = labels[r0:r1, c0:c1]
        d = ndimage.distance_transform_edt(sub != k)
        for o in np.unique(sub[(d <= radius) & (sub > 0)]).tolist():
            if o != k and o in near:
                near[k].add(int(o))
                near[int(o)].add(k)
    return near


def unit_colors(labels: np.ndarray, ids: Sequence[int]) -> Dict[int, Tuple[int, int, int]]:
    """{label: rgb} for the pieces of the piece grid `labels`. Pieces within UNIT_COLOR_RADIUS
    cells of each other get tints at least UNIT_MIN_DE apart (CIE76) when the palette allows
    it: each piece, largest first, takes the first colour from its id's palette slot on that is
    that far from every near piece already coloured, else the colour farthest from them.
    Pieces far apart may share a colour (the prompt says the id is what counts)."""
    ids = [int(k) for k in ids]
    if not ids:
        return {}
    size = {k: int(np.count_nonzero(labels == k)) for k in ids}
    near = _near_units(labels, ids, UNIT_COLOR_RADIUS)
    L = _lab(np.array([unit_tint(c) for c in UNIT_PALETTE]))
    n = len(UNIT_PALETTE)
    idx: Dict[int, int] = {}
    for k in sorted(ids, key=lambda k: (-size[k], k)):
        order = [(k - 1 + t) % n for t in range(n)]
        used = [idx[o] for o in near[k] if o in idx]
        if not used:
            idx[k] = order[0]
            continue
        dmin = {j: float(min(np.linalg.norm(L[j] - L[u]) for u in used)) for j in order}
        ok = [j for j in order if dmin[j] >= UNIT_MIN_DE]
        idx[k] = ok[0] if ok else max(order, key=lambda j: dmin[j])
    return {k: UNIT_PALETTE[idx[k]] for k in ids}


def _scale(occ: np.ndarray) -> int:
    """Pixels per cell of the planner image."""
    return max(1, int(round(TARGET_PX / max(occ.shape))))


def robot_discs(snap: Snapshot) -> Tuple[List[Tuple[float, float]], List[Tuple[float, float]], List[bool]]:
    """(true pixel of every robot, its disc centre, whether the disc was moved). Robots whose
    discs would overlap (closer than 2 * ROBOT_R + 2 px, chained) are spread evenly on a
    small circle around their mean, so no number is hidden; the image joins a moved disc
    to its true cell with a thin line. Two spread groups whose discs still overlap are
    joined and spread again, until no two discs overlap."""
    H, W = snap.occ.shape
    s = _scale(snap.occ)
    true = [((c + 0.5) * s, (H - 1 - r + 0.5) * s) for r, c in snap.robot_pos]
    n = len(true)
    parent = list(range(n))

    def find(a):
        while parent[a] != a:
            parent[a] = parent[parent[a]]
            a = parent[a]
        return a

    lim = 2 * ROBOT_R + 2
    for i in range(n):
        for j in range(i + 1, n):
            if math.hypot(true[i][0] - true[j][0], true[i][1] - true[j][1]) < lim:
                parent[find(i)] = find(j)
    for _ in range(n):
        groups: Dict[int, List[int]] = {}
        for i in range(n):
            groups.setdefault(find(i), []).append(i)
        disc, moved = list(true), [False] * n
        for g in groups.values():
            if len(g) < 2:
                continue
            m = len(g)
            cx = sum(true[i][0] for i in g) / m
            cy = sum(true[i][1] for i in g) / m
            R = (ROBOT_R + 2) / math.sin(math.pi / m)
            for t, i in enumerate(sorted(g)):
                a = -math.pi / 2 + 2 * math.pi * t / m
                x = min(max(cx + R * math.cos(a), ROBOT_R + 1), W * s - ROBOT_R - 1)
                y = min(max(cy + R * math.sin(a), ROBOT_R + 1), H * s - ROBOT_R - 1)
                disc[i], moved[i] = (x, y), True
        joined = False
        for i in range(n):
            for j in range(i + 1, n):
                if find(i) != find(j) and math.hypot(disc[i][0] - disc[j][0], disc[i][1] - disc[j][1]) < lim:
                    parent[find(i)] = find(j)
                    joined = True
        if not joined:
            break
    return true, disc, moved


def _draw_waypoint_arrows(dr, snap: Snapshot, robots_xy, xy) -> None:
    """Each robot's arrow to its committed waypoint (the frontier it is heading to)."""
    for i, pos in enumerate(snap.robot_pos):
        col = ROBOT_COLORS[i % len(ROBOT_COLORS)]
        w = snap.waypoints[i]
        if w is not None and tuple(w) != tuple(pos):
            (x0, y0), (x1, y1) = robots_xy[i], xy(w)
            dr.line([(x0, y0), (x1, y1)], fill=col, width=3)
            v = np.array([x1 - x0, y1 - y0], float)
            nv = np.linalg.norm(v)
            if nv > 0:
                v /= nv
                q = np.array([-v[1], v[0]])
                tip = np.array([x1, y1])
                base = tip - 12 * v
                dr.polygon([tuple(tip), tuple(base + 6 * q), tuple(base - 6 * q)], fill=col, outline=(0, 0, 0))


def _draw_start_robots_north(dr, start_xy, sh: int, true_xy, disc_xy, moved, W: int, s: int) -> None:
    """Team start square S, numbered robot discs (a moved disc joined to its true cell by
    a thin line ending in a dot), and the north arrow in the margin."""
    rr = ROBOT_R
    if start_xy is not None:
        x, y = start_xy
        dr.rectangle([x - sh, y - sh, x + sh, y + sh], outline=(0, 0, 0), width=3, fill=(255, 255, 255))
        f = _font(13)
        l, t, r, bt = dr.textbbox((0, 0), "S", font=f)
        dr.text((x - (r - l) / 2 - l, y - (bt - t) / 2 - t), "S", font=f, fill=(0, 0, 0))
    for i, (x, y) in enumerate(disc_xy):
        if moved[i]:
            tx, ty = true_xy[i]
            dr.line([(tx, ty), (x, y)], fill=(0, 0, 0), width=1)
            dr.ellipse([tx - 2.5, ty - 2.5, tx + 2.5, ty + 2.5], fill=(0, 0, 0))
    for i, (x, y) in enumerate(disc_xy):
        col = ROBOT_COLORS[i % len(ROBOT_COLORS)]
        dr.ellipse([x - rr, y - rr, x + rr, y + rr], fill=col, outline=(0, 0, 0), width=2)
        f = _font(14)
        t = str(i + 1)
        l, tp, r, bt = dr.textbbox((0, 0), t, font=f)
        dr.text((x - (r - l) / 2 - l, y - (bt - tp) / 2 - tp), t, font=f, fill=(255, 255, 255),
                stroke_width=1, stroke_fill=(0, 0, 0))
    cx = W * s + MARGIN_PX / 2
    dr.line([(cx, 70), (cx, 22)], fill=(0, 0, 0), width=4)
    dr.polygon([(cx, 10), (cx - 10, 28), (cx + 10, 28)], fill=(0, 0, 0))
    f = _font(18)
    l, t, r, bt = dr.textbbox((0, 0), "N", font=f)
    dr.text((cx - (r - l) / 2 - l, 76 - t), "N", font=f, fill=(0, 0, 0))


def _opening_groups(olabel: np.ndarray) -> Dict[int, np.ndarray]:
    """{opening index: int array (n, 2) of its cells, raster order}."""
    rr, cc = np.nonzero(olabel > 0)
    lab = olabel[rr, cc]
    out: Dict[int, np.ndarray] = {}
    for j in np.unique(lab).tolist():
        m = lab == j
        out[int(j)] = np.stack([rr[m], cc[m]], 1)
    return out


def _draw_openings(dr, units: Units, colors, xy) -> Dict[int, np.ndarray]:
    """Openings as bold lines OPENING_PX wide (8-adjacent cells joined) in the colour of the
    piece behind them (units.op_piece), darker, with a 1 px darker outline; openings into no
    zone dotted dark gray. Returns the opening groups."""
    groups = _opening_groups(units.olabel)
    half = OPENING_PX / 2.0
    offs = ((0, 1), (1, 0), (1, 1), (1, -1))
    marks = []                                        # (cells, fill, edge)
    for j, cells in groups.items():
        k = units.op_piece.get(j, 0)
        if k in colors:
            col = colors[k]
            marks.append((cells, _dark(col, OPENING_DARK), _dark(col, OPENING_EDGE_DARK)))
        else:                                         # dotted: every other cell (at least one)
            keep = cells[(cells[:, 0] + cells[:, 1]) % 2 == 0]
            keep = keep if len(keep) else cells[:1]
            for r, c in keep.tolist():
                x, y = xy((r, c))
                dr.rectangle([x - half + 1, y - half + 1, x + half - 1, y + half - 1],
                             fill=NO_UNIT_OPENING_RGB)
    for grow, key in ((1.0, 2), (0.0, 1)):            # outline pass, then fill pass
        for cells, fill, edge in marks:
            col = edge if key == 2 else fill
            cs = set(map(tuple, cells.tolist()))
            for r, c in cells.tolist():
                x, y = xy((r, c))
                dr.rectangle([x - half - grow, y - half - grow, x + half + grow, y + half + grow], fill=col)
                for dr_, dc_ in offs:
                    if (r + dr_, c + dc_) in cs:
                        x2, y2 = xy((r + dr_, c + dc_))
                        dr.line([(x, y), (x2, y2)], fill=col, width=int(OPENING_PX + 2 * grow))
    return groups


CORRIDOR_DASH, CORRIDOR_GAP, CORRIDOR_PX = 7, 5, 2    # continued corridor: thin dashed centre line


def _corridor_run(line, mask: np.ndarray) -> List[Tuple[int, int]]:
    """The centre line's cells inside the piece: from its first cell in the piece to the last
    one before the line leaves it (the corridor piece is the part joined to the way in)."""
    run: List[Tuple[int, int]] = []
    for r, c in line:
        if mask[int(r), int(c)]:
            run.append((int(r), int(c)))
        elif run:
            break
    return run


def _draw_dashed(dr, pts, fill, width: int, dash: float, gap: float) -> None:
    """A dashed polyline through pixel points (dash on, gap off, carried across vertices)."""
    on, left = True, float(dash)
    for (x0, y0), (x1, y1) in zip(pts[:-1], pts[1:]):
        seg = math.hypot(x1 - x0, y1 - y0)
        t = 0.0
        while seg - t > 1e-9:
            step = min(left, seg - t)
            if on:
                a, b = t / seg, (t + step) / seg
                dr.line([(x0 + (x1 - x0) * a, y0 + (y1 - y0) * a), (x0 + (x1 - x0) * b, y0 + (y1 - y0) * b)],
                        fill=fill, width=width)
            t += step
            left -= step
            if left <= 1e-9:
                on = not on
                left = float(dash if on else gap)


def _draw_corridor_lines(dr, units: Units, colors, xy) -> None:
    """Every continued corridor: a thin dashed line along its centre, in its colour, dark."""
    for u, line in units.lines.items():
        k = units.ids.get(u)
        run = _corridor_run(line, units.masks[u]) if u in units.masks else []
        if k is None or len(run) < 2:
            continue
        fill = _dark(colors.get(k, (0, 0, 0)))
        _draw_dashed(dr, [xy(run[0]), xy(run[-1])], fill, CORRIDOR_PX, CORRIDOR_DASH, CORRIDOR_GAP)


def _box_cover(labels: np.ndarray, k: int, box, s: int) -> Tuple[float, float]:
    """(share of a pixel box on label k, share on other labels > 0), measured on the cells under it."""
    H, W = labels.shape
    x0, y0, x1, y1 = box
    c0, c1 = max(int(x0 // s), 0), min(int((x1 - 1e-6) // s) + 1, W)
    r_hi = min(H - 1 - int(y0 // s), H - 1)
    r_lo = max(H - 1 - int((y1 - 1e-6) // s), 0)
    sub = labels[r_lo:r_hi + 1, c0:c1]
    if sub.size == 0:
        return 0.0, 0.0
    own = float(np.count_nonzero(sub == k)) / sub.size
    other = float(np.count_nonzero((sub > 0) & (sub != k))) / sub.size
    return own, other


# Pieces image: labels of pieces that do not fit on their piece.
LABEL_SMALL_PX = TAG_PX + 3   # font of a piece label that does not fit on its piece in the big font
LABEL_RINGS = 3               # rings of spots tried around the dot before looking farther out
LABEL_OTHER_MAX = 0.2         # a spot beside a piece should lie at most this much on other pieces or white
ZONE_EDGE_RGB = (60, 60, 60)  # thin line between two zones (none between pieces of one zone)
_RING8 = ((0, -1), (0, 1), (1, 0), (-1, 0), (1, -1), (-1, -1), (1, 1), (-1, 1))


def _spots(placer: "_LabelPlacer", x: float, y: float, w: float, h: float, rings: int):
    """(bx, by, box, centred) of the label spots around (x, y): centred, then rings farther out."""
    dx, dy = w * 0.55, h * 0.9
    cands = [(0.0, 0.0)] + [(ox * dx * k, oy * dy * k) for k in range(1, rings + 1) for ox, oy in _RING8]
    for ox, oy in cands:
        bx = min(max(x + ox - w / 2, 1), placer.w - w - 1)
        by = min(max(y + oy - h / 2, 1), placer.h - h - 1)
        yield bx, by, (bx - 1, by - 1, bx + w + 1, by + h + 1), (ox, oy) == (0.0, 0.0)


def _leader(x: float, y: float, box) -> Tuple[float, float, float]:
    """(nearest point of the box to (x, y), its distance): where a leader line ends."""
    cx, cy = min(max(x, box[0] + 1), box[2] - 1), min(max(y, box[1] + 1), box[3] - 1)
    return cx, cy, math.hypot(cx - x, cy - y)


def place_piece_label(placer: "_LabelPlacer", x: float, y: float, sizes, cover):
    """
    Spot of one piece label (pieces image). sizes: [(font px, w, h)], the big font first, the
    small one last; cover(box) -> (share of the box on the piece, share on other pieces or on
    white no-zone unknown). In order: the first free spot (centred, then rings 1..LABEL_RINGS)
    where at least LABEL_OWN_MIN of the label lies on the piece, big font, then small font;
    else, small font, the free spot nearest to the dot among those lying at most LABEL_OTHER_MAX
    on anything else (else the nearest free spot), within LABEL_RINGS rings, then 8; else the
    centred spot. Returns (font px, bx, by, box, centred, on_piece); the box is added.
    """
    for fs, w, h in sizes:
        for bx, by, box, centred in _spots(placer, x, y, w, h, LABEL_RINGS):
            if placer._free(box) and cover(box)[0] >= LABEL_OWN_MIN:
                placer.add(box)
                return fs, bx, by, box, centred, True
    fs, w, h = sizes[-1]
    for rings in (LABEL_RINGS, 8):
        best = None
        for bx, by, box, centred in _spots(placer, x, y, w, h, rings):
            if not placer._free(box):
                continue
            rank = (cover(box)[1] > LABEL_OTHER_MAX, _leader(x, y, box)[2])
            if best is None or rank < best[0]:
                best = (rank, (bx, by, box, centred))
        if best is not None:
            bx, by, box, centred = best[1]
            placer.add(box)
            return fs, bx, by, box, centred, False
    bx, by, box, centred = next(_spots(placer, x, y, w, h, 0))
    placer.add(box)
    return fs, bx, by, box, centred, False


def _zone_edges(zones: np.ndarray, s: int) -> np.ndarray:
    """bool pixel mask (row 0 at the bottom, as the image) of a thin line wherever two unknown
    cells of different zones touch (4-adjacent); none between cells of one zone."""
    H, W = zones.shape
    z = zones[::-1]
    out = np.zeros((H * s, W * s), bool)
    t = max(1, s // 4)                       # line width in pixels, on each side of the edge
    d = (z[:, 1:] != z[:, :-1]) & (z[:, 1:] > 0) & (z[:, :-1] > 0)       # vertical edges
    for r, c in zip(*np.nonzero(d)):
        x = (c + 1) * s
        out[r * s:(r + 1) * s, max(x - t, 0):x + t] = True
    d = (z[1:, :] != z[:-1, :]) & (z[1:, :] > 0) & (z[:-1, :] > 0)       # horizontal edges
    for r, c in zip(*np.nonzero(d)):
        y = (r + 1) * s
        out[max(y - t, 0):y + t, c * s:(c + 1) * s] = True
    return out


def render_png(snap: Snapshot, units: Optional[Units] = None, layout: Optional[List[Dict]] = None) -> bytes:
    """
    The planner's image as PNG bytes: known map in plain grayscale (unknown white, free light
    gray, walls black); each piece tinted in its colour over its unknown cells, with its name (placed smallest piece first,
    place_piece_label); a thin dark line between zones; each way in drawn bold (OPENING_PX
    wide) in the colour of the piece behind it, darker, with its zone's id beside it (left out
    for a zone that is not cut when its label is next to the way in); openings into no zone
    dotted gray; a dashed centre line along each continued corridor; robots numbered with arrows to their committed waypoints (discs that would hide each
    other spread apart); the team start S; a north arrow. North up. layout (optional list, for
    the tests of the label placement): gets one dict per piece label (id, font px, on_piece,
    centred, leader_cells = length of its leader line in cells, 0 without one).
    """
    from PIL import Image, ImageDraw
    snap = query_view(snap)
    units = units or prepare_units(snap)
    occ = snap.occ
    H, W = occ.shape
    unk = occ == -1
    rgb = np.empty((H, W, 3), np.float32)
    rgb[:] = FREE_RGB
    rgb[unk] = UNKNOWN_RGB
    rgb[occ == 1] = OCC_RGB
    labels = units.labels
    colors = unit_colors(labels, units.open_labels)
    if colors:
        lut = np.zeros((int(labels.max()) + 1, 3), np.float32)
        has = np.zeros(len(lut), bool)
        for k, col in colors.items():
            lut[k] = col
            has[k] = True
        m = has[labels]
        rgb[m] = rgb[m] * (1 - UNIT_ALPHA) + lut[labels[m]] * UNIT_ALPHA

    s = _scale(occ)
    img = np.repeat(np.repeat(rgb[::-1], s, axis=0), s, axis=1)
    img[_zone_edges(units.zone_grid, s)] = ZONE_EDGE_RGB
    canvas = np.full((H * s, W * s + MARGIN_PX, 3), 255, np.uint8)
    canvas[:, :W * s] = np.clip(np.rint(img), 0, 255).astype(np.uint8)
    im = Image.fromarray(canvas)
    dr = ImageDraw.Draw(im)
    dr.rectangle([0, 0, W * s - 1, H * s - 1], outline=(110, 110, 110), width=1)

    def xy(p):                               # cell centre -> pixel, row 0 at the bottom
        return ((p[1] + 0.5) * s, (H - 1 - p[0] + 0.5) * s)

    groups = _draw_openings(dr, units, colors, xy)
    _draw_corridor_lines(dr, units, colors, xy)

    sh = 9
    rr = ROBOT_R
    placer = _LabelPlacer(W * s, H * s)
    true_xy, disc_xy, moved = robot_discs(snap)
    for x, y in disc_xy:
        placer.add((x - rr - 1, y - rr - 1, x + rr + 1, y + rr + 1))
    starts = _points(snap.team_start)
    start_xy = xy(np.mean(np.asarray(starts, float), axis=0)) if starts else None
    if start_xy is not None:
        placer.add((start_xy[0] - sh, start_xy[1] - sh, start_xy[0] + sh, start_xy[1] + sh))
    _draw_waypoint_arrows(dr, snap, true_xy, xy)

    # Piece labels, smallest piece first, so small pieces get the spots next to them: on the
    # piece when most of the label lies there, else (in the small font) the nearest free spot
    # that covers little of other pieces or white no-zone space, with a line to a dot on the
    # piece's anchor cell.
    big = max(16, TARGET_PX // 34)
    label_box: Dict[str, Tuple[float, float, float, float]] = {}
    cov = np.where(unk & (labels <= 0), -1, labels)
    for u, k in sorted(zip(units.open_ids, units.open_labels), key=lambda t: (int(units.masks[t[0]].sum()), t[1])):
        fill = _dark(colors.get(k, (0, 0, 0)))
        x, y = xy(units.anchors[u])
        sizes = []
        for fs in (big, LABEL_SMALL_PX):
            l, t, r, bt = dr.textbbox((0, 0), u, font=_font(fs), stroke_width=3)
            sizes.append((fs, r - l, bt - t))

        def cover(box, k=k):
            own, other = _box_cover(cov, k, box, s)
            return own, other + _box_cover(cov, -1, box, s)[0]

        fs, bx, by, box, centred, on_piece = place_piece_label(placer, x, y, sizes, cover)
        fl = _font(fs)
        l, t, _, _ = dr.textbbox((0, 0), u, font=fl, stroke_width=3)
        lead = 0.0
        if not (centred and on_piece):
            cx, cy, lead = _leader(x, y, box)
            dr.line([(x, y), (cx, cy)], fill=fill, width=2)
            dr.ellipse([x - 3.5, y - 3.5, x + 3.5, y + 3.5], fill=fill, outline=(255, 255, 255))
        dr.text((bx - l, by - t), u, font=fl, fill=fill, stroke_width=3, stroke_fill=(255, 255, 255))
        label_box[u] = box
        if layout is not None:
            layout.append({"id": u, "font": fs, "on_piece": on_piece, "centred": centred,
                           "leader_cells": round(lead / s, 2)})
    # Id tags beside the openings (one per opening; skipped when the same id's tag is close).
    ft = _font(TAG_PX)
    placed: Dict[int, List[Tuple[float, float]]] = {}
    for j, cells in sorted(groups.items()):
        k = units.op_piece.get(j, 0)                             # coloured as the piece behind it
        if k not in colors:
            continue
        zone = units.opening_unit.get(j, 0)                      # the tag names the zone
        mid = cells.mean(0)
        r0, c0 = cells[int(np.argmin(((cells - mid) ** 2).sum(1)))].tolist()
        x, y = xy((r0, c0))
        if any(math.hypot(x - px, y - py) < TAG_DEDUP_PX for px, py in placed.get(zone, [])):
            continue
        # A zone that is not cut: its piece label is the same id; no tag when it is close by.
        own = label_box.get(f"{ID_PREFIX}{zone}")
        if own is not None and _leader(x, y, own)[2] < TAG_DEDUP_PX:
            continue
        placed.setdefault(zone, []).append((x, y))
        text = f"{ID_PREFIX}{zone}"
        fill = _dark(colors[k])
        l, t, r, bt = dr.textbbox((0, 0), text, font=ft, stroke_width=2)
        w, h = r - l, bt - t
        bx, by, mv = placer.place(x, y, w, h, OPENING_PX / 2 + 1)
        if mv:
            cx, cy = min(max(x, bx), bx + w), min(max(y, by), by + h)
            dr.line([(x, y), (cx, cy)], fill=fill, width=1)
        dr.text((bx - l, by - t), text, font=ft, fill=fill, stroke_width=2, stroke_fill=(255, 255, 255))

    _draw_start_robots_north(dr, start_xy, sh, true_xy, disc_xy, moved, W, s)
    buf = io.BytesIO()
    im.save(buf, format="PNG", compress_level=3)
    return buf.getvalue()


# ---------------------------------------------------------------------------
# Prompt and schema
# ---------------------------------------------------------------------------

def reply_schema(snap: Snapshot, units: Optional[Units] = None) -> Dict:
    """JSON schema of the reply; IDs are an enum of the pieces, so no repair is needed."""
    snap = query_view(snap)
    units = units or prepare_units(snap)
    ids = list(units.open_ids)
    n = len(snap.robot_pos)
    lst = {"type": "array", "minItems": 1, "maxItems": 1, "items": {"type": "string", "enum": ids}}
    keys = [str(i + 1) for i in range(n)]
    props: Dict[str, Dict] = {}
    required = []
    if uses_belief(snap.task):
        # Hypotheses first: with thinking off, naming where the target is before
        # assigning robots is the only reasoning the model does.
        # Any piece may be a hypothesis. The list may be empty: no hypotheses = uniform belief, so
        # the schema never forces a guess the map cannot support.
        hmax = max(1, len(ids))
        key = PIECE_WORD
        props["hypotheses"] = {
            "type": "array", "minItems": 0, "maxItems": hmax,
            "items": {"type": "object", "additionalProperties": False, "required": [key, "p"],
                      "properties": {key: {"type": "string", "enum": ids},
                                     "p": {"type": "number", "minimum": 0, "maximum": 1}}}}
        required.append("hypotheses")
    props["assignments"] = {"type": "object", "additionalProperties": False, "required": keys,
                            "properties": {k: lst for k in keys}}
    required.append("assignments")
    return {"type": "object", "additionalProperties": False, "required": required, "properties": props}


def _fmt_m(m: float) -> str:
    return f"{m:.2f}" if m >= 0.005 or m == 0 else "<0.01"


# Which IDs the reply may use (pieces): the table's piece lines; a cut zone's own id is not one.
PIECE_IDS = ("piece IDs from the table (the indented lines, and the zones that are not cut; a cut "
             "zone's own id is not a piece)")

# The second line of the search hypotheses rule.
EMPTY_HYPS = ("  The list may be empty if the map does not yet tell where the target is (e.g. only one "
              "room explored so far); the robots then treat all unknown space as equally likely.")

def _search_rules_pieces(dist: str, size_note: str = " (large size)") -> List[str]:
    """The hypotheses and assignments lines of the search RULES with a hint. A search without a
    hint never gets here: it is asked as an explore query (query_view). size_note: what follows
    "much unknown space" (no_table: nothing, the table's size column is not shown)."""
    P, Z = PIECE_WORD, UNIT_WORD
    hyp = (f"- hypotheses: {P}s whose unknown space most likely holds the target, given the hint and what "
           "is already explored, each with probability p. Give each p as honestly as you can: a hint can be "
           f"misread, and your single top guess is right less than half the time. List your 2 or 3 most "
           f"likely {P}s, unless the hint and the map together clearly point to one {P}.")
    rest = f"the nearest {P}s with much unknown space{size_note}"
    a = (f"- assignments: start with your most likely {P}: send its nearest robots there ({dist}), but never "
         f"more than half of the robots (rounded up) to one {P}. Send the other robots to your next most "
         f"likely {P}s, and any robots still left to {rest}. Put the robots in at least two different {Z}s "
         f"whenever you can: a wrong {Z} costs the most when every robot is in it.")
    return [hyp, EMPTY_HYPS, a]


# The ablation variants' rewordings (search with a hint only).
NO_TABLE_DIST = "the nearest by walking, judged from the picture"
NO_TABLE_PIECE_IDS = ("piece IDs shown on the picture (a zone cut into pieces is named by its pieces, not by "
                      "its own id)")


def build_prompt_text(snap: Snapshot, params: Optional[Params] = None, units: Optional[Units] = None,
                      variant: str = "full") -> str:
    """The prompt text. A zone is the unexplored space behind a way in; big zones are cut into
    pieces named anew at every plan ("a" at the way in, continued corridors next, then the
    others); the robots are given pieces, and the hypotheses name pieces. variant (search with a
    hint only): "no_table" leaves out the table (the rules judge distance and size from the
    picture), "no_history" the robots' current pieces and the m column, "no_picture" the IMAGE
    legend (its definitions stay, as TERMS) and the picture references."""
    params = params or Params()
    snap = query_view(snap)
    units = units or prepare_units(snap)
    if variant not in VARIANTS[:4]:
        raise ValueError(f"build_prompt_text: variant must be one of {VARIANTS[:4]}, got {variant!r}")
    if variant != "full" and snap.task != "search":
        raise ValueError(f"the {variant} variant asks a search with a hint only")
    table, history, picture = variant != "no_table", variant != "no_history", variant != "no_picture"
    n = len(snap.robot_pos)
    T_H = int(params.T_H)
    search = snap.task == "search"
    belief = uses_belief(snap.task)
    m_col = belief and history                 # the m column carries the previous plan's guesses
    steer = belief and float(params.beta) > 0
    unk = snap.occ == -1
    H, W = snap.occ.shape
    Z, P = UNIT_WORD, PIECE_WORD
    X = ID_PREFIX + "<k>"                      # placeholder ids: never a real id of the table
    L = []
    if search:
        L.append(f"You coordinate a team of {n} robots searching a partly explored building for a target. "
                 "Goal: find the target as quickly as possible. It has not been seen yet, so it lies "
                 "somewhere in the unknown " + ("(white or tinted) space." if picture else "space."))
        L.append(f'Hint about the target: "{snap.hint}"')       # (a search without one is an explore query)
    else:
        L.append(f"You coordinate a team of {n} robots exploring an unknown building. "
                 "Goal: see 95% of its free space as quickly as possible.")

    uncut = [z for z in units.zones if len(z["pieces"]) == 1 and z["pieces"][0] == z["id"]]
    if picture:
        L += ["", f"IMAGE: the team's merged map, north up. The map is {W} cells east-west by {H} cells "
                  f"north-south; 1 cell = {_scale(snap.occ)} px in the image.",
              "- light gray = explored free space; black = walls and obstacles; white or tinted = unknown.",
              "- ways in = places where explored free space meets unknown space, drawn as bold lines.",
              f"- the unknown space is split into {Z}s: a {Z} is the unexplored space behind one way in (the "
              f"unknown space closer to it than to any other way in). It is a guess of what lies behind that way "
              f"in; far parts of a large {Z} may really be reached another way. Small {Z}s are merged into a "
              f"neighbour, so a few {Z}s have 2 or 3 ways in.",
              f"- big {Z}s are cut into {P}s of about what a robot sees from one spot. {P.capitalize()}s are named "
              f"anew at every plan: the {Z}'s id plus a letter ({X}a, {X}b, ...); \"a\" is the {P} at the way in, "
              f"continued corridors come next, then the others. A {Z} that is not cut is one {P} with the {Z}'s "
              f"own id ({X}).",
              f"- a continued corridor is a corridor of the explored map carried straight on into the unknown with "
              f"its own width (thin scraps beside it are added to it); the corridor may go on there. It is drawn "
              f"with a thin dashed centre line.",
              f"- each {P} is tinted in a color and labelled with its id; a label that does not fit inside its {P} "
              f"sits beside it (in a smaller font), with a line to a dot in the {P}. A thin dark line separates "
              f"neighbouring {Z}s; {P}s of one {Z} have no line between them. Each way in is drawn in the color of "
              f"the {P} behind it, darker, with its {Z}'s id beside it. {P.capitalize()}s far apart may share a "
              "color: the id is what counts."]
        live = set(units.open_labels)
        if any(units.op_piece.get(j, 0) not in live for j in np.unique(units.olabel[units.olabel > 0]).tolist()):
            L.append(f"- dotted dark gray marks = ways in to a patch too small to count (no {Z}).")
        if (unk & (units.labels <= 0)).any():
            L.append(f"- unknown space left white belongs to no {Z} (a small patch, or space no way in leads into).")
        moved = any(robot_discs(snap)[2])
        L.append(f"- numbered circles = robots 1..{n}; an arrow points from a robot to the spot it was driving to "
                 "(chosen on its own map before the maps were merged, so this map may show it already explored)."
                 + (" Circles that would cover each other are drawn apart, each joined to its robot's true cell "
                    "by a thin line ending in a dot." if moved else ""))
        L.append("- black square S = the team's start.")
    else:
        # no_picture: the same definitions the IMAGE section gives, without what is drawn, so the table
        # can still be read.
        L += ["", "TERMS:",
              "- ways in = places where explored free space meets unknown space.",
              f"- the unknown space is split into {Z}s: a {Z} is the unexplored space behind one way in (the "
              f"unknown space closer to it than to any other way in). It is a guess of what lies behind that way "
              f"in; far parts of a large {Z} may really be reached another way. Small {Z}s are merged into a "
              f"neighbour, so a few {Z}s have 2 or 3 ways in.",
              f"- big {Z}s are cut into {P}s of about what a robot sees from one spot. {P.capitalize()}s are named "
              f"anew at every plan: the {Z}'s id plus a letter ({X}a, {X}b, ...); \"a\" is the {P} at the way in, "
              f"continued corridors come next, then the others. A {Z} that is not cut is one {P} with the {Z}'s "
              f"own id ({X}).",
              f"- a continued corridor is a corridor of the explored map carried straight on into the unknown with "
              f"its own width (thin scraps beside it are added to it); the corridor may go on there."]

    L += ["", "HOW YOUR PLAN IS USED:",
          "Robots always drive to frontiers (boundaries between explored and unknown space), one cell per step."]
    toward = (f"it prefers, among the frontiers its own map shows into that {P}'s {Z}, the ones nearest to the "
              f"{P} (by a walk through the {Z}'s unknown space; a frontier into the {P} itself is nearest)")
    L.append(f"You give each robot one {P}: {toward}. When its map shows no frontier into that {Z} any more, "
             "it picks frontiers on its own.")
    if steer:
        L.append("Robots also prefer frontiers that look into unknown space where your hypotheses place the target.")
    L.append(f"A new plan is made every {T_H} steps; yours is used until then.")

    if table:
        L += [""]
        arrow = "its arrow's tip; its position if it has no arrow" if picture else "its position if it has none"
        zhead = [f"way in = total width in cells of the {Z}'s way(s) in; a wide way in usually means a corridor "
                 "or open area continues",
                 f"d1..d{n} = path length in cells from the spot robot i was driving to ({arrow}) to the {Z}'s "
                 "nearest way in, through explored space (unreachable = no known path)"]
        boxed = "black" if picture else "walls"
        phead = [f"size = unknown cells in the {P} (how much is left to explore there). It counts every unknown cell "
                 f"up to the map edge, so a {P} that runs past the building's outer wall, or a thin strip boxed in "
                 f"by {boxed} (inside a shelf or a thick wall), may be mostly solid",
                 f"beyond = how far the {P} lies past the way in: the length in cells of the shortest walk from the "
                 f"way in through the {Z}'s unknown space to the {P}, 0 = it starts at the way in"
                 " (so d_i + beyond is about robot i's walk to the piece)",
                 f"corridor = yes for a continued corridor"]
        if m_col:
            phead.append(f"m = share of the target belief in the {P}")
        zcols = [Z, "way in"] + [f"d{i + 1}" for i in range(n)]
        pcols = [P, "size", "beyond", "corridor"] + (["m"] if m_col else [])
        L.append(f"{Z.capitalize()}s and their {P}s. One line per {Z}: " + "; ".join(zhead) + f". Under a cut {Z}, "
                 f"one indented line per {P}: " + "; ".join(phead) + f". A {Z} that is not cut is a single {P} with "
                 f"the {Z}'s id, written on one line with both parts. {P.capitalize()}s do not overlap.")
        L.append(" | ".join(zcols))
        L.append("  " + " | ".join(pcols))
        if uncut:
            L.append(" | ".join(zcols + pcols[1:]) + f"   (a {Z} that is not cut)")
        rows = {r["id"]: r for r in units.rows}

        def piece_cells(r):
            pc = [str(r["size"]), "?" if r["beyond"] is None else str(r["beyond"]), "yes" if r["corridor"] else "no"]
            return pc + ([_fmt_m(r["m"])] if m_col else [])

        for z in units.zones:
            cells = [z["id"], str(z["width"])] + ["unreachable" if d is None else str(d) for d in z["dgeo"]]
            if len(z["pieces"]) == 1 and z["pieces"][0] == z["id"]:
                L.append(" | ".join(cells + piece_cells(rows[z["id"]])))
                continue
            L.append(" | ".join(cells))
            for u in z["pieces"]:
                L.append("  " + " | ".join([u] + piece_cells(rows[u])))

    if history:
        L += ["", f"Current {P}s (from the previous plan, under their ids in this {'table' if table else 'picture'}):"]
        open_set = set(units.open_ids)
        for i in range(n):
            shown = []
            for u in snap.lists[i]:
                if u in open_set:
                    shown.append(u)
                elif u == GONE:
                    shown.append(f"gone (its {P} was explored, or its {Z} was cut anew)")
                else:
                    shown.append(f"{u} (gone)")
            L.append(f"Robot {i + 1}: " + (", ".join(shown) if shown else "none"))

    L += ["", "RULES:"]
    dist = "small d, then small beyond" if table else NO_TABLE_DIST
    if belief:
        L += _search_rules_pieces(dist, " (large size)" if table else "")    # with a hint (query_view)
    else:
        L.append(f"- assignments: spread the robots over {P}s with much unknown space (large size), weighing "
                 f"{dist}; give a robot a near {P} with much unknown space, and let two robots share a "
                 f"{P} only if it is very large.")
    L.append(f"- use only {PIECE_IDS if table else NO_TABLE_PIECE_IDS}; every robot 1..{n} gets exactly 1 ID.")

    L += ["", "Reply with only this JSON object:"]
    a = ", ".join(f'"{i + 1}": [<id>]' for i in range(n))
    if belief:
        L.append('{"hypotheses": [{"' + PIECE_WORD + '": <id>, "p": <number 0..1>}, ...], "assignments": {'
                 + a + "}}")
    else:
        L.append('{"assignments": {' + a + "}}")
    return "\n".join(L)


def build_messages(snap: Snapshot, params: Optional[Params] = None, png: Optional[bytes] = None,
                   units: Optional[Units] = None, variant: str = "full") -> List[Dict]:
    """One user message: the image, then the text (no_picture: the text only; no picture is drawn)."""
    params = params or Params()
    snap = query_view(snap)
    units = units or prepare_units(snap)
    text = {"type": "text", "text": build_prompt_text(snap, params, units, variant)}
    if variant == "no_picture":
        return [{"role": "user", "content": [text]}]
    if png is None:
        png = render_png(snap, units)
    b64 = base64.b64encode(png).decode("ascii")
    return [{"role": "user", "content": [
        {"type": "image_url", "image_url": {"url": f"data:image/png;base64,{b64}"}},
        text,
    ]}]


# ---------------------------------------------------------------------------
# Reply parsing and the planner
# ---------------------------------------------------------------------------

def parse_reply(text: str, open_ids: Sequence[str], n_robots: int,
                task: str) -> Tuple[Dict[int, List[str]], List[Tuple[str, float]]]:
    """(one-piece lists by 0-based robot index: the robot's first valid piece, or [] if it has none;
    normalised hypotheses). Raises if the reply is not JSON."""
    start, end = text.find("{"), text.rfind("}") + 1
    if start == -1 or end <= start:
        raise ValueError("no JSON object in the reply")
    data = json.loads(text[start:end])
    ok = set(open_ids)
    assign = data.get("assignments") or {}
    lists: Dict[int, List[str]] = {}
    for i in range(n_robots):
        lists[i] = next(([u] for u in (assign.get(str(i + 1)) or []) if u in ok), [])
    hyps: List[Tuple[str, float]] = []
    if uses_belief(task):
        acc: Dict[str, float] = {}
        for h in data.get("hypotheses") or []:
            u = h.get(PIECE_WORD) if isinstance(h, dict) else None
            if u in ok:
                p = h.get("p", 0.0)
                p = float(p) if isinstance(p, (int, float)) and np.isfinite(p) else 0.0
                acc[u] = acc.get(u, 0.0) + max(p, 0.0)
        tot = sum(acc.values())
        hyps = [(u, (p / tot) if tot > 0 else 1.0 / len(acc)) for u, p in acc.items()]
    return lists, hyps


class VLMPlanner:
    """plan(snap) -> (Plan or None, query record). One synchronous VLM call per plan (thinking off).
    variant: "full" (the method) or an ablation variant (VARIANTS; search with a hint only)."""

    def __init__(self, task: str, hint: str, n_robots: int, params: Optional[Params] = None,
                 model: Optional[str] = None, base_url: Optional[str] = None, variant: str = "full"):
        if task not in TASKS:
            raise ValueError(f"task must be one of {TASKS}, got {task!r}")
        if variant not in VARIANTS:
            raise ValueError(f"variant must be one of {VARIANTS}, got {variant!r}")
        if variant != "full" and (task != "search" or not hint):
            raise ValueError(f"the {variant} variant runs on search with a hint only")
        self.variant = variant
        self.task, self.hint, self.n = task, hint or "", int(n_robots)
        self.params = params or Params()
        self.model, self.base_url = model, base_url
        self.last_messages: Optional[List[Dict]] = None    # last query, for offline inspection
        self.last_schema: Optional[Dict] = None

    def plan(self, snap: Snapshot) -> Tuple[Optional[Plan], Dict]:
        if self.variant == "no_decomp":
            from comres_points import plan_points
            return plan_points(self, snap)
        t0 = time.perf_counter()
        # query_view: a search without a hint is asked as an explore query
        snap = query_view(dataclasses.replace(snap, task=self.task, hint=self.hint))
        units = prepare_units(snap)
        rec: Dict[str, Any] = dict(step=int(snap.step), n_open=len(units.open_ids), n_robots=self.n,
                                   prep_seconds=round(units.prep_seconds, 4))
        if not units.open_ids:
            rec.update(skipped="no_open_unit", valid=False, cpu_seconds=round(time.perf_counter() - t0, 4),
                       call_seconds=0.0, lock_wait_seconds=0.0)
            return None, rec
        t1 = time.perf_counter()
        png = None if self.variant == "no_picture" else render_png(snap, units)
        t2 = time.perf_counter()
        messages = build_messages(snap, self.params, png, units, self.variant)
        schema = reply_schema(snap, units)
        t3 = time.perf_counter()
        self.last_messages, self.last_schema = messages, schema
        rec.update(render_seconds=round(t2 - t1, 4), build_seconds=round(t3 - t2 + units.prep_seconds, 4),
                   table=units.rows)
        text, usage, err = "", {}, None
        try:
            text, usage = vlm_client.call(messages, schema, model=self.model, base_url=self.base_url)
        except Exception as e:                      # server down / HTTP error: no plan
            err = f"{type(e).__name__}: {e}"
        t4 = time.perf_counter()
        if err is not None:
            rec["failed_call_seconds"] = round(t4 - t3, 4)   # incl. any lock wait; no plan to charge
        plan = None
        lists, hyps = {}, []
        if err is None:
            try:
                lists, hyps = parse_reply(text, units.open_ids, self.n, snap.task)
                plan = Plan(lists=lists, hyps=hyps, omegas=dict(units.masks), labels=units.labels,
                            query_step=int(snap.step), ids=dict(units.ids), piece_unit=dict(units.piece_unit))
            except Exception as e:
                err = f"{type(e).__name__}: {e}"
        t5 = time.perf_counter()
        # CPU part charged by the strategy: everything but the VLM call itself.
        cpu = (t3 - t0) + (t5 - t4)
        rec.update(usage)
        rec.update(parse_seconds=round(t5 - t4, 4), cpu_seconds=round(max(cpu, 0.0), 4),
                   raw=text, valid=plan is not None, error=err,
                   lists={str(i + 1): l for i, l in lists.items()}, hyps=[[u, round(p, 4)] for u, p in hyps])
        rec.setdefault("call_seconds", 0.0)
        rec.setdefault("lock_wait_seconds", 0.0)
        return plan, rec
