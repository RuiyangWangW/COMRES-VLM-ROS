"""
The planning grid, and the conversion of a ROS occupancy grid onto it.

COMRES plans on a grid of square cells in one shared world frame (REP-103: x east, y north).
Cell (row, col) covers x in [x0 + col * cell, x0 + (col + 1) * cell) and y in [y0 + row * cell,
y0 + (row + 1) * cell), so a larger row is further north, as in COMRES-VLM (origin="lower").
Values: -1 unknown, 0 free, 1 wall.

The COMRES parameters are in cells and were tuned with a 20-cell sensor range (d_det = 20); a
cell of (the robots' useful sensing range) / 20 keeps them meaning the same thing.

A robot's SLAM map (nav_msgs/OccupancyGrid: row-major, data[j * width + i] is the cell at
(i, j) of the map, j along the map's y axis, origin = the pose of cell (0, 0)'s corner in the
map's frame; -1 unknown, 0..100 occupied probability) is sampled at sub x sub points per planning
cell, after growing its occupied pixels (>= occ_threshold) by one pixel so that thin walls stay
closed at any angle between the map and the grid: a cell is a wall if any sample is occupied;
else free if at least free_fraction of its samples are free (<= free_threshold); else unknown
(values between the two thresholds are unknown, as in Nav2's map convention).
"""

from dataclasses import dataclass
import math
from typing import Optional, Tuple

import numpy as np
from scipy import ndimage

UNKNOWN, FREE, WALL = -1, 0, 1

Pose2D = Tuple[float, float, float]          # x, y, yaw


@dataclass(frozen=True)
class GridSpec:
    x0: float              # world x of the grid's west edge (m)
    y0: float              # world y of the grid's south edge (m)
    cell: float            # cell side (m)
    rows: int
    cols: int

    @classmethod
    def from_extent(cls, x0: float, y0: float, width_m: float, height_m: float, cell: float) -> "GridSpec":
        return cls(float(x0), float(y0), float(cell), int(math.ceil(height_m / cell)), int(math.ceil(width_m / cell)))

    @property
    def shape(self) -> Tuple[int, int]:
        return (self.rows, self.cols)

    def cell_of(self, x: float, y: float) -> Optional[Tuple[int, int]]:
        """The cell holding world point (x, y), or None outside the grid."""
        r, c = int(math.floor((y - self.y0) / self.cell)), int(math.floor((x - self.x0) / self.cell))
        return (r, c) if 0 <= r < self.rows and 0 <= c < self.cols else None

    def center(self, cell: Tuple[int, int]) -> Tuple[float, float]:
        """World (x, y) of a cell's centre."""
        r, c = cell
        return self.x0 + (c + 0.5) * self.cell, self.y0 + (r + 0.5) * self.cell


def compose(a: Pose2D, b: Pose2D) -> Pose2D:
    """a o b: the pose b (given in a's frame) in the frame a is given in."""
    ca, sa = math.cos(a[2]), math.sin(a[2])
    return (a[0] + ca * b[0] - sa * b[1], a[1] + sa * b[0] + ca * b[1], a[2] + b[2])


def invert(a: Pose2D) -> Pose2D:
    c, s = math.cos(a[2]), math.sin(a[2])
    return (-(c * a[0] + s * a[1]), -(-s * a[0] + c * a[1]), -a[2])


def occupancy_to_grid(data: np.ndarray, resolution: float, origin: Pose2D, world_from_map: Pose2D,
                      spec: GridSpec, occ_threshold: int = 65, free_fraction: float = 0.5,
                      sub: Optional[int] = None, free_threshold: int = 25) -> np.ndarray:
    """
    A SLAM occupancy grid on the planning grid. data: (height, width) int array, data[j, i] the
    map cell (i, j); origin: the pose of the map's cell (0, 0) corner in the map's frame;
    world_from_map: the map frame's pose in the world frame. sub: samples per cell side
    (default: one per map cell, at least 1).
    """
    data = np.asarray(data).astype(np.int16)
    h, w = data.shape
    occ = ndimage.binary_dilation(data >= occ_threshold, structure=np.ones((3, 3), bool))
    data = np.where(occ, 100, data)
    s = int(sub) if sub else max(1, int(math.ceil(spec.cell / float(resolution))))
    off = (np.arange(s) + 0.5) / s                                    # sample offsets in a cell
    cols = (np.arange(spec.cols)[:, None] + off[None, :]).ravel()     # (cols*s,)
    rows = (np.arange(spec.rows)[:, None] + off[None, :]).ravel()     # (rows*s,)
    X = spec.x0 + cols[None, :] * spec.cell                            # world, (1, cols*s)
    Y = spec.y0 + rows[:, None] * spec.cell                            # world, (rows*s, 1)
    # world -> map pixel: inverse of (world_from_map o origin)
    T = invert(compose(world_from_map, origin))
    c, sn = math.cos(T[2]), math.sin(T[2])
    px = T[0] + c * X - sn * Y
    py = T[1] + sn * X + c * Y
    i = np.floor(px / resolution).astype(np.int64)
    j = np.floor(py / resolution).astype(np.int64)
    inside = (i >= 0) & (i < w) & (j >= 0) & (j < h)
    v = np.full(inside.shape, -1, dtype=np.int16)
    v[inside] = data[j[inside], i[inside]]
    v = v.reshape(spec.rows, s, spec.cols, s)
    wall = (v >= occ_threshold).any(axis=(1, 3))
    free = ((v >= 0) & (v <= free_threshold)).mean(axis=(1, 3)) >= free_fraction
    out = np.full(spec.shape, UNKNOWN, dtype=np.int8)
    out[free] = FREE
    out[wall] = WALL
    return out


def free_cell_near(grid: np.ndarray, cell: Tuple[int, int], radius: int = 3) -> Tuple[int, int]:
    """The cell itself if it is not a wall, else the nearest FREE cell within `radius` (or, if
    none, the nearest unknown one): a robot never stands in a wall; discretisation can put its cell
    there, and its seen side is the likelier one."""
    r, c = cell
    if grid[r, c] != WALL:
        return cell
    for want in (FREE, UNKNOWN):
        best, bd = None, None
        for dr in range(-radius, radius + 1):
            for dc in range(-radius, radius + 1):
                rr, cc = r + dr, c + dc
                if 0 <= rr < grid.shape[0] and 0 <= cc < grid.shape[1] and grid[rr, cc] == want:
                    d = dr * dr + dc * dc
                    if bd is None or d < bd:
                        best, bd = (rr, cc), d
        if best is not None:
            return best
    return cell
