"""The planning grid and the SLAM-map conversion."""
import math

import numpy as np
import pytest

from comres_core.grid import FREE, UNKNOWN, WALL, GridSpec, compose, free_cell_near, invert, occupancy_to_grid


def test_cells_and_centres():
    g = GridSpec.from_extent(-2.0, -1.0, 4.0, 2.0, 0.25)
    assert g.shape == (8, 16)
    assert g.cell_of(-2.0, -1.0) == (0, 0) and g.cell_of(1.99, 0.99) == (7, 15) and g.cell_of(2.0, 0.0) is None
    assert g.cell_of(*g.center((3, 5))) == (3, 5)
    assert g.cell_of(0.1, 0.9)[0] > g.cell_of(0.1, -0.9)[0]          # north = a larger row


def test_pose_algebra():
    a, b = (1.0, 2.0, 0.7), (-0.3, 0.5, -1.2)
    c = compose(compose(a, b), invert(compose(a, b)))
    assert max(abs(v) for v in c) < 1e-9


def _slam(res=0.05, size=(80, 80)):
    """A 4 m x 4 m SLAM map: free floor, a wall along map-x at map y = 2.0 m, unknown north of it."""
    m = np.zeros(size, np.int16)
    j_wall = int(2.0 / res)
    m[j_wall, :] = 100
    m[j_wall + 1:, :] = -1
    return m


def test_identity_frames():
    spec = GridSpec.from_extent(0.0, 0.0, 4.0, 4.0, 0.25)
    g = occupancy_to_grid(_slam(), 0.05, (0.0, 0.0, 0.0), (0.0, 0.0, 0.0), spec)
    assert (g[:7] == FREE).all()                       # rows 0-6: y 0 .. 1.75 m, free
    assert (g[7:9] == WALL).any(axis=0).all()          # the wall (y 2.0 m, grown one pixel) stays closed
    assert not (g[7:9] == FREE).all(axis=0).any()
    assert (g[9:] == UNKNOWN).all()


def _room(res=0.05, n=120, wall_at=20):
    """A closed square room drawn with 1-pixel walls, free inside, unknown outside."""
    m = np.full((n, n), -1, np.int16)
    a, b = wall_at, n - wall_at
    m[a:b, a:b] = 0
    m[a, a:b] = m[b - 1, a:b] = m[a:b, a] = m[a:b, b - 1] = 100
    return m


def _leaks(g):
    """Whether the room's free cells reach the outside through free cells (8-connected)."""
    from scipy import ndimage
    lab, _ = ndimage.label(g == FREE, structure=np.ones((3, 3), bool))
    border = set(lab[0, :]) | set(lab[-1, :]) | set(lab[:, 0]) | set(lab[:, -1])
    inner = lab[g.shape[0] // 2, g.shape[1] // 2]
    return inner != 0 and inner in border


@pytest.mark.parametrize("yaw", [0.0, 0.3, math.pi / 4, 1.1, 2.0, -0.7])
def test_thin_walls_stay_closed_at_any_angle(yaw):
    """1-pixel walls, the map frame turned against the grid: no gap the geodesic could pass."""
    spec = GridSpec.from_extent(-6.0, -6.0, 12.0, 12.0, 0.25)
    m = _room()
    g = occupancy_to_grid(m, 0.05, (-3.0, -3.0, 0.0), (0.13, -0.07, yaw), spec)
    assert (g == FREE).any() and not _leaks(g)


def test_uncertain_cells_are_unknown():
    spec = GridSpec.from_extent(0.0, 0.0, 1.0, 1.0, 0.25)
    m = np.full((20, 20), 40, np.int16)                 # between free (<= 25) and occupied (>= 65)
    g = occupancy_to_grid(m, 0.05, (0.0, 0.0, 0.0), (0.0, 0.0, 0.0), spec)
    assert (g == UNKNOWN).all()


def test_rotated_map_frame():
    """The robot's map frame is turned 90 deg left in the world: its +y wall becomes a wall at world
    x = -2.0 m, the unknown side to the west."""
    spec = GridSpec.from_extent(-4.0, 0.0, 4.0, 4.0, 0.25)
    g = occupancy_to_grid(_slam(), 0.05, (0.0, 0.0, 0.0), (0.0, 0.0, math.pi / 2), spec)
    col_wall = spec.cell_of(-2.1, 1.0)[1]
    assert (g[:, col_wall] == WALL).all()                 # closed along its whole length
    assert (g[:, col_wall + 2:] == FREE).all() and (g[:, :col_wall] == UNKNOWN).all()
    assert (g[:, col_wall + 1] != UNKNOWN).all()           # the grown pixel may reach the next column


def test_free_cell_near():
    occ = np.zeros((5, 5), np.int8)
    occ[2, 2] = WALL
    assert free_cell_near(occ, (2, 2)) != (2, 2) and occ[free_cell_near(occ, (2, 2))] != WALL
    assert free_cell_near(occ, (1, 1)) == (1, 1)
