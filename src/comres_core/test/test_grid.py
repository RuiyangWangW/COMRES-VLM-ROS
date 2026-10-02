"""The planning grid and the SLAM-map conversion."""
import math

import numpy as np

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
    assert (g[:8] == FREE).all()                       # rows 0-7: y 0 .. 2 m, free
    assert (g[8] == WALL).all()                        # the wall row (y 2.0 .. 2.25) stays closed
    assert (g[9:] == UNKNOWN).all()


def test_rotated_map_frame():
    """The robot's map frame is turned 90 deg left in the world: its +y wall becomes a wall at world
    x = -2.0 m, the unknown side to the west."""
    spec = GridSpec.from_extent(-4.0, 0.0, 4.0, 4.0, 0.25)
    g = occupancy_to_grid(_slam(), 0.05, (0.0, 0.0, 0.0), (0.0, 0.0, math.pi / 2), spec)
    col_wall = spec.cell_of(-2.1, 1.0)[1]
    assert (g[:, col_wall] == WALL).all()
    assert (g[:, col_wall + 1:] == FREE).all() and (g[:, :col_wall] == UNKNOWN).all()


def test_free_cell_near():
    occ = np.zeros((5, 5), np.int8)
    occ[2, 2] = WALL
    assert free_cell_near(occ, (2, 2)) != (2, 2) and occ[free_cell_near(occ, (2, 2))] != WALL
    assert free_cell_near(occ, (1, 1)) == (1, 1)
