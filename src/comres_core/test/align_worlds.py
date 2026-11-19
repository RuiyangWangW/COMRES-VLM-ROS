"""Worlds and robot maps for the map-alignment tests (test_map_align.py) and tools/map_align_eval.py.

A world is a grid at RES (0 free, 100 occupied, -1 unknown) in the world frame, its cell (0, 0) at
the origin. A robot's SLAM map of it is view(): the world's cells within a region, sampled in a map
frame whose pose in the world is T, with lidar-like noise.
"""
import math
import os

import numpy as np
from scipy import ndimage

from comres_core.map_align import GridMap, rot, transform

RES = 0.05
DATA = os.path.join(os.path.dirname(os.path.abspath(__file__)), "data")


def outlines(g: np.ndarray) -> np.ndarray:
    """Obstacles as a lidar maps them: their outline occupied, their inside unknown."""
    occ = g >= 65
    inside = occ & ndimage.binary_erosion(occ, structure=np.ones((3, 3), bool))
    out = g.copy()
    out[inside] = -1
    return out


def floor_plan(seed: int = 0, size=(16.0, 12.0), res: float = RES) -> np.ndarray:
    """A random office floor: outer walls, inner walls with 1 m doors, boxes and round tables."""
    rng = np.random.default_rng(seed)
    h, w = int(size[1] / res), int(size[0] / res)
    g = np.zeros((h, w), np.int16)
    t = max(1, int(round(0.1 / res)))                       # 0.1 m walls
    door = int(1.0 / res)
    g[:t, :] = g[-t:, :] = g[:, :t] = g[:, -t:] = 100
    for _ in range(rng.integers(2, 4)):
        x = int(rng.uniform(0.2, 0.8) * w)
        g[:, x:x + t] = 100
        for _ in range(rng.integers(1, 3)):
            y = int(rng.uniform(0.1, 0.85) * h)
            g[y:y + door, x:x + t] = 0
    for _ in range(rng.integers(1, 3)):
        y = int(rng.uniform(0.2, 0.8) * h)
        g[y:y + t, :] = 100
        for _ in range(rng.integers(1, 3)):
            x = int(rng.uniform(0.1, 0.85) * w)
            g[y:y + t, x:x + door] = 0
    yy, xx = np.ogrid[:h, :w]
    for _ in range(rng.integers(12, 25)):
        cx, cy = rng.uniform(0.05, 0.95) * w, rng.uniform(0.05, 0.95) * h
        if rng.random() < 0.6:
            bw, bh = rng.uniform(0.3, 1.2) / res, rng.uniform(0.3, 0.8) / res
            g[int(cy):int(cy + bh), int(cx):int(cx + bw)] = 100
        else:
            r = rng.uniform(0.2, 0.5) / res
            g[(yy - cy) ** 2 + (xx - cx) ** 2 <= r * r] = 100
    return outlines(g)


def twin_rooms(res: float = RES) -> np.ndarray:
    """Two identical 5 m x 5 m rooms side by side (x 0.5..5.5 and 6.5..11.5, y 3..8), each with
    the same furniture and a door to the corridor south of them (y 0.5..2.5), whose only telling
    feature is a pillar at (2, 1.5)."""
    w, h = int(12.0 / res), int(9.0 / res)
    g = np.zeros((h, w), np.int16)
    t = max(1, int(round(0.1 / res)))
    c = lambda m: int(round(m / res))                       # noqa: E731
    g[:t, :] = g[-t:, :] = g[:, :t] = g[:, -t:] = 100
    g[c(0.5) - t:c(0.5), c(0.5):c(11.5)] = 100              # corridor south wall
    g[c(3.0):c(3.0) + t, :] = 100                           # rooms' south wall, doors below
    for x0 in (0.5, 6.5):
        g[c(3.0):c(3.0) + t, c(x0 + 2.0):c(x0 + 3.0)] = 0   # door
        g[c(3.0):c(8.0), c(x0):c(x0) + t] = 100              # west wall of the room
        g[c(3.0):c(8.0), c(x0 + 5.0):c(x0 + 5.0) + t] = 100  # east wall
        g[c(8.0):c(8.0) + t, c(x0):c(x0 + 5.0)] = 100       # north wall
        g[c(5.0):c(5.8), c(x0 + 0.8):c(x0 + 2.0)] = 100     # a desk
        g[c(6.5):c(7.5), c(x0 + 3.5):c(x0 + 4.2)] = 100     # a cabinet
        yy, xx = np.ogrid[:h, :w]
        g[(yy - c(4.2)) ** 2 + (xx - c(x0 + 3.6)) ** 2 <= c(0.3) ** 2] = 100   # a round table
    yy, xx = np.ogrid[:h, :w]
    g[(yy - c(1.5)) ** 2 + (xx - c(2.0)) ** 2 <= c(0.25) ** 2] = 100           # the pillar
    g[:, :t] = 100
    return outlines(g)


def real_room() -> np.ndarray:
    """The Go2's RTAB-Map grid of a lab room (2026-10-24, 0.05 m): -1 unknown, 0 free, 100 occupied."""
    return np.load(os.path.join(DATA, "go2_room_map.npz"))["data"].astype(np.int16)


def view(world: np.ndarray, region, T=(0.0, 0.0, 0.0), res: float = RES, speckle: float = 0.003,
         dropout: float = 0.05, seed: int = 0) -> GridMap:
    """A robot's map of the world's `region` (x0, y0, x1, y1, world metres) in a frame whose pose
    in the world is T: speckle (free cells read occupied) and dropout (wall cells unknown)."""
    rng = np.random.default_rng(seed)
    x0, y0, x1, y1 = region
    corners = np.array([[x0, y0], [x1, y0], [x0, y1], [x1, y1]], float)
    cb = (corners - np.array(T[:2])) @ rot(T[2])                # world -> map frame
    lo, hi = cb.min(axis=0), cb.max(axis=0)
    w, h = int(math.ceil((hi[0] - lo[0]) / res)), int(math.ceil((hi[1] - lo[1]) / res))
    i, j = np.meshgrid(np.arange(w), np.arange(h))
    pm = np.stack([lo[0] + (i + 0.5) * res, lo[1] + (j + 0.5) * res], axis=-1).reshape(-1, 2)
    pw = transform(T, pm)
    wi, wj = np.floor(pw[:, 0] / RES).astype(int), np.floor(pw[:, 1] / RES).astype(int)
    inside = ((pw[:, 0] >= x0) & (pw[:, 0] < x1) & (pw[:, 1] >= y0) & (pw[:, 1] < y1)
              & (wi >= 0) & (wi < world.shape[1]) & (wj >= 0) & (wj < world.shape[0]))
    data = np.full(len(pw), -1, np.int16)
    data[inside] = world[wj[inside], wi[inside]]
    data[(data == 0) & (rng.random(len(data)) < speckle)] = 100
    data[(data >= 65) & (rng.random(len(data)) < dropout)] = -1
    return GridMap(data.reshape(h, w), res, (float(lo[0]), float(lo[1]), 0.0))


def in_frame(T, points_world) -> list:
    """World points in the frame whose pose in the world is T."""
    p = (np.asarray(points_world, float) - np.array(T[:2])) @ rot(T[2])
    return [tuple(map(float, q)) for q in p]
