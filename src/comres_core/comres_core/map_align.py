"""
Where one robot's SLAM map lies in another's: the transform between two robots' map frames,
estimated from the maps themselves (no known start poses), to fuse several robots' maps.

T = (x, y, yaw) is the pose of map B's frame in map A's frame: a point p of B lies at
R(yaw) p + (x, y) in A. The maps are 2D occupancy grids (nav_msgs/OccupancyGrid values: -1
unknown, 0..100; occupied >= 65, free <= 25) of levelled SLAM (the Go2's RTAB-Map), so T has three
degrees of freedom, and it stays put while the robots move (it drifts only as their SLAM does).

  candidates(a, b)  a global search: at every yaw step, B's obstacle points (rotated) are
                    correlated by FFT with A's score image over all translations; the best
                    distinct transforms come back with their scores.
  AlignmentFilter   a particle filter over T, seeded around the candidates and, for a share of
                    the particles, anywhere robot B could stand in A's free space. Each update
                    weighs the particles by what robot B sees now (its map's obstacle and free
                    cells within range, in B's frame) against A: an obstacle by A's likelihood
                    field (its distance to A's nearest obstacle), much worse on A's free space; a
                    free cell is penalised on A's obstacles; unknown and unmapped cells are
                    neutral. Old evidence fades (`forget`): the same map seen again is not new
                    information, so noise cannot add up to a false certainty. Low-variance
                    resampling, then a small diffusion (SLAM drift).
  ready             the particles form one cluster (most of the weight within 0.3 m and 4 deg,
                    spread under 0.1 m and 1.5 deg) whose transform, checked on the whole maps,
                    puts enough of B's obstacles on A's (`min_agree`), contradicts A little (B's
                    obstacles on A's free space, B's free cells on A's obstacles: `max_conflict`
                    of B's points on A's known space), and beats every plausible rival by
                    `min_margin`, several updates in a row. Rivals are the global search's other
                    candidates, refined locally on the whole maps, that would pass those checks
                    themselves: in a symmetric place another transform fits as well, and the filter
                    waits. Fuse the maps only when ready.
"""

import math
from dataclasses import dataclass
from typing import Dict, List, Optional, Tuple

import numpy as np
from scipy import fft as sfft
from scipy import ndimage

OCCUPIED, FREE = 65, 25
Pose2D = Tuple[float, float, float]                  # x, y, yaw


def rot(yaw: float) -> np.ndarray:
    c, s = math.cos(yaw), math.sin(yaw)
    return np.array([[c, -s], [s, c]])


def wrap(a):
    return (np.asarray(a) + np.pi) % (2 * np.pi) - np.pi


def compose(a: Pose2D, b: Pose2D) -> Pose2D:
    """a o b: the pose b (given in a's frame) in the frame a is given in."""
    x, y = rot(a[2]) @ np.array(b[:2]) + np.array(a[:2])
    return (float(x), float(y), float(wrap(a[2] + b[2])))


def transform(T: Pose2D, p: np.ndarray) -> np.ndarray:
    """Points p (n, 2) of B's frame in A's frame."""
    return p @ rot(T[2]).T + np.array(T[:2])


def thin(p: np.ndarray, cell: float) -> np.ndarray:
    """One point per cell of side `cell`."""
    if len(p) == 0:
        return p
    _, idx = np.unique(np.floor(p / cell).astype(np.int64), axis=0, return_index=True)
    return p[np.sort(idx)]


def turned(T: Pose2D, moves: np.ndarray, pivot: np.ndarray) -> np.ndarray:
    """Poses (n, 3): T, then each move (dx, dy, dyaw) in A's frame, turning about `pivot` (in A)."""
    c, s = np.cos(moves[:, 2]), np.sin(moves[:, 2])
    rx, ry = T[0] - pivot[0], T[1] - pivot[1]
    return np.column_stack([pivot[0] + c * rx - s * ry + moves[:, 0], pivot[1] + s * rx + c * ry + moves[:, 1],
                            wrap(T[2] + moves[:, 2])])


def distinct(T: Pose2D, U: Pose2D, sep: Tuple[float, float] = (0.5, math.radians(8.0))) -> bool:
    return math.hypot(T[0] - U[0], T[1] - U[1]) > sep[0] or abs(float(wrap(T[2] - U[2]))) > sep[1]


@dataclass
class GridMap:
    """An occupancy grid as nav_msgs/OccupancyGrid lays it out: data[j, i] is cell (i, j), j along
    the map's y; origin is the pose of cell (0, 0)'s corner in the map frame."""
    data: np.ndarray
    resolution: float
    origin: Pose2D = (0.0, 0.0, 0.0)

    @property
    def occupied(self) -> np.ndarray:
        return self.data >= OCCUPIED

    @property
    def free(self) -> np.ndarray:
        return (self.data >= 0) & (self.data <= FREE)

    def points(self, mask: np.ndarray) -> np.ndarray:
        """Map-frame centres (n, 2) of the cells in mask."""
        j, i = np.nonzero(mask)
        local = np.stack([(i + 0.5) * self.resolution, (j + 0.5) * self.resolution], axis=1)
        return transform(self.origin, local)

    def cell_of(self, x: np.ndarray, y: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
        """Integer (i, j) of map-frame coordinates (any shape; outside the grid: out of range)."""
        ox, oy, oyaw = self.origin
        c, s = math.cos(oyaw), math.sin(oyaw)
        dx, dy = x - ox, y - oy
        return (np.floor((c * dx + s * dy) / self.resolution).astype(np.int64),
                np.floor((-s * dx + c * dy) / self.resolution).astype(np.int64))


def map_points(m: GridMap, center: Optional[Tuple[float, float]] = None, radius: float = 5.0,
               obstacle_cell: float = 0.1, free_cell: float = 0.25, max_obstacles: int = 500, max_free: int = 250,
               rng: Optional[np.random.Generator] = None) -> Tuple[np.ndarray, np.ndarray]:
    """A map's obstacle and free points (map frame), thinned and capped (randomly): within `radius`
    of `center` (what a robot there sees), or of the whole map (center None)."""
    rng = rng if rng is not None else np.random.default_rng(0)
    out = []
    for mask, cell, cap in ((m.occupied, obstacle_cell, max_obstacles), (m.free, free_cell, max_free)):
        p = m.points(mask)
        if center is not None:
            p = p[np.hypot(p[:, 0] - center[0], p[:, 1] - center[1]) <= radius]
        p = thin(p, cell)
        if cap and len(p) > cap:
            p = p[rng.choice(len(p), cap, replace=False)]
        out.append(p)
    return out[0], out[1]


class ScoreField:
    """How a point of map B fits map A where it lands, as log-likelihoods on A's grid."""

    P_UNKNOWN = 0.1          # an obstacle landing on A's unknown space or outside A: no evidence
    P_CONFLICT = 0.01        # an obstacle on A's free space, or a free cell on A's obstacle

    def __init__(self, a: GridMap, sigma: float = 0.1):
        self.a = a
        occ, free = a.occupied, a.free
        if occ.any():
            d = ndimage.distance_transform_edt(~occ) * a.resolution
        else:
            d = np.full(occ.shape, np.inf)
        self.distance = d                                    # m to A's nearest obstacle (0 on them)
        hit = np.log(0.9 * np.exp(-0.5 * (d / sigma) ** 2) + 0.02)
        near = d <= 2 * sigma
        self.obstacle = np.where(near, hit, math.log(self.P_UNKNOWN))
        self.obstacle[free & ~near] = math.log(self.P_CONFLICT)
        self.free = np.where(occ, math.log(self.P_CONFLICT), 0.0)
        self.known = (occ | free).astype(float)

    def lookup(self, table: np.ndarray, x: np.ndarray, y: np.ndarray, outside: float) -> np.ndarray:
        i, j = self.a.cell_of(x, y)
        inside = (i >= 0) & (i < table.shape[1]) & (j >= 0) & (j < table.shape[0])
        out = np.full(np.shape(x), outside, dtype=float)
        out[inside] = table[j[inside], i[inside]]
        return out

    def score(self, poses: np.ndarray, obstacles: np.ndarray, free: np.ndarray, free_weight: float = 0.5) -> np.ndarray:
        """Mean log-likelihood per point of each pose (n, 3) of B in A."""
        total = np.zeros(len(poses))
        for pts, table, outside, w in ((obstacles, self.obstacle, math.log(self.P_UNKNOWN), 1.0),
                                       (free, self.free, 0.0, free_weight)):
            if len(pts) == 0:
                continue
            c, s = np.cos(poses[:, 2])[:, None], np.sin(poses[:, 2])[:, None]
            x = poses[:, 0:1] + c * pts[None, :, 0] - s * pts[None, :, 1]
            y = poses[:, 1:2] + s * pts[None, :, 0] + c * pts[None, :, 1]
            total += w * self.lookup(table, x, y, outside).mean(axis=1)
        return total

    def check(self, T: Pose2D, obstacles: np.ndarray, free: np.ndarray, near: float = 0.15, far: float = 0.25) -> Dict:
        """How B's points agree with A at T: `agree` B obstacles within `near` of A's; `conflict`
        the share, of B's points on A's known space, of B obstacles on A's free space (beyond `far`
        from A's obstacles) and B free points on A's obstacles."""
        po, pf = transform(T, obstacles), transform(T, free)
        d_o = self.lookup(self.distance, po[:, 0], po[:, 1], np.inf)
        k_o = self.lookup(self.known, po[:, 0], po[:, 1], 0.0) > 0
        d_f = self.lookup(self.distance, pf[:, 0], pf[:, 1], np.inf)
        k_f = self.lookup(self.known, pf[:, 0], pf[:, 1], 0.0) > 0
        agree = int((d_o <= near).sum())
        bad = int((k_o & (d_o > far)).sum() + (d_f < 0.5 * self.a.resolution).sum())
        on_known = int((k_o | (d_o <= near)).sum() + k_f.sum())
        return {"agree": agree, "conflict": bad / on_known if on_known else 0.0, "on_known": on_known}


def candidates(a: GridMap, b: GridMap, res: float = 0.1, yaw_step: float = math.radians(2.0), k: int = 20,
               sigma: float = 0.1) -> List[Tuple[float, Pose2D]]:
    """The k best distinct transforms of B into A, (score, T), best first: B's obstacle points
    (thinned to res) rotated by every yaw step and correlated by FFT with A's score image (near A's
    obstacles up to 1, A's free space -0.5, unknown 0) over all translations; the score is the
    mean per point."""
    q = thin(b.points(b.occupied), res)
    if len(q) == 0 or not a.occupied.any():
        return []
    # A's score image on its own (axis-aligned) grid frame, at res
    f = max(1, int(round(res / a.resolution)))
    res = f * a.resolution
    d = ndimage.distance_transform_edt(~a.occupied) * a.resolution
    s_img = np.where(a.free, -0.5, 0.0)
    s_img = np.where(d <= 2 * sigma, np.exp(-0.5 * (d / sigma) ** 2), s_img)
    h, w = (s_img.shape[0] // f) * f, (s_img.shape[1] // f) * f
    s_img = s_img[:h, :w].reshape(h // f, f, w // f, f).mean(axis=(1, 3))
    # B's points around their centroid, on a canvas that fits every rotation
    c = q.mean(axis=0)
    rho = float(np.max(np.hypot(*(q - c).T))) + res
    n_p = int(math.ceil(2 * rho / res)) + 1
    shape = (s_img.shape[0] + n_p - 1, s_img.shape[1] + n_p - 1)
    s_fft = sfft.rfft2(s_img, shape)
    found = []
    for yaw in np.arange(0.0, 2 * math.pi, yaw_step):
        r = rot(yaw)
        qr = (q - c) @ r.T
        u = np.clip(np.floor((qr[:, 0] + rho) / res).astype(int), 0, n_p - 1)
        v = np.clip(np.floor((qr[:, 1] + rho) / res).astype(int), 0, n_p - 1)
        canvas = np.zeros((n_p, n_p))
        canvas[v, u] = 1.0
        corr = sfft.irfft2(s_fft * sfft.rfft2(canvas[::-1, ::-1], shape), shape) / len(q)
        peaks = (corr == ndimage.maximum_filter(corr, size=5)) & (corr > 0)
        idx = np.argwhere(peaks)
        if len(idx) == 0:
            continue
        for kv, ku in idx[np.argsort(corr[peaks])[::-1][:3]]:
            dv, du = kv - (n_p - 1), ku - (n_p - 1)
            t_grid = np.array([du * res + rho, dv * res + rho]) - r @ c
            found.append((float(corr[kv, ku]), compose(a.origin, (t_grid[0], t_grid[1], float(yaw)))))
    found.sort(key=lambda sp: -sp[0])
    kept: List[Tuple[float, Pose2D]] = []
    for sc, T in found:
        if all(distinct(T, K) for _, K in kept):
            kept.append((sc, T))
            if len(kept) == k:
                break
    return kept


def refine(field: ScoreField, T: Pose2D, points: Tuple[np.ndarray, np.ndarray],
           steps=((0.1, math.radians(1.0)), (0.05, math.radians(0.5)), (0.02, math.radians(0.2))),
           max_moves: int = 10) -> Pose2D:
    """T moved to the best score of B's points (map_points(b)) nearby: hill climbing over the 27
    moves of one step in x, y and yaw (B's map turned about its centroid), for each step in turn."""
    obstacles, free = points
    if len(obstacles) == 0:
        return T
    d = np.array(np.meshgrid([-1, 0, 1], [-1, 0, 1], [-1, 0, 1], indexing="ij")).reshape(3, -1).T
    stay = int(np.flatnonzero(~d.any(axis=1))[0])
    for step_xy, step_yaw in steps:
        for _ in range(max_moves):
            poses = turned(T, d * np.array([step_xy, step_xy, step_yaw]), transform(T, obstacles).mean(axis=0))
            scores = field.score(poses, obstacles, free)
            best = int(np.argmax(scores))
            if scores[best] <= scores[stay] + 1e-12:
                break
            T = tuple(map(float, poses[best]))
    return T


class AlignmentFilter:
    """A particle filter over T, the pose of map B's frame in map A's frame (see the module)."""

    def __init__(self, n: int = 2000, seed: int = 0, seed_spread: Tuple[float, float] = (0.15, math.radians(3.0)),
                 anywhere: float = 0.1, diffusion: Tuple[float, float] = (0.02, math.radians(0.3)),
                 points_per_update: float = 30.0, forget: float = 0.5, ready_after: int = 3,
                 min_agree: int = 60, max_conflict: float = 0.05, min_margin: float = 0.15):
        self.n = n
        self.rng = np.random.default_rng(seed)
        self.seed_spread, self.anywhere, self.diffusion = seed_spread, anywhere, diffusion
        self.gamma = points_per_update            # how many independent points one update counts as
        self.forget = forget                      # how much of the old evidence an update keeps
        self.ready_after, self.min_agree, self.max_conflict, self.min_margin = ready_after, min_agree, max_conflict, min_margin
        self.particles = np.zeros((0, 3))
        self.logw = np.zeros(0)
        self.candidates: List[Pose2D] = []        # the global search's, as last given
        self.rivals: Optional[List[Pose2D]] = None   # those refined and plausible (None: to work out)
        self.streak = 0
        self.ready = False
        self.last: Dict = {}

    # -- particles ------------------------------------------------------------------------
    def _around(self, cands: List[Tuple[float, Pose2D]], n: int) -> np.ndarray:
        poses = np.array([T for _, T in cands])
        p = poses[self.rng.integers(len(poses), size=n)]
        p[:, :2] += self.rng.normal(0.0, self.seed_spread[0], (n, 2))
        p[:, 2] = wrap(p[:, 2] + self.rng.normal(0.0, self.seed_spread[1], n))
        return p

    def _anywhere(self, a: GridMap, robot_b: Tuple[float, float], n: int) -> np.ndarray:
        """Transforms that put robot B (at robot_b in B's frame) on a random free cell of A."""
        free = a.points(a.free)
        if len(free) == 0 or n == 0:
            return np.zeros((0, 3))
        f = free[self.rng.integers(len(free), size=n)]
        yaw = self.rng.uniform(-math.pi, math.pi, n)
        c, s = np.cos(yaw), np.sin(yaw)
        t = f - np.stack([c * robot_b[0] - s * robot_b[1], s * robot_b[0] + c * robot_b[1]], axis=1)
        return np.column_stack([t, yaw])

    def seed(self, cands: List[Tuple[float, Pose2D]], a: GridMap, robot_b: Tuple[float, float]) -> None:
        """New particles: around the candidates, and `anywhere` of them anywhere B could stand in A."""
        n_any = self.n if not cands else int(self.n * self.anywhere)
        self.particles = np.vstack([self._around(cands, self.n - n_any) if cands else np.zeros((0, 3)),
                                    self._anywhere(a, robot_b, n_any)])
        self.logw = np.zeros(len(self.particles))
        self.candidates, self.rivals = [T for _, T in cands], None
        self.streak, self.ready = 0, False

    def inject(self, cands: List[Tuple[float, Pose2D]], share: float = 0.2) -> None:
        """New candidates (after the maps have grown): the rivals are worked out from them again,
        and particles around them replace the lightest `share`."""
        if not cands:
            return
        self.candidates, self.rivals = [T for _, T in cands], None
        self._replace_lightest(cands, share)

    def _replace_lightest(self, cands: List[Tuple[float, Pose2D]], share: float) -> None:
        if len(self.particles) == 0:
            return
        m = int(len(self.particles) * share)
        worst = np.argsort(self.logw)[:m]
        self.particles[worst] = self._around(cands, m)
        self.logw[worst] = np.median(self.logw)

    # -- the filter ---------------------------------------------------------------------------
    def update(self, field: ScoreField, seen: Tuple[np.ndarray, np.ndarray], whole_b: Tuple[np.ndarray, np.ndarray]) -> Dict:
        """Weigh the particles by what robot B sees now (`seen`: obstacle and free points near it,
        B's frame), then check the estimate on B's whole map (`whole_b`, map_points(b))."""
        obstacles, free = seen
        if len(self.particles) and len(obstacles):
            self.logw = self.forget * self.logw + self.gamma * field.score(self.particles, obstacles, free)
            self.logw -= self.logw.max()
            w = np.exp(self.logw)
            w /= w.sum()
            if 1.0 / np.sum(w ** 2) < 0.5 * len(w):           # low-variance resampling
                u = (self.rng.random() + np.arange(len(w))) / len(w)
                idx = np.minimum(np.searchsorted(np.cumsum(w), u), len(w) - 1)
                self.particles = self.particles[idx]
                self.particles[:, :2] += self.rng.normal(0.0, self.diffusion[0], (len(w), 2))
                self.particles[:, 2] = wrap(self.particles[:, 2] + self.rng.normal(0.0, self.diffusion[1], len(w)))
                self.logw = np.zeros(len(w))
        est = self.estimate(field, whole_b)
        if est["T"] is not None and est["single"] and est["agrees"] and est["consistent"] and est["unique"]:
            self.streak += 1
            self.ready = self.ready or self.streak >= self.ready_after
        else:
            self.streak, self.ready = 0, False
            if est["T"] is not None and est["single"] and not (est["agrees"] and est["consistent"]):
                # one cluster, but the whole maps refute it: particles to the transforms they allow
                others = [(0.0, R) for R in self.rivals or [] if distinct(est["T"], R)]
                if others:
                    self._replace_lightest(others, 0.3)
        est["ready"] = self.ready
        self.last = est
        return est

    def estimate(self, field: Optional[ScoreField] = None, whole_b: Optional[Tuple[np.ndarray, np.ndarray]] = None,
                 radius: Tuple[float, float] = (0.3, math.radians(4.0))) -> Dict:
        """The main cluster (the particles within `radius` of the heaviest 0.2 m x 3 deg bin): its
        weighted mean and spread and the weight it carries; and, given A's field and B's whole map,
        how its transform agrees with A, and its margin over the best distinct rival."""
        out = {"T": None, "ready": self.ready, "single": False, "agrees": False, "consistent": False, "unique": False}
        if len(self.particles) == 0:
            return out
        w = np.exp(self.logw - self.logw.max())
        w /= w.sum()
        p = self.particles
        keys = np.column_stack([np.floor(p[:, 0] / 0.2), np.floor(p[:, 1] / 0.2),
                                np.floor((p[:, 2] + math.pi) / math.radians(3.0))]).astype(np.int64)
        _, inv = np.unique(keys, axis=0, return_inverse=True)
        inv = inv.ravel()
        mode = p[np.argmax(w * np.bincount(inv, weights=w)[inv])]
        near = (np.hypot(p[:, 0] - mode[0], p[:, 1] - mode[1]) <= radius[0]) & (np.abs(wrap(p[:, 2] - mode[2])) <= radius[1])
        wc = w[near] / w[near].sum()
        xy = (p[near, :2] * wc[:, None]).sum(axis=0)
        yaw = math.atan2(float((np.sin(p[near, 2]) * wc).sum()), float((np.cos(p[near, 2]) * wc).sum()))
        T = (float(xy[0]), float(xy[1]), float(yaw))
        sd_xy = float(math.sqrt((wc * ((p[near, 0] - xy[0]) ** 2 + (p[near, 1] - xy[1]) ** 2)).sum() / 2))
        sd_yaw = float(math.sqrt((wc * wrap(p[near, 2] - yaw) ** 2).sum()))
        mass = float(w[near].sum())
        single = mass >= 0.9 and sd_xy <= 0.1 and sd_yaw <= math.radians(1.5)
        if field is not None and whole_b is not None:
            obstacles, free = whole_b
            if single:                 # the cluster's transform refined on the whole maps; the cluster follows
                R = refine(field, T, whole_b)
                dyaw = float(wrap(R[2] - T[2]))
                q = p[near, :2] @ rot(dyaw).T
                self.particles[near, :2] = q + (np.array(R[:2]) - rot(dyaw) @ np.array(T[:2]))
                self.particles[near, 2] = wrap(p[near, 2] + dyaw)
                T = R
        out.update(T=T, sd_xy=sd_xy, sd_yaw=sd_yaw, mass=mass, single=single)
        if field is not None and whole_b is not None:
            chk = field.check(T, obstacles, free)
            if self.rivals is None:
                self.rivals = self.plausible(field, whole_b)
            rivals = [R for R in self.rivals if distinct(T, R)]
            scores = field.score(np.array([T] + rivals), obstacles, free)
            margin = float(scores[0] - scores[1:].max()) if rivals else math.inf
            out.update(agree=chk["agree"], conflict=chk["conflict"], margin=margin, rivals=len(rivals),
                       agrees=chk["agree"] >= self.min_agree, consistent=chk["conflict"] <= self.max_conflict,
                       unique=margin >= self.min_margin)
        return out

    def plausible(self, field: ScoreField, whole_b: Tuple[np.ndarray, np.ndarray]) -> List[Pose2D]:
        """The candidates, each refined on the whole maps, that agree with A and contradict it
        little: transforms that could be the truth."""
        obstacles, free = whole_b
        out = []
        for T in self.candidates:
            R = refine(field, T, whole_b)
            chk = field.check(R, obstacles, free)
            if chk["agree"] >= self.min_agree and chk["conflict"] <= self.max_conflict:
                out.append(R)
        return out
