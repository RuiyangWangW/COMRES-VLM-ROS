"""
Places every robot's SLAM map in one shared frame, world_frame, from the maps themselves
(comres_core.map_align): no known start poses. The first robot's map frame is the world; each
other robot's map is aligned to it by a particle filter (seeded by a global search over the two
maps, redone as they grow, and updated every second with what that robot sees now), and the
transform world -> {robot}/map is published as a static TF once the filter is ready. Until then
COMRES waits on that TF ("no TF world <- go2_2/map"): the maps are never fused on a guess.

Once fused, it follows the maps: each robot's SLAM reshapes its map at its loop closures, so the
transform is republished whenever the estimate is a tight cluster that agrees with the first
robot's map (conflict up to track_max_conflict) and lies within track_max_move / track_max_turn_deg
of the published one; a bigger change needs the full readiness checks again. The global search
runs in the background (it takes up to a minute on large maps), before fusion every
search_period, after it only when the estimate has not fitted for lost_after seconds; on maps over
fine_extent metres it searches on 0.2 m cells instead of 0.1 m.

    ros2 run comres_ros comres_map_align --ros-args --params-file <config.yaml>   (section map_align)

In: /{robot}/map (dog_link's), TF {robot}/map <- the robot's base. Out: the static TFs, and
/comres/map_align (std_msgs/String, latched, every second): JSON per robot {ready, T, agree,
conflict, margin, rivals, mass, published, mode, search}. In RViz, fixed frame world: both maps where they were fused.
"""
import json
import math
import threading
import time
import zlib
from typing import Dict, Optional

import numpy as np
import rclpy
from geometry_msgs.msg import TransformStamped
from nav_msgs.msg import OccupancyGrid
from rclpy.executors import ExternalShutdownException
from rclpy.node import Node
from rclpy.qos import DurabilityPolicy, QoSProfile, ReliabilityPolicy
from rclpy.time import Time
from std_msgs.msg import String
from tf2_ros import Buffer, StaticTransformBroadcaster, TransformListener

from comres_core.map_align import AlignmentFilter, GridMap, ScoreField, candidates, distinct, map_points

BASE_CANDIDATES = ("{robot}/base_footprint", "{robot}/base_link")


def yaw_of(q) -> float:
    return math.atan2(2.0 * (q.w * q.z + q.x * q.y), 1.0 - 2.0 * (q.y * q.y + q.z * q.z))


def grid_of(m: OccupancyGrid) -> GridMap:
    o = m.info.origin
    data = np.asarray(m.data, dtype=np.int16).reshape(m.info.height, m.info.width)
    return GridMap(data, m.info.resolution, (o.position.x, o.position.y, yaw_of(o.orientation)))


def num(v):
    """A JSON-safe number (inf and nan: None)."""
    if isinstance(v, float):
        return round(v, 3) if math.isfinite(v) else None
    return v


def key_of(m: OccupancyGrid):
    """The same map, republished, has the same key."""
    o = m.info.origin.position
    return m.info.width, m.info.height, round(o.x, 3), round(o.y, 3), zlib.crc32(bytes(m.data))


class Pair:
    """Robot k's map against the first robot's: its filter, and the maps it was last given."""

    def __init__(self, robot: str, seed: int, max_conflict: float):
        self.robot = robot
        self.filter = AlignmentFilter(seed=seed, max_conflict=max_conflict)
        self.b: Optional[GridMap] = None
        self.whole = None                     # map_points of its whole map
        self.key = None
        self.seeded = False
        self.searched_keys = None             # the maps' keys at the last global search
        self.searched_at = 0.0
        self.published = None                 # the transform last published
        self.fitted_at = 0.0                  # when the estimate last fitted (once fused)
        self.search: Optional[threading.Thread] = None   # the global search running, if any
        self.found = None                     # its result, for the next tick: (candidates, s, cell)
        self.base: Optional[str] = None
        self.status: Dict = {}


class MapAlign(Node):
    def __init__(self):
        super().__init__("map_align")
        self.robots = list(self.declare_parameter("robots", ["go2_1", "go2_2"]).value)
        self.world = self.declare_parameter("world_frame", "world").value
        topic = self.declare_parameter("map_topic", "/{robot}/map").value
        self.base_fmt = self.declare_parameter("base_frame", "").value        # "": the first of BASE_CANDIDATES
        self.search_period = float(self.declare_parameter("search_period", 10.0).value)
        self.lost_after = float(self.declare_parameter("lost_after", 20.0).value)
        self.fine_extent = float(self.declare_parameter("fine_extent", 30.0).value)
        self.track_move = (float(self.declare_parameter("track_max_move", 2.0).value),
                           math.radians(float(self.declare_parameter("track_max_turn_deg", 15.0).value)))
        self.track_conflict = float(self.declare_parameter("track_max_conflict", 0.15).value)
        # the first fusion's conflict limit: 2026-11-18 the two Go2 maps, correctly aligned, had 5.5-5.7%
        max_conflict = float(self.declare_parameter("max_conflict", 0.08).value)
        period = float(self.declare_parameter("period", 1.0).value)
        if len(self.robots) < 2:
            raise ValueError("map_align needs two robots or more (the first one's map is the world)")
        self.tf = Buffer()
        self.tf_listener = TransformListener(self.tf, self)
        self.static = StaticTransformBroadcaster(self)
        self.msgs: Dict[str, OccupancyGrid] = {}
        self.a: Optional[GridMap] = None
        self.a_key = None
        self.a_frame: Optional[str] = None
        self.field: Optional[ScoreField] = None
        self.pairs = [Pair(r, seed=k, max_conflict=max_conflict) for k, r in enumerate(self.robots[1:], start=1)]
        self.tfs: Dict[str, TransformStamped] = {}
        qos = QoSProfile(depth=1, reliability=ReliabilityPolicy.RELIABLE, durability=DurabilityPolicy.TRANSIENT_LOCAL)
        for r in self.robots:
            self.create_subscription(OccupancyGrid, topic.format(robot=r), lambda m, r=r: self.msgs.__setitem__(r, m), qos)
        self.status_pub = self.create_publisher(String, "/comres/map_align", qos)
        self.create_timer(period, self.tick)
        self.get_logger().info(f"aligning {', '.join(self.robots[1:])} to {self.robots[0]} (its map frame is {self.world})")

    # -- helpers ------------------------------------------------------------------------------
    def robot_xy(self, p: Pair, frame: str):
        """Robot p's position in its own map frame, or None."""
        names = [p.base] if p.base else [self.base_fmt] if self.base_fmt else list(BASE_CANDIDATES)
        for f in names:
            f = f.format(robot=p.robot)
            try:
                t = self.tf.lookup_transform(frame, f, Time()).transform.translation
            except Exception:                                   # noqa: BLE001
                continue
            p.base = f
            return float(t.x), float(t.y)
        return None

    def publish_tf(self, frame: str, T) -> None:
        t = TransformStamped()
        t.header.stamp = self.get_clock().now().to_msg()
        t.header.frame_id, t.child_frame_id = self.world, frame
        t.transform.translation.x, t.transform.translation.y = float(T[0]), float(T[1])
        t.transform.rotation.z, t.transform.rotation.w = math.sin(T[2] / 2.0), math.cos(T[2] / 2.0)
        self.tfs[frame] = t
        self.static.sendTransform(list(self.tfs.values()))

    def start_search(self, p: Pair) -> None:
        """The global search over the current maps, in the background; step() takes its result."""
        a, b = self.a, p.b
        extent = max(max(m.data.shape) * m.resolution for m in (a, b))
        cell = 0.1 if extent <= self.fine_extent else 0.2

        def run():
            t0 = time.monotonic()
            try:
                cands = candidates(a, b, res=cell)
            except Exception as e:                              # noqa: BLE001
                self.get_logger().error(f"{p.robot}: global search failed: {e}")
                cands = []
            p.found = (cands, time.monotonic() - t0, cell)
            p.search = None

        p.search = threading.Thread(target=run, daemon=True)
        p.search.start()

    # -- the loop -----------------------------------------------------------------------------
    def tick(self) -> None:
        m = self.msgs.get(self.robots[0])
        if m is None:
            self.get_logger().info(f"waiting for {self.robots[0]}'s map", throttle_duration_sec=10.0)
            return
        k = key_of(m)
        if k != self.a_key:
            self.a, self.a_key = grid_of(m), k
            self.field = ScoreField(self.a)
            if self.a_frame != m.header.frame_id:
                self.a_frame = m.header.frame_id
                self.publish_tf(self.a_frame, (0.0, 0.0, 0.0))
        for p in self.pairs:
            self.step(p)
        self.status_pub.publish(String(data=json.dumps({p.robot: p.status for p in self.pairs})))

    def step(self, p: Pair) -> None:
        m = self.msgs.get(p.robot)
        if m is None:
            p.status = {"ready": False, "why": "no map yet"}
            return
        k = key_of(m)
        if k != p.key:
            p.b, p.key = grid_of(m), k
            p.whole = map_points(p.b, max_obstacles=1500, max_free=1500)
        if not p.b.occupied.any() or not self.a.occupied.any():
            p.status = {"ready": False, "why": "a map has no obstacles yet"}
            return
        xy = self.robot_xy(p, m.header.frame_id)
        if xy is None:
            p.status = {"ready": False, "why": f"no TF {m.header.frame_id} <- {p.robot}'s base"}
            return
        now = time.monotonic()
        keys = (self.a_key, p.key)
        if p.found is not None:                                  # a global search has finished
            cands, secs, cell = p.found
            p.found = None
            if cands:
                p.filter.seed(cands, self.a, xy) if not p.seeded else p.filter.inject(cands)
                p.seeded = True
            self.get_logger().info(f"{p.robot}: global search, {len(cands)} candidates "
                                   f"({secs:.1f} s, {cell:.1f} m cells)")
        if p.published is None:                                  # not fused: search as the maps grow
            due = not p.seeded or (keys != p.searched_keys and now - p.searched_at >= self.search_period)
        else:                                                    # fused: only once the estimate is lost
            due = now - p.fitted_at >= self.lost_after and now - p.searched_at >= self.search_period
        if due and p.search is None:
            self.start_search(p)
            p.searched_keys, p.searched_at = keys, now
        mode = "searching" if p.published is None else (
            "following" if now - p.fitted_at < self.lost_after else "lost: searching again")
        if not p.seeded:
            p.status = {"ready": False, "mode": mode, "why": "the first global search is running"}
            return
        est = p.filter.update(self.field, map_points(p.b, center=xy, rng=p.filter.rng), p.whole)
        T = est["T"]
        p.status = {"ready": bool(est["ready"]), "T": [num(v) for v in T] if T else None,
                    **{f: num(est.get(f)) for f in ("agree", "conflict", "margin", "rivals", "mass")},
                    "published": [num(v) for v in p.published] if p.published else None,
                    "mode": mode, "search": "running" if p.search is not None else None}
        # once fused: follow the maps while the estimate fits and stays near the published transform
        fits = (T is not None and p.published is not None and est["single"]
                and (est.get("agree") or 0) >= p.filter.min_agree
                and (est.get("conflict") if est.get("conflict") is not None else 1.0) <= self.track_conflict
                and not distinct(T, p.published, sep=self.track_move))
        if fits:
            p.fitted_at = now
        if (est["ready"] or fits) and (p.published is None or distinct(T, p.published, sep=(0.02, math.radians(0.2)))):
            first, old = p.published is None, p.published
            p.published, p.fitted_at = T, now
            self.publish_tf(m.header.frame_id, T)
            p.status["published"] = [num(v) for v in T]
            if first:
                self.get_logger().info(
                    f"{p.robot}: fused, {self.world} -> {m.header.frame_id} = ({T[0]:.2f} m, {T[1]:.2f} m, "
                    f"{math.degrees(T[2]):.1f} deg); agree {est['agree']}, conflict {est['conflict']:.3f}, "
                    f"margin {est['margin']:.2f}")
            else:
                moved = math.hypot(T[0] - old[0], T[1] - old[1])
                turned = abs(math.degrees(math.atan2(math.sin(T[2] - old[2]), math.cos(T[2] - old[2]))))
                log = self.get_logger().info if moved > 0.1 or turned > 1.0 else self.get_logger().debug
                log(f"{p.robot}: following its map ({'ready' if est['ready'] else 'tracking'}): "
                    f"({T[0]:.2f} m, {T[1]:.2f} m, {math.degrees(T[2]):.1f} deg), moved {moved:.2f} m "
                    f"{turned:.1f} deg; agree {est['agree']}, conflict {est['conflict']:.3f}")


def main(args=None):
    rclpy.init(args=args)
    node = MapAlign()
    try:
        rclpy.spin(node)
    except (KeyboardInterrupt, ExternalShutdownException):
        pass
    finally:
        node.destroy_node()
        rclpy.try_shutdown()


if __name__ == "__main__":
    main()
