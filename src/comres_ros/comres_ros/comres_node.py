"""
The central COMRES node (ROS 2 Jazzy). It runs comres_core.online.Mission in real time for a team
of robots that each run their own SLAM (e.g. SLAM Toolbox), Nav2 and target detector:

  in   /<robot>/map              nav_msgs/OccupancyGrid  each robot's live SLAM map
       TF                        world <- each SLAM map frame, and each map frame <- robot base
       /<robot>/detections       the robot's detector (detection_type: vision_msgs Detection2DArray
                                 or Detection3DArray, yolo_msgs DetectionArray, or std_msgs String)
  out  /<robot>/navigate_to_pose nav2_msgs/action/NavigateToPose, one goal at a time per robot
       ~/merged_map              nav_msgs/OccupancyGrid (world frame), the last merged map
       ~/targets                 geometry_msgs/PoseArray (world frame), the robots' targets
  srv  ~/start, ~/stop           std_srvs/Trigger

Every tick (tick_seconds; one tick = one simulator step) it converts each robot's SLAM map onto
the planning grid (comres_core.grid), reads the robots' cells and calls Mission.tick, which runs
COMRES as the simulator does: every T_H ticks the maps are merged and the VLM is asked (in the
background); a robot whose target was seen, reached or failed gets a new one from the COMRES
selector, sent to its Nav2 at once (Nav2 replaces the previous goal). The search ends when a
detector reports a target class detection_frames times in a row; exploration when no robot has
a frontier left for no_frontier_ticks ticks; either on ~/stop or after max_mission_seconds. One
node runs one mission: restart it for the next.

Frames: with one robot, the world frame and the goal frame default to its SLAM map's frame and the
base frame is found among the usual names, so nothing has to be set. The planning grid is centred
on the team's start (grid_auto_center) and turned by grid_yaw_deg: its +y is what the VLM's picture
and the hints call north, its +x east.

The VLM is our server (serve_vlm.sh in COMRES-VLM) at vlm_base_url; no key is read from files.
Everything is logged under run_dir/<start time>/: meta.json, events.jsonl, poses.jsonl,
plans.jsonl, what the VLM saw for every plan (plans/plan_<step>.png and .txt) and the merged map of
every merge (maps/merge_<step>.npz).
"""

import base64
import json
import math
import os
import time
import urllib.request
from typing import Dict, List, Optional

import numpy as np
import rclpy
from scipy import ndimage
from action_msgs.msg import GoalStatus
from geometry_msgs.msg import Pose, PoseArray, PoseStamped
from nav2_msgs.action import NavigateToPose
from nav_msgs.msg import OccupancyGrid
from rcl_interfaces.msg import ParameterDescriptor
from rclpy.action import ActionClient
from rclpy.node import Node
from rclpy.qos import DurabilityPolicy, QoSProfile, ReliabilityPolicy, qos_profile_sensor_data
from rclpy.time import Time
from std_srvs.srv import Trigger
from tf2_ros import Buffer, TransformListener

from comres_core import core_version
from comres_core.grid import GridSpec, compose, invert, occupancy_to_grid

BASE_CANDIDATES = ("{robot}/base_footprint", "{robot}/base_link", "base_footprint", "base_link")
DETECTION_TYPES = {
    "vision_msgs/Detection2DArray": ("vision_msgs.msg", "Detection2DArray"),
    "vision_msgs/Detection3DArray": ("vision_msgs.msg", "Detection3DArray"),
    "yolo_msgs/DetectionArray": ("yolo_msgs.msg", "DetectionArray"),
    "std_msgs/String": ("std_msgs.msg", "String"),
}


def yaw_of(q) -> float:
    return math.atan2(2.0 * (q.w * q.z + q.x * q.y), 1.0 - 2.0 * (q.y * q.y + q.z * q.z))


def quat_of(yaw: float):
    return 0.0, 0.0, math.sin(yaw / 2.0), math.cos(yaw / 2.0)


def _jsonable(v):
    if isinstance(v, dict):
        return {str(k): _jsonable(x) for k, x in v.items()}
    if isinstance(v, (list, tuple)):
        return [_jsonable(x) for x in v]
    if isinstance(v, np.ndarray):
        return None if v.size > 64 else v.tolist()
    if isinstance(v, np.integer):
        return int(v)
    if isinstance(v, np.floating):
        return float(v)
    return v


def detections_of(msg, kind: str):
    """(class name, score) pairs of one detection message of any supported type."""
    if kind == "std_msgs/String":
        text = msg.data.strip()
        try:
            v = json.loads(text)
            items = v if isinstance(v, list) else [v]
            out = []
            for it in items:
                if isinstance(it, dict):
                    out.append((str(it.get("class") or it.get("class_name") or it.get("label")), float(it.get("score", 1.0))))
                else:
                    out.append((str(it), 1.0))
            return out
        except (ValueError, TypeError):
            return [(c.strip(), 1.0) for c in text.split(",") if c.strip()]
    if kind == "yolo_msgs/DetectionArray":
        return [(d.class_name, float(d.score)) for d in msg.detections]
    return [(h.hypothesis.class_id, float(h.hypothesis.score)) for d in msg.detections for h in d.results]


class ComresNode(Node):
    def __init__(self):
        super().__init__("comres_node")
        self.robots: List[str] = list(self.param("robot_ids", ["go2_1"]))
        self.world_frame = self.param("world_frame", "")          # "": one robot -> its SLAM map frame
        self.map_topic = self.param("map_topic", "/{robot}/map")
        self.base_frame_fmt = self.param("base_frame", "")         # "": the first of BASE_CANDIDATES found
        self.nav_frame_fmt = self.param("nav_frame", "")           # "": the robot's SLAM map frame
        self.nav_action = self.param("nav_action", "/{robot}/navigate_to_pose")
        self.det_topic = self.param("detection_topic", "/{robot}/detections")
        self.det_type = self.param("detection_type", "vision_msgs/Detection2DArray")
        self.map_qos = self.param("map_qos", "reliable")
        self.grid_auto_center = self.param("grid_auto_center", True)
        self.grid_xy0 = (self.param("grid_x0", -30.0), self.param("grid_y0", -30.0))
        self.grid_wh = (self.param("grid_width", 60.0), self.param("grid_height", 60.0))
        self.cell = self.param("cell_size", 0.25)
        self.grid_yaw = math.radians(self.param("grid_yaw_deg", 0.0))
        self.occ_threshold = self.param("occ_threshold", 65)
        self.free_threshold = self.param("free_threshold", 25)
        self.free_fraction = self.param("free_fraction", 0.5)
        self.task = self.param("task", "search")
        self.hint = self.param("hint", "")
        self.T_H = self.param("T_H", 100)
        self.tick_seconds = self.param("tick_seconds", 1.0)
        self.max_seconds = self.param("max_mission_seconds", 1800.0)
        self.goal_timeout = self.param("goal_timeout", 60.0)
        self.goal_clearance = self.param("goal_clearance", 0.3)
        self.arrival_hold = self.param("arrival_hold", 5)
        self.target_classes = set(self.param("target_classes", ["backpack"]))
        self.min_score = self.param("detection_min_score", 0.5)
        self.det_frames = self.param("detection_frames", 3)
        self.vlm_url = self.param("vlm_base_url", "http://127.0.0.1:18765/v1")
        self.vlm_model = self.param("vlm_model", "qwen3.8-27b")
        self.fake_vlm = self.param("fake_vlm", False)
        self.run_root = os.path.expanduser(self.param("run_dir", "~/comres_runs"))
        self.auto_start = self.param("auto_start", False)
        self.no_frontier_ticks = self.param("no_frontier_ticks", 30)
        if self.det_type not in DETECTION_TYPES:
            raise ValueError(f"detection_type must be one of {sorted(DETECTION_TYPES)}, got {self.det_type!r}")
        if len(self.robots) > 1 and not self.world_frame:
            raise ValueError("with several robots, set world_frame (a frame every robot's SLAM map frame is placed in)")
        os.environ["COMRES_VLM_BASE_URL"] = self.vlm_url
        os.environ["COMRES_VLM_MODEL"] = self.vlm_model
        os.environ["COMRES_VLM_TIMEOUT"] = str(self.param("vlm_timeout", 90.0))
        os.environ.pop("COMRES_VLM_PROVIDER", None)            # our server only
        os.environ.pop("COMRES_FAKE_VLM", None)
        if self.fake_vlm:                                       # dry runs without the server
            os.environ["COMRES_FAKE_VLM"] = "1"

        self.mission = None
        self.spec: Optional[GridSpec] = None
        self.world_from_plan = (0.0, 0.0, 0.0)
        self.run_dir: Optional[str] = None
        self.step = 0
        self.t0: Optional[float] = None
        self.finished = False
        self._n_logged = 0
        self._n_errors = 0
        self._idle_ticks = 0

        self.tf = Buffer()
        self.tf_listener = TransformListener(self.tf, self)
        if self.map_qos == "best_effort":
            map_qos = QoSProfile(depth=1, reliability=ReliabilityPolicy.BEST_EFFORT, durability=DurabilityPolicy.VOLATILE)
        else:                                                   # matches latched and volatile reliable publishers
            map_qos = QoSProfile(depth=1, reliability=ReliabilityPolicy.RELIABLE, durability=DurabilityPolicy.VOLATILE)
        mod, name = DETECTION_TYPES[self.det_type]
        det_cls = getattr(__import__(mod, fromlist=[name]), name)
        self.map_msgs: Dict[int, OccupancyGrid] = {}
        self.grids: Dict[int, np.ndarray] = {}
        self._map_key: Dict[int, tuple] = {}
        self.base_frames: Dict[int, str] = {}
        self.nav: Dict[int, ActionClient] = {}
        self.goal_handle: Dict[int, object] = {}               # the accepted goal Nav2 is running
        self.goal_seq: Dict[int, int] = {}                     # bumped by every send and retirement
        self.last_sent: Dict[int, int] = {}                    # seq of the last goal sent
        self.goal_sent: Dict[int, Optional[dict]] = {}         # the current goal: target, where, progress
        self.need_goal: Dict[int, bool] = {}
        self.last_dropped: Dict[int, Optional[tuple]] = {}
        self.det_count: Dict[int, int] = {}
        self.det_last: Dict[int, float] = {}
        self.det_seen_any = False
        self.det_seen_target = False
        for i, r in enumerate(self.robots):
            self.create_subscription(OccupancyGrid, self.map_topic.format(robot=r),
                                     lambda m, i=i: self.map_msgs.__setitem__(i, m), map_qos)
            self.create_subscription(det_cls, self.det_topic.format(robot=r),
                                     lambda m, i=i: self.on_detections(i, m), qos_profile_sensor_data)
            self.nav[i] = ActionClient(self, NavigateToPose, self.nav_action.format(robot=r))
            self.goal_seq[i] = 0
            self.last_sent[i] = -1
            self.goal_sent[i] = None
            self.need_goal[i] = False
            self.last_dropped[i] = None
            self.det_count[i] = 0
            self.det_last[i] = 0.0
        latched = QoSProfile(depth=1, reliability=ReliabilityPolicy.RELIABLE, durability=DurabilityPolicy.TRANSIENT_LOCAL)
        self.merged_pub = self.create_publisher(OccupancyGrid, "~/merged_map", latched)
        self.targets_pub = self.create_publisher(PoseArray, "~/targets", 10)
        self.create_service(Trigger, "~/start", self.on_start)
        self.create_service(Trigger, "~/stop", self.on_stop)
        self.timer = self.create_timer(self.tick_seconds, self.on_tick)
        self.get_logger().info(
            f"COMRES node: robots {self.robots}, task {self.task}, T_H {self.T_H} ticks of {self.tick_seconds} s, "
            f"cells {self.cell} m; VLM {self.vlm_model} at {self.vlm_url}" + (" [FAKE]" if self.fake_vlm else "")
            + ("" if self.auto_start else "; waiting for ~/start"))

    def param(self, name: str, default):
        """A parameter of the default's type; numbers accept ints and floats alike, booleans
        true/false (or "true"/"false"), lists a single value as a one-item list."""
        v = self.declare_parameter(name, default, ParameterDescriptor(dynamic_typing=True)).value
        try:
            if isinstance(default, list):
                return [] if v is None else [str(x) for x in v] if isinstance(v, (list, tuple)) else [str(v)]
            if v is None:
                return default
            if isinstance(default, bool):
                if isinstance(v, str) and v.strip().lower() in ("true", "false"):
                    return v.strip().lower() == "true"
                if isinstance(v, bool):
                    return v
                raise TypeError
            if isinstance(default, (int, float)):
                if isinstance(v, bool) or not isinstance(v, (int, float, str)):
                    raise TypeError
                x = float(v)
                if isinstance(default, int):
                    if x != int(x):
                        raise TypeError
                    return int(x)
                return x
            return str(v)
        except (TypeError, ValueError):
            raise ValueError(f"parameter {name}: {v!r} is not a {type(default).__name__}") from None

    # -- frames ----------------------------------------------------------------------------
    def pose2d(self, target: str, source: str):
        """The pose of frame `source` in frame `target` (x, y, yaw), or None."""
        if not target or not source:
            return None
        if target == source:
            return (0.0, 0.0, 0.0)
        try:
            t = self.tf.lookup_transform(target, source, Time())
        except Exception:                                      # noqa: BLE001
            return None
        tr = t.transform
        return (tr.translation.x, tr.translation.y, yaw_of(tr.rotation))

    def map_frame(self, i: int) -> str:
        m = self.map_msgs.get(i)
        return m.header.frame_id if m is not None else ""

    def world(self) -> str:
        return self.world_frame or self.map_frame(0)

    def nav_frame(self, i: int) -> str:
        return self.nav_frame_fmt.format(robot=self.robots[i]) if self.nav_frame_fmt else self.map_frame(i)

    def base_frame(self, i: int) -> Optional[str]:
        if i in self.base_frames:
            return self.base_frames[i]
        names = [self.base_frame_fmt] if self.base_frame_fmt else list(BASE_CANDIDATES)
        for f in names:
            f = f.format(robot=self.robots[i])
            if self.pose2d(self.world(), f) is not None:
                self.base_frames[i] = f
                return f
        return None

    def robot_pose(self, i: int):
        """Robot i's pose in the planning frame, or None."""
        b = self.base_frame(i)
        p = self.pose2d(self.world(), b) if b else None
        return None if p is None else compose(invert(self.world_from_plan), p)

    def update_grid(self, i: int) -> bool:
        m = self.map_msgs.get(i)
        if m is None or self.spec is None:
            return i in self.grids
        world_from_map = self.pose2d(self.world(), m.header.frame_id)
        if world_from_map is None:
            return i in self.grids
        plan_from_map = compose(invert(self.world_from_plan), world_from_map)
        pose_key = tuple(round(v, 3) for v in plan_from_map)
        last = self._map_key.get(i)
        if last is not None and last[0] is m and last[1] == pose_key:   # the same message, the same place
            return True
        o = m.info.origin
        data = np.asarray(m.data, dtype=np.int16).reshape(m.info.height, m.info.width)
        self.grids[i] = occupancy_to_grid(data, m.info.resolution, (o.position.x, o.position.y, yaw_of(o.orientation)),
                                          plan_from_map, self.spec, self.occ_threshold, self.free_fraction,
                                          free_threshold=self.free_threshold)
        self._map_key[i] = (m, pose_key)                         # holds the message: identity stays valid
        self.warn_outside(i, data, m, plan_from_map)
        return True

    def warn_outside(self, i: int, data, m, plan_from_map) -> None:
        """Warn when the robot's SLAM map knows space outside the planning grid (it is dropped)."""
        known = np.argwhere(data >= 0)
        if not len(known):
            return
        j0, i0 = known.min(axis=0)
        j1, i1 = known.max(axis=0) + 1
        o = m.info.origin
        T = compose(plan_from_map, (o.position.x, o.position.y, yaw_of(o.orientation)))
        res = m.info.resolution
        for (a, b) in ((i0, j0), (i1, j0), (i0, j1), (i1, j1)):
            x, y, _ = compose(T, (a * res, b * res, 0.0))
            if self.spec.cell_of(x, y) is None:
                self.get_logger().warning(f"{self.robots[i]}'s SLAM map reaches outside the planning grid "
                                       f"({x:.1f}, {y:.1f}); widen grid_width/grid_height",
                                       throttle_duration_sec=30.0)
                return

    def check(self, need_grid: bool = True) -> List[str]:
        """What is missing for each robot (empty: ready)."""
        missing = []
        for i, r in enumerate(self.robots):
            if self.map_msgs.get(i) is None:
                missing.append(f"{r}: no map on {self.map_topic.format(robot=r)}")
                continue
            if self.pose2d(self.world(), self.map_frame(i)) is None:
                missing.append(f"{r}: no TF {self.world()} <- {self.map_frame(i)}")
            if self.base_frame(i) is None:
                tried = self.base_frame_fmt or ", ".join(BASE_CANDIDATES)
                missing.append(f"{r}: no TF {self.world()} <- the robot base (tried {tried.format(robot=r)})")
            if self.pose2d(self.nav_frame(i), self.world()) is None:
                missing.append(f"{r}: no TF {self.nav_frame(i)} <- {self.world()} (goal frame)")
            if not self.nav[i].server_is_ready():
                missing.append(f"{r}: Nav2 action {self.nav_action.format(robot=r)} not available")
            if need_grid and self.spec is not None:
                p = self.robot_pose(i)
                if p is not None and self.spec.cell_of(p[0], p[1]) is None:
                    missing.append(f"{r}: outside the planning grid at ({p[0]:.1f}, {p[1]:.1f})")
        return missing

    def vlm_ok(self) -> Optional[str]:
        """None if the VLM server answers with our model, else why not."""
        if self.fake_vlm:
            return None
        try:
            import openai  # noqa: F401
        except ImportError:
            return ("the python package openai is missing (pip install --user --break-system-packages openai, "
                    "or a venv made with --system-site-packages)")
        try:
            with urllib.request.urlopen(self.vlm_url.rstrip("/") + "/models", timeout=5) as resp:
                ids = [m.get("id") for m in json.loads(resp.read()).get("data", [])]
        except Exception as e:                                 # noqa: BLE001
            return f"VLM server {self.vlm_url} not reachable: {e}"
        return None if self.vlm_model in ids else f"VLM server serves {ids}, not {self.vlm_model}"

    # -- start / stop ----------------------------------------------------------------------
    def start(self) -> str:
        if self.finished:
            return "mission over; restart the node for a new one"
        if self.mission is not None:
            return "already running"
        missing = self.check(need_grid=False)
        if missing:
            return "not ready: " + "; ".join(missing)
        why = self.vlm_ok()
        if why:
            return "not ready: " + why
        starts = [self.pose2d(self.world(), self.base_frame(i)) for i in range(len(self.robots))]
        if self.grid_auto_center:                              # the grid centred on the team's start
            cx, cy = float(np.mean([p[0] for p in starts])), float(np.mean([p[1] for p in starts]))
            self.world_from_plan = (cx, cy, self.grid_yaw)
            self.spec = GridSpec.from_extent(-self.grid_wh[0] / 2, -self.grid_wh[1] / 2, *self.grid_wh, self.cell)
        else:
            self.world_from_plan = (0.0, 0.0, self.grid_yaw)
            self.spec = GridSpec.from_extent(*self.grid_xy0, *self.grid_wh, self.cell)
        cells = []
        for i in range(len(self.robots)):
            self.update_grid(i)
            p = self.robot_pose(i)
            c = self.spec.cell_of(p[0], p[1])
            if c is None:
                self.spec = None
                return f"not ready: {self.robots[i]} is outside the planning grid"
            cells.append((c, p))
        from params import Params                               # the algorithm's (comres_core put it on sys.path)
        from comres_core.online import Mission
        self.mission = Mission(self.robots, self.spec.shape, task=self.task, hint=self.hint,
                               params=Params(T_H=self.T_H, dt=self.tick_seconds), arrival_hold=self.arrival_hold)
        self.mission.start([c for c, _ in cells])
        self.t0 = time.monotonic()
        self.step = 0
        self.run_dir = os.path.join(self.run_root, time.strftime("%Y%m%d_%H%M%S"))
        for d in ("maps", "plans"):
            os.makedirs(os.path.join(self.run_dir, d), exist_ok=True)
        os.environ["COMRES_VLM_LOCK"] = os.path.join(self.run_dir, "vlm.lock")
        meta = {"robots": self.robots, "task": self.task, "hint": self.hint, "T_H": self.T_H,
                "tick_seconds": self.tick_seconds, "grid": vars(self.spec), "world_frame": self.world(),
                "world_from_plan": list(self.world_from_plan), "base_frames": self.base_frames,
                "nav_frames": [self.nav_frame(i) for i in range(len(self.robots))],
                "map_frames": [self.map_frame(i) for i in range(len(self.robots))],
                "start_cells": [list(c) for c, _ in cells], "start_poses_plan": [list(p) for _, p in cells],
                "target_classes": sorted(self.target_classes), "detection_type": self.det_type,
                "vlm_model": self.vlm_model, "vlm_base_url": self.vlm_url, "fake_vlm": self.fake_vlm,
                "arrival_hold": self.arrival_hold, "goal_timeout": self.goal_timeout,
                "core_version": core_version(), "wall_clock_start": time.time()}
        with open(os.path.join(self.run_dir, "meta.json"), "w") as f:
            json.dump(meta, f, indent=1)
        self.get_logger().info(f"mission started: grid {self.spec.shape} x {self.cell} m, logging to {self.run_dir}")
        return "started"

    def on_start(self, req, res):
        msg = self.start()
        res.success, res.message = msg == "started", msg
        if res.success:
            self.get_logger().info(msg)
        else:
            self.get_logger().warning(msg)
        return res

    def on_stop(self, req, res):
        if self.mission is None:
            self.auto_start = False                            # nothing runs; nor will it start by itself
            self.finished = True
            res.success, res.message = True, "no mission was running; this node will not start one"
        elif self.finished:
            res.success, res.message = True, "the mission had already ended"
        else:
            self.finish("stopped by operator")
            res.success, res.message = True, "stopped"
        self.get_logger().info(f"~/stop: {res.message}")
        return res

    def finish(self, why: str, **kw) -> None:
        if self.finished or self.mission is None:
            return
        self.finished = True
        for i in range(len(self.robots)):
            self.cancel(i)
        elapsed = time.monotonic() - self.t0 if self.t0 is not None else None
        self.flush_events()
        self.log("events", {"kind": "end", "why": why, "elapsed_s": elapsed, **kw})
        self.mission.stop()
        self.get_logger().info(f"mission ended after {elapsed:.1f} s: {why}")

    # -- the tick --------------------------------------------------------------------------
    def on_tick(self) -> None:
        try:
            self._tick()
        except Exception as e:                                 # noqa: BLE001
            import traceback
            self._n_errors += 1
            self.get_logger().error(f"tick failed ({self._n_errors}): {e}\n{traceback.format_exc()[-1500:]}")
            self.log("events", {"kind": "node_error", "error": traceback.format_exc()[-1500:]})
            if self._n_errors >= 5:
                self.finish("too many errors")

    def _tick(self) -> None:
        if self.mission is None:
            if self.auto_start and not self.finished:
                msg = self.start()
                if msg != "started":
                    self.get_logger().warning(msg, throttle_duration_sec=5.0)
            return
        if self.finished:
            return
        elapsed = time.monotonic() - self.t0
        if elapsed > self.max_seconds:
            self.finish("time limit")
            return
        cells = []
        for i in range(len(self.robots)):
            ok = self.update_grid(i)
            p = self.robot_pose(i)
            c = self.spec.cell_of(p[0], p[1]) if p is not None else None
            if not ok or c is None:
                self.get_logger().warning("holding this tick: " + "; ".join(self.check() or [f"{self.robots[i]}: no pose"]),
                                       throttle_duration_sec=5.0)
                return
            cells.append((c, p))
        for i, (c, _) in enumerate(cells):
            self.mission.update_robot(i, self.grids[i], c)
        changed = self.mission.tick(self.step)
        for i, target in changed.items():
            if target is None:
                self.cancel(i)                                 # stop: nothing to go to
            else:
                self.retire(i)                                 # the old goal's outcome no longer counts
                self.need_goal[i] = True
        for i, (_, p) in enumerate(cells):
            r = self.mission.robots[i]
            if self.need_goal[i] and r.target is not None:
                self.send_goal(i, r.target, p)
            self.watch_goal(i, p)
        self.flush_events(elapsed)
        if self.task == "explore":                             # explored: no frontier left for anyone
            idle = all(r.target is None for r in self.mission.robots) and not self.mission.strategy.in_flight()
            self._idle_ticks = self._idle_ticks + 1 if idle else 0
            if self.no_frontier_ticks > 0 and self._idle_ticks >= self.no_frontier_ticks:
                self.finish("no frontier left")
                return
        self.log("poses", {"step": self.step, "elapsed_s": round(elapsed, 3),
                           "cells": [list(c) for c, _ in cells], "poses_plan": [list(map(float, p)) for _, p in cells],
                           "targets": [list(r.target) if r.target else None for r in self.mission.robots]})
        self.publish_targets()
        self.step += 1

    def flush_events(self, elapsed: Optional[float] = None) -> None:
        """Write every Mission event not yet logged (tick and goal callbacks alike) and act on
        merges and plans."""
        if self.mission is None:
            return
        for e in self.mission.events[self._n_logged:]:
            self.log("events", {**e, **({"elapsed_s": round(elapsed, 3)} if elapsed is not None else {})})
            k = e["kind"]
            if k == "merge" and self.mission.merged is not None:
                np.savez_compressed(os.path.join(self.run_dir, "maps", f"merge_{e['step']:05d}.npz"), occ=self.mission.merged)
                self.publish_merged()
            elif k == "plan":
                rec = next((r for r in reversed(self.mission.strategy.plan_records)
                            if r.get("query_step") == e.get("query_step")), None)
                if rec is not None:
                    self.log("plans", {key: v for key, v in rec.items() if not str(key).startswith("_")})
                    self.save_vlm_input(e.get("query_step"), rec.get("_messages"))
                if e.get("valid"):
                    self.get_logger().info(f"plan (asked at step {e.get('query_step')}, {e.get('answer_seconds')} s): "
                                           f"pieces {e.get('plan')}, guesses {e.get('hyps')}")
                else:
                    self.get_logger().warning(f"plan asked at step {e.get('query_step')} unusable: {e.get('error')}")
            elif k == "plan_error":
                self.get_logger().error(f"the planner failed (asked at step {e.get('query_step')}): {e.get('error', '')[-400:]}")
            elif k in ("failed", "stuck_frontier"):
                self.get_logger().warning(f"{e.get('robot')}: {k} {e.get('target')}")
        self._n_logged = len(self.mission.events)

    def save_vlm_input(self, step, msgs) -> None:
        """The picture and text the VLM was given for the plan asked at `step`."""
        if not msgs:
            return
        name = os.path.join(self.run_dir, "plans", f"plan_{int(step or 0):05d}")
        texts = []
        for m in msgs:
            for part in (m["content"] if isinstance(m["content"], list) else [{"type": "text", "text": m["content"]}]):
                if part["type"] == "text":
                    texts.append(part["text"])
                elif part["type"] == "image_url":
                    with open(name + ".png", "wb") as f:
                        f.write(base64.b64decode(part["image_url"]["url"].split(",", 1)[1]))
        with open(name + ".txt", "w") as f:
            f.write("\n\n".join(texts))

    # -- Nav2 ------------------------------------------------------------------------------
    def goal_cell(self, i: int, cell):
        """Where Nav2 is sent for target `cell`: the cell itself when its centre is at least
        goal_clearance from every wall on the robot's map, else the nearest free cell within 0.5 m
        that is, else the free cell there farthest from walls (Nav2 refuses goals inside its
        obstacle margin; COMRES's frontier points often touch a wall). COMRES itself keeps the
        target cell (seen / arrival tests)."""
        k = self.goal_clearance / self.cell
        if k <= 0:
            return tuple(cell)
        g = self.mission.robots[i].local_map
        clear = ndimage.distance_transform_edt(g != 1) - 0.71    # centre to the nearest wall's edge, at worst
        r, c = cell
        if clear[r, c] >= k:
            return tuple(cell)
        R = 0.5 / self.cell
        best, key = tuple(cell), None
        for dr in range(-int(R), int(R) + 1):
            for dc in range(-int(R), int(R) + 1):
                rr, cc = r + dr, c + dc
                if dr * dr + dc * dc <= R * R and 0 <= rr < g.shape[0] and 0 <= cc < g.shape[1] and g[rr, cc] == 0:
                    ok = clear[rr, cc] >= k
                    kk = (not ok, dr * dr + dc * dc if ok else -clear[rr, cc])
                    if key is None or kk < key:
                        best, key = (rr, cc), kk
        return best

    def send_goal(self, i: int, cell, pose) -> None:
        """Send robot i's target to its Nav2; Nav2 replaces the robot's previous goal. If the goal
        frame or Nav2 is not available, the old goal is stopped and the target sent next tick."""
        r = self.robots[i]
        gc = self.goal_cell(i, cell)
        x, y = self.spec.center(gc)
        yaw = math.atan2(y - pose[1], x - pose[0])
        nav_frame = self.nav_frame(i)
        nav_from_world = self.pose2d(nav_frame, self.world())
        if nav_from_world is None or not self.nav[i].server_is_ready():
            gh = self.goal_handle.pop(i, None)
            if gh is not None:                                 # it would keep driving to the old target
                gh.cancel_goal_async()
            self.get_logger().warning(f"{r}: cannot send its goal yet ("
                                      + ("no TF " + nav_frame + " <- " + self.world() if nav_from_world is None
                                         else "Nav2 not available") + "); retrying", throttle_duration_sec=5.0)
            return
        gx, gy, gyaw = compose(nav_from_world, compose(self.world_from_plan, (x, y, yaw)))
        goal = NavigateToPose.Goal()
        goal.pose = PoseStamped()
        goal.pose.header.frame_id = nav_frame                  # stamp 0: the latest transform
        goal.pose.pose.position.x, goal.pose.pose.position.y = float(gx), float(gy)
        o = goal.pose.pose.orientation
        o.x, o.y, o.z, o.w = quat_of(gyaw)
        self.goal_seq[i] += 1
        seq = self.last_sent[i] = self.goal_seq[i]
        self.goal_handle.pop(i, None)                          # Nav2 preempts it
        now = time.monotonic()
        self.goal_sent[i] = {"seq": seq, "cell": tuple(cell), "xy": (x, y), "t_sent": now, "t_best": now,
                             "line": math.dist((x, y), pose[:2]), "path": math.inf}
        self.need_goal[i] = False
        fut = self.nav[i].send_goal_async(goal, feedback_callback=lambda fb, i=i, seq=seq: self.on_feedback(i, seq, fb))
        fut.add_done_callback(lambda f, i=i, seq=seq: self.on_goal_response(i, seq, f))
        self.log("events", {"kind": "goal", "robot": r, "step": self.step, "cell": list(cell),
                            "goal_cell": list(gc), "plan_xy": [x, y], "nav": [gx, gy, gyaw], "frame": nav_frame})

    def on_feedback(self, i: int, seq: int, fb) -> None:
        """Nav2's remaining path length: progress along the path, even around walls."""
        g = self.goal_sent[i]
        rem = float(getattr(fb.feedback, "distance_remaining", 0.0) or 0.0)
        if g is None or g["seq"] != seq or rem <= 0.0:
            return
        if rem < g["path"] - 0.5 * self.cell:
            g["path"], g["t_best"] = rem, time.monotonic()

    def watch_goal(self, i: int, pose) -> None:
        """A goal that makes no progress (neither Nav2's remaining path nor the straight-line
        distance shrinks) for goal_timeout seconds is given up as unreachable."""
        g = self.goal_sent[i]
        r = self.mission.robots[i]
        if g is None or r.target is None or tuple(r.target) != g["cell"]:
            return
        d = math.dist(g["xy"], pose[:2])
        now = time.monotonic()
        if d < g["line"] - 0.5 * self.cell:
            g["line"], g["t_best"] = d, now
        elif now - g["t_best"] > self.goal_timeout:
            self.get_logger().warning(f"{self.robots[i]}: no progress toward {list(g['cell'])} for "
                                      f"{self.goal_timeout:.0f} s; giving it up")
            cell = g["cell"]
            self.cancel(i)
            self.mission.report(i, "failed", cell)

    def on_goal_response(self, i: int, seq: int, fut) -> None:
        gh = fut.result()
        accepted = gh is not None and gh.accepted
        if seq != self.goal_seq[i] or self.finished:
            if accepted and seq == self.last_sent[i]:          # retired while Nav2 was answering, and no
                gh.cancel_goal_async()                         # newer goal will preempt it: stop it
            return
        g = self.goal_sent[i]
        if not accepted:
            self.get_logger().warning(f"{self.robots[i]}: Nav2 rejected the goal")
            self.goal_sent[i] = None
            self.mission.report(i, "failed", g["cell"] if g else None)
            return
        self.goal_handle[i] = gh
        gh.get_result_async().add_done_callback(lambda f, i=i, seq=seq: self.on_goal_result(i, seq, f))

    def on_goal_result(self, i: int, seq: int, fut) -> None:
        if seq != self.goal_seq[i] or self.finished:
            return
        res = fut.result()
        status = res.status
        code = getattr(res.result, "error_code", None)
        g = self.goal_sent[i]
        cell = g["cell"] if g else None
        self.goal_handle.pop(i, None)
        self.goal_sent[i] = None
        if status == GoalStatus.STATUS_SUCCEEDED:
            self.last_dropped[i] = None
            self.mission.report(i, "reached", cell)
        elif status == GoalStatus.STATUS_ABORTED and not (code == 0 and self.last_dropped[i] != cell):
            self.get_logger().warning(f"{self.robots[i]}: Nav2 aborted the goal (error {code})")
            self.mission.report(i, "failed", cell)
        else:                                                  # cancelled or preempted outside COMRES: choose
            self.get_logger().warning(f"{self.robots[i]}: the goal ended with status {status} (error {code}); "
                                      "choosing again")        # again (a second abort without an error: failed)
            self.last_dropped[i] = cell
            self.mission.report(i, "dropped", cell)

    def retire(self, i: int) -> None:
        """The current goal no longer counts: its late responses and results are ignored."""
        self.goal_seq[i] += 1
        self.goal_sent[i] = None
        self.need_goal[i] = False

    def cancel(self, i: int) -> None:
        """Retire the current goal and stop the robot (a goal Nav2 has not accepted yet is
        cancelled when its acceptance arrives)."""
        self.retire(i)
        gh = self.goal_handle.pop(i, None)
        if gh is not None:
            gh.cancel_goal_async()

    # -- detection -------------------------------------------------------------------------
    def on_detections(self, i: int, msg) -> None:
        if self.mission is None or self.finished:
            return
        dets = detections_of(msg, self.det_type)
        if dets and not self.det_seen_any:
            self.det_seen_any = True
            self.get_logger().info(f"first detections from {self.robots[i]}: {dets[:5]}")
        best = max(((c, s) for c, s in dets if c in self.target_classes and s >= self.min_score),
                   key=lambda cs: cs[1], default=None)
        now = time.monotonic()
        if best is None:
            if now - self.det_last[i] > 1.0:                  # no sighting for a second: start over
                self.det_count[i] = 0
            return
        self.det_count[i] = self.det_count[i] + 1 if now - self.det_last[i] <= 1.0 else 1
        self.det_last[i] = now
        if self.det_count[i] >= self.det_frames:
            p = self.robot_pose(i)
            if self.task == "search":
                self.finish("target found", robot=self.robots[i], cls=best[0], score=best[1],
                            pose_plan=list(p) if p else None)
            elif not self.det_seen_target:                     # exploring: note it, keep going
                self.det_seen_target = True
                self.log("events", {"kind": "target_seen", "robot": self.robots[i], "step": self.step,
                                    "cls": best[0], "score": best[1], "pose_plan": list(p) if p else None})
                self.get_logger().info(f"{self.robots[i]} sees a {best[0]} (exploring: not ending)")

    # -- output ----------------------------------------------------------------------------
    def log(self, name: str, rec: Dict) -> None:
        if self.run_dir is None:
            return
        with open(os.path.join(self.run_dir, f"{name}.jsonl"), "a") as f:
            f.write(json.dumps(_jsonable(rec)) + "\n")

    def publish_merged(self) -> None:
        g = self.mission.merged
        m = OccupancyGrid()
        m.header.frame_id = self.world()
        m.header.stamp = self.get_clock().now().to_msg()
        m.info.resolution = float(self.spec.cell)
        m.info.width, m.info.height = int(self.spec.cols), int(self.spec.rows)
        ox, oy, oyaw = compose(self.world_from_plan, (self.spec.x0, self.spec.y0, 0.0))
        m.info.origin.position.x, m.info.origin.position.y = float(ox), float(oy)
        o = m.info.origin.orientation
        o.x, o.y, o.z, o.w = quat_of(oyaw)
        out = np.where(g == 1, 100, np.where(g == 0, 0, -1)).astype(np.int8)
        m.data = out.ravel().tolist()                          # row-major, row = y: the grid's own order
        self.merged_pub.publish(m)

    def publish_targets(self) -> None:
        pa = PoseArray()
        pa.header.frame_id = self.world()
        pa.header.stamp = self.get_clock().now().to_msg()
        for r in self.mission.robots:
            if r.target is not None:
                x, y, _ = compose(self.world_from_plan, (*self.spec.center(r.target), 0.0))
                p = Pose()
                p.position.x, p.position.y = float(x), float(y)
                p.orientation.w = 1.0
                pa.poses.append(p)
        self.targets_pub.publish(pa)


def main(args=None):
    from rclpy.executors import ExternalShutdownException
    rclpy.init(args=args)
    node = ComresNode()
    try:
        rclpy.spin(node)
    except (KeyboardInterrupt, ExternalShutdownException):
        pass
    finally:
        node.finish("node shut down")
        node.destroy_node()
        rclpy.try_shutdown()
