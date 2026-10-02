"""
The central COMRES node (ROS 2 Jazzy). It runs comres_core.online.Mission in real time for a team
of robots that each run their own SLAM, Nav2 and a target detector:

  in   /<robot>/map              nav_msgs/OccupancyGrid  each robot's live SLAM map (any frame;
                                 TF to the world frame must exist)
       TF world <- <robot> base  each robot's pose
       /<robot>/detections       vision_msgs/Detection2DArray from the robot's own detector (the
                                 YOLO on each Go2; class name in results[].hypothesis.class_id)
  out  /<robot>/navigate_to_pose nav2_msgs/action/NavigateToPose, one goal at a time per robot
       ~/merged_map              nav_msgs/OccupancyGrid (world frame), the last merged map
       ~/targets                 geometry_msgs/PoseArray (world frame), the robots' targets
  srv  ~/start, ~/stop           std_srvs/Trigger

Every tick (tick_seconds) it converts each robot's SLAM map onto the planning grid
(comres_core.grid), reads the robots' cells and calls Mission.tick: every T_H ticks the maps are
merged and the VLM is asked (in the background); robots whose target was seen, reached or failed
get a new one from the COMRES selector, sent to Nav2. The search ends when a detector reports a
target class for detection_frames frames in a row, or on ~/stop, or after max_mission_seconds.

The VLM is our server (serve_vlm.sh in COMRES-VLM) at vlm_base_url; no key is read from files.
Everything is logged under run_dir/<start time>/: meta.json, events.jsonl, poses.jsonl,
plans.jsonl and the merged map of every merge (maps/merge_<step>.npz).
"""

import json
import math
import os
import time
from typing import Dict, List, Optional

import numpy as np
import rclpy
from action_msgs.msg import GoalStatus
from geometry_msgs.msg import Pose, PoseArray, PoseStamped
from nav2_msgs.action import NavigateToPose
from nav_msgs.msg import OccupancyGrid
from rclpy.action import ActionClient
from rclpy.node import Node
from rclpy.qos import DurabilityPolicy, QoSProfile, ReliabilityPolicy
from rclpy.time import Time
from std_srvs.srv import Trigger
from tf2_ros import Buffer, TransformListener
from vision_msgs.msg import Detection2DArray

from comres_core import core_version
from comres_core.grid import GridSpec, compose, occupancy_to_grid
from comres_core.online import Mission


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
    if isinstance(v, (np.integer,)):
        return int(v)
    if isinstance(v, (np.floating,)):
        return float(v)
    return v


class ComresNode(Node):
    def __init__(self):
        super().__init__("comres_node")
        d = self.declare_parameter
        self.robots: List[str] = list(d("robot_ids", ["go2_1", "go2_2"]).value)
        self.world_frame = d("world_frame", "map").value
        self.map_topic = d("map_topic", "/{robot}/map").value
        self.base_frame = d("base_frame", "{robot}/base_link").value
        self.nav_frame = d("nav_frame", "{robot}/map").value
        self.nav_action = d("nav_action", "/{robot}/navigate_to_pose").value
        self.det_topic = d("detection_topic", "/{robot}/detections").value
        self.spec = GridSpec.from_extent(d("grid_x0", -20.0).value, d("grid_y0", -20.0).value,
                                         d("grid_width", 40.0).value, d("grid_height", 40.0).value,
                                         d("cell_size", 0.25).value)
        self.occ_threshold = int(d("occ_threshold", 65).value)
        self.free_fraction = float(d("free_fraction", 0.5).value)
        self.task = d("task", "search").value
        self.hint = d("hint", "").value
        self.T_H = int(d("T_H", 100).value)
        self.tick_seconds = float(d("tick_seconds", 1.0).value)
        self.max_seconds = float(d("max_mission_seconds", 1800.0).value)
        self.target_classes = set(d("target_classes", ["backpack"]).value)
        self.min_score = float(d("detection_min_score", 0.5).value)
        self.det_frames = int(d("detection_frames", 3).value)
        run_root = os.path.expanduser(d("run_dir", "~/comres_runs").value)
        os.environ["COMRES_VLM_BASE_URL"] = d("vlm_base_url", "http://127.0.0.1:18765/v1").value
        os.environ["COMRES_VLM_MODEL"] = d("vlm_model", "qwen3.8-27b").value
        os.environ.pop("COMRES_VLM_PROVIDER", None)            # our server only
        os.environ.pop("COMRES_FAKE_VLM", None)
        if d("fake_vlm", False).value:                          # dry runs without the server
            os.environ["COMRES_FAKE_VLM"] = "1"
        auto_start = bool(d("auto_start", False).value)

        from params import Params                               # the algorithm's (comres_core puts it on sys.path)
        self.mission = Mission(self.robots, self.spec.shape, task=self.task, hint=self.hint,
                               params=Params(T_H=self.T_H, dt=self.tick_seconds))
        self.run_root = run_root
        self.run_dir: Optional[str] = None
        self.step = 0
        self.t0: Optional[float] = None
        self.finished = False

        self.tf = Buffer()
        self.tf_listener = TransformListener(self.tf, self)
        latched = QoSProfile(depth=1, reliability=ReliabilityPolicy.RELIABLE, durability=DurabilityPolicy.TRANSIENT_LOCAL)
        self.map_msgs: Dict[int, OccupancyGrid] = {}
        self.grids: Dict[int, np.ndarray] = {}
        self._map_stamp: Dict[int, tuple] = {}
        self.nav: Dict[int, ActionClient] = {}
        self.goal_handle: Dict[int, object] = {}
        self.goal_seq: Dict[int, int] = {}
        self.det_count: Dict[int, int] = {}
        for i, r in enumerate(self.robots):
            self.create_subscription(OccupancyGrid, self.map_topic.format(robot=r),
                                     lambda m, i=i: self.map_msgs.__setitem__(i, m), latched)
            self.create_subscription(Detection2DArray, self.det_topic.format(robot=r),
                                     lambda m, i=i: self.on_detections(i, m), 10)
            self.nav[i] = ActionClient(self, NavigateToPose, self.nav_action.format(robot=r))
            self.goal_seq[i] = 0
            self.det_count[i] = 0
        self.merged_pub = self.create_publisher(OccupancyGrid, "~/merged_map", latched)
        self.targets_pub = self.create_publisher(PoseArray, "~/targets", 10)
        self.create_service(Trigger, "~/start", self.on_start)
        self.create_service(Trigger, "~/stop", self.on_stop)
        self.timer = self.create_timer(self.tick_seconds, self.on_tick)
        self._auto_start = auto_start
        self.get_logger().info(f"COMRES node: robots {self.robots}, grid {self.spec.shape} x {self.spec.cell} m, "
                               f"task {self.task}, T_H {self.T_H} ticks of {self.tick_seconds} s; "
                               f"VLM {os.environ['COMRES_VLM_MODEL']} at {os.environ['COMRES_VLM_BASE_URL']}"
                               + (" [FAKE]" if os.environ.get("COMRES_FAKE_VLM") else "")
                               + ("; waiting for ~/start" if not auto_start else ""))

    # -- frames ----------------------------------------------------------------------------
    def pose2d(self, target: str, source: str):
        """The pose of frame `source` in frame `target` (x, y, yaw), or None."""
        if target == source:
            return (0.0, 0.0, 0.0)
        try:
            t = self.tf.lookup_transform(target, source, Time())
        except Exception:
            return None
        tr = t.transform
        return (tr.translation.x, tr.translation.y, yaw_of(tr.rotation))

    def robot_pose(self, i: int):
        return self.pose2d(self.world_frame, self.base_frame.format(robot=self.robots[i]))

    def update_grid(self, i: int) -> bool:
        m = self.map_msgs.get(i)
        if m is None:
            return i in self.grids
        stamp = (m.header.stamp.sec, m.header.stamp.nanosec, m.info.width, m.info.height)
        world_from_map = self.pose2d(self.world_frame, m.header.frame_id)
        if world_from_map is None:
            return i in self.grids
        if self._map_stamp.get(i) == stamp and i in self.grids:
            return True
        o = m.info.origin
        data = np.asarray(m.data, dtype=np.int16).reshape(m.info.height, m.info.width)
        self.grids[i] = occupancy_to_grid(data, m.info.resolution, (o.position.x, o.position.y, yaw_of(o.orientation)),
                                          world_from_map, self.spec, self.occ_threshold, self.free_fraction)
        self._map_stamp[i] = stamp
        return True

    def ready(self):
        """Every robot's grid and cell, or None if any is missing."""
        cells = []
        for i in range(len(self.robots)):
            p = self.robot_pose(i)
            if p is None or not self.update_grid(i):
                return None
            c = self.spec.cell_of(p[0], p[1])
            if c is None:
                self.get_logger().warn(f"{self.robots[i]} is outside the planning grid at ({p[0]:.2f}, {p[1]:.2f})",
                                       throttle_duration_sec=10.0)
                return None
            cells.append((c, p))
        return cells

    # -- start / stop ----------------------------------------------------------------------
    def start(self) -> str:
        if self.mission.started:
            return "already running"
        cells = self.ready()
        if cells is None:
            return "not ready: waiting for every robot's map and pose"
        self.mission.start([c for c, _ in cells])
        self.t0 = time.monotonic()
        self.step = 0
        self.run_dir = os.path.join(self.run_root, time.strftime("%Y%m%d_%H%M%S"))
        os.makedirs(os.path.join(self.run_dir, "maps"), exist_ok=True)
        os.environ["COMRES_VLM_LOCK"] = os.path.join(self.run_dir, "vlm.lock")
        meta = {"robots": self.robots, "task": self.task, "hint": self.hint, "T_H": self.T_H,
                "tick_seconds": self.tick_seconds, "grid": vars(self.spec), "world_frame": self.world_frame,
                "start_cells": [list(c) for c, _ in cells], "start_poses": [list(p) for _, p in cells],
                "target_classes": sorted(self.target_classes), "vlm_model": os.environ["COMRES_VLM_MODEL"],
                "vlm_base_url": os.environ["COMRES_VLM_BASE_URL"], "fake_vlm": bool(os.environ.get("COMRES_FAKE_VLM")),
                "core_version": core_version(), "wall_clock_start": time.time()}
        with open(os.path.join(self.run_dir, "meta.json"), "w") as f:
            json.dump(meta, f, indent=1)
        self.get_logger().info(f"mission started; logging to {self.run_dir}")
        return "started"

    def on_start(self, req, res):
        msg = self.start()
        res.success, res.message = msg == "started", msg
        return res

    def on_stop(self, req, res):
        self.finish("stopped by operator")
        res.success, res.message = True, "stopped"
        return res

    def finish(self, why: str, **kw) -> None:
        if self.finished or not self.mission.started:
            return
        self.finished = True
        for i in range(len(self.robots)):
            self.cancel(i)
        elapsed = time.monotonic() - self.t0 if self.t0 is not None else None
        self.log("events", {"kind": "end", "why": why, "elapsed_s": elapsed, **kw})
        self.mission.stop()
        self.get_logger().info(f"mission ended after {elapsed:.1f} s: {why}")

    # -- the tick --------------------------------------------------------------------------
    def on_tick(self) -> None:
        if not self.mission.started:
            if self._auto_start:
                self.start()
            return
        if self.finished:
            return
        elapsed = time.monotonic() - self.t0
        if elapsed > self.max_seconds:
            self.finish("time limit")
            return
        cells = self.ready()
        if cells is None:
            self.get_logger().warn("a robot's map or pose is missing; holding this tick", throttle_duration_sec=5.0)
            return
        for i, (c, _) in enumerate(cells):
            self.mission.update_robot(i, self.grids[i], c)
        n_events = len(self.mission.events)
        changed = self.mission.tick(self.step)
        for e in self.mission.events[n_events:]:
            self.log("events", {**e, "elapsed_s": round(elapsed, 3)})
            if e["kind"] == "merge" and self.mission.merged is not None:
                np.savez_compressed(os.path.join(self.run_dir, "maps", f"merge_{self.step:05d}.npz"), occ=self.mission.merged)
                self.publish_merged()
            if e["kind"] == "plan":
                self.log("plans", _jsonable(self.mission.strategy.plan_records[-1]))
        for i, target in changed.items():
            self.cancel(i)
            if target is not None:
                self.send_goal(i, target, cells[i][1])
        self.log("poses", {"step": self.step, "elapsed_s": round(elapsed, 3),
                           "cells": [list(c) for c, _ in cells], "poses": [list(map(float, p)) for _, p in cells],
                           "targets": [list(r.target) if r.target else None for r in self.mission.robots]})
        self.publish_targets()
        self.step += 1

    # -- Nav2 ------------------------------------------------------------------------------
    def send_goal(self, i: int, cell, pose) -> None:
        r = self.robots[i]
        x, y = self.spec.center(cell)
        yaw = math.atan2(y - pose[1], x - pose[0])
        nav_frame = self.nav_frame.format(robot=r)
        nav_from_world = self.pose2d(nav_frame, self.world_frame)
        if nav_from_world is None:
            self.get_logger().warn(f"no transform {self.world_frame} -> {nav_frame}; {r} gets no goal")
            self.mission.report(i, "failed")
            return
        gx, gy, gyaw = compose(nav_from_world, (x, y, yaw))
        goal = NavigateToPose.Goal()
        goal.pose = PoseStamped()
        goal.pose.header.frame_id = nav_frame
        goal.pose.header.stamp = self.get_clock().now().to_msg()
        goal.pose.pose.position.x, goal.pose.pose.position.y = float(gx), float(gy)
        q = quat_of(gyaw)
        o = goal.pose.pose.orientation
        o.x, o.y, o.z, o.w = q
        if not self.nav[i].server_is_ready():
            self.get_logger().warn(f"{r}: Nav2 is not ready; the goal is reported failed")
            self.mission.report(i, "failed")
            return
        self.goal_seq[i] += 1
        seq = self.goal_seq[i]
        fut = self.nav[i].send_goal_async(goal)
        fut.add_done_callback(lambda f, i=i, seq=seq: self.on_goal_response(i, seq, f))
        self.log("events", {"kind": "goal", "robot": r, "step": self.step, "cell": list(cell),
                            "world": [x, y], "nav": [gx, gy, gyaw]})

    def on_goal_response(self, i: int, seq: int, fut) -> None:
        gh = fut.result()
        if seq != self.goal_seq[i]:                 # superseded meanwhile
            if gh is not None and gh.accepted:
                gh.cancel_goal_async()
            return
        if gh is None or not gh.accepted:
            self.mission.report(i, "failed")
            return
        self.goal_handle[i] = gh
        gh.get_result_async().add_done_callback(lambda f, i=i, seq=seq: self.on_goal_result(i, seq, f))

    def on_goal_result(self, i: int, seq: int, fut) -> None:
        if seq != self.goal_seq[i] or self.finished:
            return
        status = fut.result().status
        self.goal_handle.pop(i, None)
        if status == GoalStatus.STATUS_SUCCEEDED:
            self.mission.report(i, "reached")
        elif status == GoalStatus.STATUS_ABORTED:
            self.mission.report(i, "failed")

    def cancel(self, i: int) -> None:
        self.goal_seq[i] += 1                        # late results of the old goal are ignored
        gh = self.goal_handle.pop(i, None)
        if gh is not None:
            gh.cancel_goal_async()

    # -- detection -------------------------------------------------------------------------
    def on_detections(self, i: int, msg: Detection2DArray) -> None:
        if not self.mission.started or self.finished:
            return
        best = None
        for det in msg.detections:
            for h in det.results:
                if h.hypothesis.class_id in self.target_classes and h.hypothesis.score >= self.min_score:
                    if best is None or h.hypothesis.score > best[1]:
                        best = (h.hypothesis.class_id, float(h.hypothesis.score))
        self.det_count[i] = self.det_count[i] + 1 if best else 0
        if best and self.det_count[i] >= self.det_frames:
            p = self.robot_pose(i)
            self.finish("target found", robot=self.robots[i], cls=best[0], score=best[1],
                        pose=list(p) if p else None)

    # -- output ----------------------------------------------------------------------------
    def log(self, name: str, rec: Dict) -> None:
        if self.run_dir is None:
            return
        with open(os.path.join(self.run_dir, f"{name}.jsonl"), "a") as f:
            f.write(json.dumps(_jsonable(rec)) + "\n")

    def publish_merged(self) -> None:
        g = self.mission.merged
        m = OccupancyGrid()
        m.header.frame_id = self.world_frame
        m.header.stamp = self.get_clock().now().to_msg()
        m.info.resolution = float(self.spec.cell)
        m.info.width, m.info.height = int(self.spec.cols), int(self.spec.rows)
        m.info.origin.position.x, m.info.origin.position.y = float(self.spec.x0), float(self.spec.y0)
        m.info.origin.orientation.w = 1.0
        out = np.where(g == 1, 100, np.where(g == 0, 0, -1)).astype(np.int8)
        m.data = out.ravel().tolist()                # row-major, row = y: the planning grid's own order
        self.merged_pub.publish(m)

    def publish_targets(self) -> None:
        pa = PoseArray()
        pa.header.frame_id = self.world_frame
        pa.header.stamp = self.get_clock().now().to_msg()
        for r in self.mission.robots:
            if r.target is not None:
                p = Pose()
                p.position.x, p.position.y = self.spec.center(r.target)
                p.orientation.w = 1.0
                pa.poses.append(p)
        self.targets_pub.publish(pa)


def main(args=None):
    rclpy.init(args=args)
    node = ComresNode()
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    finally:
        node.finish("node shut down")
        node.destroy_node()
        if rclpy.ok():
            rclpy.shutdown()
