"""
A fake robot for testing the COMRES node without hardware (ROS 2 Jazzy, rclpy only).

It stands in for one Go2 with SLAM Toolbox, Nav2 and a detector:
  * the world is a grid map (.npy, 0 free / 1 wall; e.g. a COMRES-VLM benchmark map) of
    --world-cell metres per cell; the robot starts at --start (row, col);
  * "SLAM": a 360-degree scan of --range metres reveals the world into a map at --slam-res metres
    per pixel, published every --map-period seconds on /<robot>/map in frame --map-frame (the start
    pose is the frame's origin, as SLAM Toolbox does), latched;
  * TF --map-frame -> --base-frame at 20 Hz;
  * "Nav2": a NavigateToPose action server on /<robot>/navigate_to_pose: a shortest path over
    the world's free cells, followed at --speed m/s, with feedback (distance_remaining); a new
    goal replaces the running one (aborted, error 0, as Nav2 does); it succeeds within 0.25 m of
    the goal and aborts with error 208 (Nav2's "no valid path") if there is no path;
  * "YOLO": vision_msgs/Detection2DArray on /<robot>/detections at 5 Hz, holding --target-class
    when the target cell (--target row col) is within --range and in line of sight.

    python tools/fake_dog.py --world <map.npy> --start R C --target R C [--robot go2_1] ...
"""

import argparse
import math
import threading
import time
from collections import deque

import numpy as np
import rclpy
from geometry_msgs.msg import TransformStamped
from nav2_msgs.action import NavigateToPose
from nav_msgs.msg import OccupancyGrid
from rclpy.action import ActionServer, CancelResponse, GoalResponse
from rclpy.callback_groups import ReentrantCallbackGroup
from rclpy.executors import MultiThreadedExecutor
from rclpy.node import Node
from rclpy.qos import DurabilityPolicy, QoSProfile, ReliabilityPolicy
from tf2_ros import TransformBroadcaster
from vision_msgs.msg import Detection2D, Detection2DArray, ObjectHypothesisWithPose


class FakeDog(Node):
    def __init__(self, a):
        super().__init__("fake_dog")
        self.a = a
        self.world = (np.load(a.world) != 0).astype(np.int8)
        if self.world.ndim == 3:
            self.world = self.world[..., 0]
        self.wc = a.world_cell
        self.r0, self.c0 = a.start
        # pose in the map frame (metres); the map frame's origin is the start cell's centre
        self.x = self.y = self.yaw = 0.0
        self.lock = threading.Lock()
        n = int(math.ceil(max(self.world.shape) * self.wc / a.slam_res))
        self.known = np.full((n, n), -1, np.int8)            # SLAM pixels, row = map y
        self.origin = (-(self.c0 + 0.5) * self.wc, -(self.r0 + 0.5) * self.wc)   # map pixel (0,0) corner
        self.goal_id = 0
        latched = QoSProfile(depth=1, reliability=ReliabilityPolicy.RELIABLE, durability=DurabilityPolicy.TRANSIENT_LOCAL)
        self.map_pub = self.create_publisher(OccupancyGrid, f"/{a.robot}/map", latched)
        self.det_pub = self.create_publisher(Detection2DArray, f"/{a.robot}/detections", 10)
        self.tfb = TransformBroadcaster(self)
        cb = ReentrantCallbackGroup()
        self.srv = ActionServer(self, NavigateToPose, f"/{a.robot}/navigate_to_pose", self.execute,
                                goal_callback=lambda g: GoalResponse.ACCEPT,
                                cancel_callback=lambda g: CancelResponse.ACCEPT, callback_group=cb)
        self.create_timer(0.05, self.tf_tick)
        self.create_timer(a.map_period, self.map_tick)
        self.create_timer(0.2, self.det_tick)
        self.n_goals = 0
        self.sense()

    # world cells <-> map metres
    def cell(self, x=None, y=None):
        x = self.x if x is None else x
        y = self.y if y is None else y
        return (int(math.floor(self.r0 + 0.5 + y / self.wc)), int(math.floor(self.c0 + 0.5 + x / self.wc)))

    def center(self, rc):
        return ((rc[1] - self.c0) * self.wc, (rc[0] - self.r0) * self.wc)

    def los(self, a, b) -> bool:
        from skimage.draw import line
        rr, cc = line(a[0], a[1], b[0], b[1])
        return not any(self.world[r, c] for r, c in zip(rr[1:-1], cc[1:-1]))

    def sense(self):
        """Reveal the world within range and line of sight into the SLAM map."""
        from skimage.draw import line
        R = int(self.a.range / self.wc)
        r0, c0 = self.cell()
        H, W = self.world.shape
        k = int(round(self.wc / self.a.slam_res))
        for ang in np.linspace(0, 2 * math.pi, 720, endpoint=False):
            r1, c1 = int(round(r0 + R * math.sin(ang))), int(round(c0 + R * math.cos(ang)))
            rr, cc = line(r0, c0, r1, c1)
            for r, c in zip(rr, cc):
                if not (0 <= r < H and 0 <= c < W):
                    break
                v = 100 if self.world[r, c] else 0
                self.known[r * k:(r + 1) * k, c * k:(c + 1) * k] = v
                if self.world[r, c]:
                    break

    def tf_tick(self):
        t = TransformStamped()
        t.header.stamp = self.get_clock().now().to_msg()
        t.header.frame_id, t.child_frame_id = self.a.map_frame, self.a.base_frame
        with self.lock:
            t.transform.translation.x, t.transform.translation.y = self.x, self.y
            t.transform.rotation.z, t.transform.rotation.w = math.sin(self.yaw / 2), math.cos(self.yaw / 2)
        self.tfb.sendTransform(t)

    def map_tick(self):
        m = OccupancyGrid()
        m.header.frame_id = self.a.map_frame
        m.header.stamp = self.get_clock().now().to_msg()
        m.info.resolution = self.a.slam_res
        m.info.height, m.info.width = self.known.shape
        m.info.origin.position.x, m.info.origin.position.y = self.origin
        m.info.origin.orientation.w = 1.0
        m.data = self.known.ravel().tolist()
        self.map_pub.publish(m)

    def det_tick(self):
        out = Detection2DArray()
        out.header.stamp = self.get_clock().now().to_msg()
        tr, tc = self.a.target
        r, c = self.cell()
        if math.dist((r, c), (tr, tc)) * self.wc <= self.a.range and self.los((r, c), (tr, tc)):
            d = Detection2D()
            h = ObjectHypothesisWithPose()
            h.hypothesis.class_id, h.hypothesis.score = self.a.target_class, 0.9
            d.results.append(h)
            out.detections.append(d)
        self.det_pub.publish(out)

    def path(self, goal):
        """A shortest path over the world's free cells (8-connected), or None."""
        start = self.cell()
        H, W = self.world.shape
        prev = {start: None}
        q = deque([start])
        while q:
            u = q.popleft()
            if u == goal:
                break
            for dr in (-1, 0, 1):
                for dc in (-1, 0, 1):
                    v = (u[0] + dr, u[1] + dc)
                    if (dr or dc) and 0 <= v[0] < H and 0 <= v[1] < W and v not in prev and not self.world[v]:
                        if dr and dc and (self.world[u[0] + dr, u[1]] or self.world[u[0], u[1] + dc]):
                            continue
                        prev[v] = u
                        q.append(v)
        if goal not in prev:
            return None
        out = [goal]
        while prev[out[-1]] is not None:
            out.append(prev[out[-1]])
        return out[::-1]

    def execute(self, gh):
        self.n_goals += 1
        with self.lock:
            self.goal_id += 1
            me = self.goal_id
        p = gh.request.pose.pose.position
        goal = self.cell(p.x, p.y)
        path = self.path(goal) if 0 <= goal[0] < self.world.shape[0] and 0 <= goal[1] < self.world.shape[1] else None
        if path is None:
            gh.abort()
            res = NavigateToPose.Result()
            res.error_code = 208
            return res
        dt = 0.05
        fb = NavigateToPose.Feedback()
        for k, rc in enumerate(path[1:]):
            fb.distance_remaining = float((len(path) - 1 - k) * self.wc)
            gh.publish_feedback(fb)
            tx, ty = self.center(rc)
            while True:
                if gh.is_cancel_requested:
                    gh.canceled()
                    return NavigateToPose.Result()
                with self.lock:
                    if me != self.goal_id:                      # replaced by a newer goal
                        gh.abort()
                        return NavigateToPose.Result()
                    dx, dy = tx - self.x, ty - self.y
                    d = math.hypot(dx, dy)
                    step = self.a.speed * dt
                    if d <= step:
                        self.x, self.y = tx, ty
                    else:
                        self.x += dx / d * step
                        self.y += dy / d * step
                        self.yaw = math.atan2(dy, dx)
                self.sense() if d <= step else None
                if d <= step:
                    break
                time.sleep(dt)
            if math.hypot(p.x - self.x, p.y - self.y) <= 0.25:
                break
        gh.succeed()
        return NavigateToPose.Result()


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--world", required=True)
    ap.add_argument("--start", nargs=2, type=int, required=True)
    ap.add_argument("--target", nargs=2, type=int, required=True)
    ap.add_argument("--robot", default="go2_1")
    ap.add_argument("--map-frame", default="map")
    ap.add_argument("--base-frame", default="base_link")
    ap.add_argument("--world-cell", type=float, default=0.25)
    ap.add_argument("--slam-res", type=float, default=0.05)
    ap.add_argument("--range", type=float, default=5.0)
    ap.add_argument("--speed", type=float, default=0.5)
    ap.add_argument("--map-period", type=float, default=1.0)
    ap.add_argument("--target-class", default="backpack")
    ap.add_argument("--seconds", type=float, default=0.0, help="stop after this long (0: run until Ctrl-C)")
    a, ros_args = ap.parse_known_args()
    rclpy.init(args=ros_args)
    dog = FakeDog(a)
    ex = MultiThreadedExecutor(num_threads=4)
    ex.add_node(dog)
    if a.seconds > 0:
        threading.Thread(target=ex.spin, daemon=True).start()
        time.sleep(a.seconds)
        print(f"fake_dog: {dog.n_goals} goals, ended at cell {dog.cell()}", flush=True)
    else:
        try:
            ex.spin()
        except KeyboardInterrupt:
            pass
    dog.destroy_node()
    rclpy.try_shutdown()


if __name__ == "__main__":
    main()
