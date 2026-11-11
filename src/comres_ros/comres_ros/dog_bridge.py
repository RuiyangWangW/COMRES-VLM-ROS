"""
The dog's end of the link to the central COMRES node (dog_wire.py), for any ROS 2 (the Go2 runs
Foxy). It listens on --port; the central computer's dog_link connects. It sends the SLAM map, the
base pose in the map frame, the detector's classes and its Nav2 goals' outcomes, and gives the
central computer's goals to Nav2. When the connection drops it cancels the goals it gave, so the
robot does not drive on alone.

    python3 dog_bridge.py [--robot go2_1] [--port 9100] [--map-topic /map] [--base-frame base_link]
        [--nav-action /navigate_to_pose] [--det-topic /detected_objects] [--det-type markers]

--det-type: markers (visualization_msgs/MarkerArray with TEXT labels "<class> (<score>%)", as
cpsl_detection's yoloDetector publishes them, once per new object), detection2d
(vision_msgs/Detection2DArray) or string (std_msgs/String, comma-separated class names).
--bt: the behavior tree Nav2 runs COMRES's goals with (default comres_bt.xml beside this file:
fail fast, no spinning; "" for bt_navigator's default).
On the dog, copy this file, dog_wire.py and comres_bt.xml into one folder.
"""

import argparse
import math
import os
import re
import socket
import threading
import time
import zlib
from collections import deque

import numpy as np
import rclpy
from action_msgs.msg import GoalStatus
from nav2_msgs.action import NavigateToPose
from nav_msgs.msg import OccupancyGrid
from rclpy.action import ActionClient
from rclpy.node import Node
from rclpy.qos import DurabilityPolicy, QoSProfile, ReliabilityPolicy, qos_profile_sensor_data
from rclpy.time import Time
from tf2_ros import Buffer, TransformListener

try:
    from comres_ros import dog_wire
except ImportError:                                          # on the dog: dog_wire.py alongside
    import dog_wire

OUTCOME = {GoalStatus.STATUS_SUCCEEDED: "succeeded", GoalStatus.STATUS_ABORTED: "aborted",
           GoalStatus.STATUS_CANCELED: "canceled"}
LABEL = re.compile(r"^\s*(.+?)\s*\((\d+(?:\.\d+)?)%\)\s*$")    # "backpack (87%)"


class DogBridge(Node):
    def __init__(self, a):
        super().__init__("comres_dog_bridge")
        self.a = a
        log = self.get_logger()
        self.warn = getattr(log, "warning", None) or log.warn  # Foxy and later
        self.tf = Buffer()
        self.tf_listener = TransformListener(self.tf, self)
        self.nav = ActionClient(self, NavigateToPose, a.nav_action)
        self.map_msg = None
        self.map_dirty = False
        self.map_sent = 0.0
        self.goals = {}                                     # central goal id -> goal handle (None: pending)
        self.cancel_pending = set()
        self.inbox = deque()                                # commands, handled on the executor thread
        self.conn = None
        self.out_cv = threading.Condition()
        self.fifo = deque()                                 # messages to send in order
        self.latest = {}                                    # channel -> its newest message (pose, map)

        latched = QoSProfile(depth=1, reliability=ReliabilityPolicy.RELIABLE, durability=DurabilityPolicy.TRANSIENT_LOCAL)
        self.create_subscription(OccupancyGrid, a.map_topic, self.on_map, latched)
        if a.det_type == "markers":
            from visualization_msgs.msg import MarkerArray
            self.create_subscription(MarkerArray, a.det_topic, self.on_markers, qos_profile_sensor_data)
        elif a.det_type == "detection2d":
            from vision_msgs.msg import Detection2DArray
            self.create_subscription(Detection2DArray, a.det_topic, self.on_detection2d, qos_profile_sensor_data)
        else:
            from std_msgs.msg import String
            self.create_subscription(String, a.det_topic, self.on_string, qos_profile_sensor_data)
        self.create_timer(0.1, self.tick)
        self.create_timer(1.0, lambda: self.send({"ch": "hello", "robot": a.robot, "nav_ready": self.nav.server_is_ready()}))
        threading.Thread(target=self.serve, daemon=True).start()
        threading.Thread(target=self.sender, daemon=True).start()
        log.info("dog bridge %s: port %d, map %s, base %s, Nav2 %s (tree %s), detections %s (%s)" % (
            a.robot, a.port, a.map_topic, a.base_frame, a.nav_action, a.bt or "default", a.det_topic, a.det_type))

    # -- the connection (its own threads) --------------------------------------------------
    def serve(self):
        srv = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
        srv.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
        srv.bind(("0.0.0.0", self.a.port))
        srv.listen(1)
        while rclpy.ok():
            c, addr = srv.accept()
            c.setsockopt(socket.IPPROTO_TCP, socket.TCP_NODELAY, 1)
            with self.out_cv:
                old, self.conn = self.conn, c
                self.fifo.clear()
                self.latest.clear()
            if old is not None:                             # one central computer at a time
                old.close()
            self.get_logger().info("central computer connected from %s:%d" % addr)
            self.map_dirty = self.map_msg is not None        # the newcomer needs the map
            try:
                while True:
                    h, _ = dog_wire.recv(c)
                    self.inbox.append(h)
            except (ConnectionError, OSError, ValueError) as e:
                self.warn("central computer gone (%s)" % e)
            with self.out_cv:
                if self.conn is c:
                    self.conn = None
                    self.inbox.append({"ch": "lost"})
            c.close()

    def send(self, header, body=b"", latest=False):
        if self.conn is None:
            return
        data = dog_wire.pack(header, body)
        with self.out_cv:
            if latest:
                self.latest[header["ch"]] = data
            else:
                self.fifo.append(data)
            self.out_cv.notify()

    def sender(self):
        while True:
            with self.out_cv:
                while not self.fifo and not self.latest:
                    self.out_cv.wait()
                data = self.fifo.popleft() if self.fifo else self.latest.pop(next(iter(self.latest)))
                c = self.conn
            if c is None:
                continue
            try:
                c.sendall(data)
            except OSError:
                try:
                    c.shutdown(socket.SHUT_RDWR)                # the receiving thread sees it closed
                except OSError:
                    pass

    # -- ROS (the executor thread) ---------------------------------------------------------
    def map_frame(self):
        return self.map_msg.header.frame_id if self.map_msg is not None else None

    def on_map(self, m):
        self.map_msg = m
        self.map_dirty = True

    def tick(self):
        while self.inbox:
            self.on_command(self.inbox.popleft())
        if self.conn is None:
            return
        frame = self.map_frame()
        if frame is not None:
            try:
                t = self.tf.lookup_transform(frame, self.a.base_frame, Time()).transform
                self.send({"ch": "pose", "frame": frame, "child": self.a.base_frame,
                           "tf": [t.translation.x, t.translation.y, t.translation.z,
                                  t.rotation.x, t.rotation.y, t.rotation.z, t.rotation.w]}, latest=True)
            except Exception as e:                          # noqa: BLE001
                self.warn("no TF %s <- %s: %s" % (frame, self.a.base_frame, e), throttle_duration_sec=5.0)
        if self.map_dirty and time.monotonic() - self.map_sent >= self.a.map_period:
            m, i = self.map_msg, self.map_msg.info
            o, q = i.origin.position, i.origin.orientation
            cells = np.asarray(m.data, dtype=np.int8).tobytes()
            self.send({"ch": "map", "frame": m.header.frame_id, "resolution": i.resolution, "width": i.width,
                       "height": i.height, "origin": [o.x, o.y, o.z, q.x, q.y, q.z, q.w]},
                      zlib.compress(cells, 1), latest=True)
            self.map_dirty = False
            self.map_sent = time.monotonic()

    def on_command(self, h):
        ch = h.get("ch")
        if ch == "goal":
            self.give_goal(h)
        elif ch == "cancel":
            gh = self.goals.get(h["id"])
            if gh is not None:
                gh.cancel_goal_async()
            elif h["id"] in self.goals:                     # not accepted yet: cancel on acceptance
                self.cancel_pending.add(h["id"])
        elif ch == "lost":
            for gh in self.goals.values():
                if gh is not None:
                    gh.cancel_goal_async()
            if self.goals:
                self.warn("link lost: cancelled %d goal(s)" % len(self.goals))

    def give_goal(self, h):
        gid = h["id"]
        frame = h.get("frame") or self.map_frame()
        if frame is None or not self.nav.server_is_ready():
            self.send({"ch": "nav", "id": gid, "status": "rejected", "code": None})
            self.warn("goal %d rejected: %s" % (gid, "no map yet" if frame is None else "Nav2 not available"))
            return
        g = NavigateToPose.Goal()
        g.pose.header.frame_id = frame                      # stamp 0: the latest transform
        g.pose.pose.position.x, g.pose.pose.position.y = float(h["x"]), float(h["y"])
        g.pose.pose.orientation.z = math.sin(h["yaw"] / 2.0)
        g.pose.pose.orientation.w = math.cos(h["yaw"] / 2.0)
        g.behavior_tree = self.a.bt
        self.goals[gid] = None
        self.nav.send_goal_async(g).add_done_callback(lambda f, gid=gid: self.on_response(gid, f))
        self.get_logger().info("goal %d: (%.2f, %.2f) in %s" % (gid, h["x"], h["y"], frame))

    def on_response(self, gid, fut):
        gh = fut.result()
        if gh is None or not gh.accepted:
            self.goals.pop(gid, None)
            self.cancel_pending.discard(gid)
            self.send({"ch": "nav", "id": gid, "status": "rejected", "code": None})
            return
        self.goals[gid] = gh
        self.send({"ch": "nav", "id": gid, "status": "accepted", "code": None})
        if gid in self.cancel_pending:
            self.cancel_pending.discard(gid)
            gh.cancel_goal_async()
        gh.get_result_async().add_done_callback(lambda f, gid=gid: self.on_result(gid, f))

    def on_result(self, gid, fut):
        r = fut.result()
        self.goals.pop(gid, None)
        code = getattr(r.result, "error_code", None)       # Foxy's NavigateToPose has none
        status = OUTCOME.get(r.status, "aborted")
        self.send({"ch": "nav", "id": gid, "status": status, "code": code})
        self.get_logger().info("goal %d %s%s" % (gid, status, "" if code is None else " (error %s)" % code))

    def send_items(self, items):
        if items:
            self.send({"ch": "dets", "items": items})

    def on_markers(self, msg):
        items = []
        for m in msg.markers:
            hit = LABEL.match(m.text) if m.type == m.TEXT_VIEW_FACING and m.action == m.ADD else None
            if hit:
                items.append([hit.group(1), float(hit.group(2)) / 100.0])
        self.send_items(items)

    def on_detection2d(self, msg):
        items = []
        for d in msg.detections:
            for r in d.results:
                hyp = getattr(r, "hypothesis", r)            # Foxy: id and score on the result itself
                items.append([str(getattr(hyp, "class_id", getattr(hyp, "id", ""))), float(hyp.score)])
        self.send_items(items)

    def on_string(self, msg):
        self.send_items([[c.strip(), 1.0] for c in msg.data.split(",") if c.strip()])


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--robot", default="go2_1")
    ap.add_argument("--port", type=int, default=dog_wire.PORT)
    ap.add_argument("--map-topic", default="/map")
    ap.add_argument("--map-period", type=float, default=1.0, help="send the map at most this often (s)")
    ap.add_argument("--base-frame", default="base_link")
    ap.add_argument("--nav-action", default="/navigate_to_pose")
    ap.add_argument("--det-topic", default="/detected_objects")
    ap.add_argument("--det-type", default="markers", choices=["markers", "detection2d", "string"])
    bt = os.path.join(os.path.dirname(os.path.abspath(__file__)), "comres_bt.xml")
    ap.add_argument("--bt", default=bt if os.path.exists(bt) else "")
    a, ros_args = ap.parse_known_args(argv)
    if a.bt and not os.path.exists(a.bt):
        ap.error("--bt %s: no such file" % a.bt)
    rclpy.init(args=ros_args)
    node = DogBridge(a)
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    except Exception as e:                                  # noqa: BLE001
        if type(e).__name__ != "ExternalShutdownException":  # Jazzy, on SIGTERM; Foxy has no such class
            raise
    node.destroy_node()
    if rclpy.ok():
        rclpy.shutdown()


if __name__ == "__main__":
    main()
