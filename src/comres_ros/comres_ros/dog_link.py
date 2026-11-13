"""
The central computer's end of the link to each dog (dog_wire.py). For every robot it connects to
that dog's dog_bridge (retrying until the dog is up) and puts the dog on this computer's ROS graph
the way comres_node expects a robot, whatever ROS 2 the dog runs:

  /<robot>/map               nav_msgs/OccupancyGrid, latched, frame <robot>/<the dog's map frame>;
                             the last map again every second, as SLAM Toolbox does (RTAB-Map only
                             publishes a changed map, and comres_node subscribes volatile)
  TF                         <robot>/<map frame> -> <robot>/<base frame>, 10 Hz
  /<robot>/detections        std_msgs/String, JSON [{"class": ..., "score": ...}]
                             (comres_node: detection_type std_msgs/String)
  /<robot>/navigate_to_pose  nav2_msgs/action/NavigateToPose: the goal goes to the dog's Nav2 and
                             ends as it ends there; a newer goal replaces the running one (aborted),
                             as Nav2 does. Rejected while the dog is not connected or its Nav2 is
                             down; aborted when the link drops.
  /comres/mission            std_msgs/String, latched: JSON {id, task, hint, targets, robot}, each new
                             mission a dog asks for (dog_bridge --task; comres_node starts it)

Everything is stamped with this computer's clock when it arrives.

    ros2 run comres_ros comres_dog_link --ros-args -p robots:="[go2_1]" -p hosts:="[100.125.2.71:9100]"
"""

import array
import json
import math
import socket
import threading
import time
import zlib
from typing import Dict, List

import rclpy
from geometry_msgs.msg import TransformStamped
from nav2_msgs.action import NavigateToPose
from nav_msgs.msg import OccupancyGrid
from rclpy.action import ActionServer, CancelResponse, GoalResponse
from rclpy.callback_groups import ReentrantCallbackGroup
from rclpy.executors import ExternalShutdownException, MultiThreadedExecutor
from rclpy.node import Node
from rclpy.qos import DurabilityPolicy, QoSProfile, ReliabilityPolicy
from std_msgs.msg import String
from tf2_ros import TransformBroadcaster

from comres_ros import dog_wire

ENDED = ("succeeded", "aborted", "canceled", "rejected", "replaced", "lost")


def yaw_of(q) -> float:
    return math.atan2(2.0 * (q.w * q.z + q.x * q.y), 1.0 - 2.0 * (q.y * q.y + q.z * q.z))


class Dog:
    """One robot: its connection and the goals running on it."""

    def __init__(self, name: str, host: str, port: int):
        self.name, self.host, self.port = name, host, port
        self.sock = None
        self.send_lock = threading.Lock()
        self.lock = threading.Lock()
        self.nav_ready = False
        self.map_msg = None                                  # the last map, republished every second
        self.map_time = 0.0
        self.mission_id = None                               # the last mission it asked for
        self.goals: Dict[int, dict] = {}                     # id -> {status, code, done: Event}
        self.active = None
        self.next_id = 0

    def frame(self, f: str) -> str:
        return f"{self.name}/{f}"

    def send(self, header: dict) -> bool:
        with self.send_lock:
            if self.sock is None:
                return False
            try:
                self.sock.sendall(dog_wire.pack(header))
                return True
            except OSError:
                return False

    def end_goal(self, gid: int, status: str, code=None) -> None:
        with self.lock:
            g = self.goals.get(gid)
            if g is not None and not g["done"].is_set():
                g["status"], g["code"] = status, code
                g["done"].set()


class DogLink(Node):
    def __init__(self):
        super().__init__("dog_link")
        robots = list(self.declare_parameter("robots", ["go2_1"]).value)
        hosts = list(self.declare_parameter("hosts", [f"127.0.0.1:{dog_wire.PORT}"]).value)
        if len(robots) != len(hosts):
            raise ValueError(f"robots {robots} and hosts {hosts} must pair up")
        self.tfb = TransformBroadcaster(self)
        latched = QoSProfile(depth=1, reliability=ReliabilityPolicy.RELIABLE, durability=DurabilityPolicy.TRANSIENT_LOCAL)
        group = ReentrantCallbackGroup()
        self.mission_pub = self.create_publisher(String, "/comres/mission", latched)
        self.dogs: List[Dog] = []
        for r, h in zip(robots, hosts):
            host, _, port = h.rpartition(":")
            dog = Dog(r, host or h, int(port) if host else dog_wire.PORT)
            dog.map_pub = self.create_publisher(OccupancyGrid, f"/{r}/map", latched)
            dog.det_pub = self.create_publisher(String, f"/{r}/detections", 10)
            dog.server = ActionServer(self, NavigateToPose, f"/{r}/navigate_to_pose",
                                      execute_callback=lambda gh, d=dog: self.execute(d, gh),
                                      goal_callback=lambda req, d=dog: GoalResponse.ACCEPT if d.sock is not None and d.nav_ready else GoalResponse.REJECT,
                                      cancel_callback=lambda gh: CancelResponse.ACCEPT,
                                      callback_group=group)
            self.dogs.append(dog)
            threading.Thread(target=self.run, args=(dog,), daemon=True).start()
        self.create_timer(1.0, self.republish_maps)
        self.get_logger().info("dog link: " + ", ".join(f"{d.name} at {d.host}:{d.port}" for d in self.dogs))

    # -- the connection (one thread per dog) -----------------------------------------------
    def run(self, dog: Dog) -> None:
        warned = False
        while rclpy.ok():
            try:
                s = socket.create_connection((dog.host, dog.port), timeout=5.0)
            except OSError as e:
                if not warned:
                    self.get_logger().warning(f"{dog.name}: cannot reach {dog.host}:{dog.port} ({e}); retrying")
                    warned = True
                time.sleep(2.0)
                continue
            s.settimeout(None)
            s.setsockopt(socket.IPPROTO_TCP, socket.TCP_NODELAY, 1)
            with dog.send_lock:
                dog.sock = s
            warned = False
            self.get_logger().info(f"{dog.name}: connected to {dog.host}:{dog.port}")
            try:
                while True:
                    h, body = dog_wire.recv(s)
                    self.on_message(dog, h, body)
            except (ConnectionError, OSError, ValueError) as e:
                self.get_logger().warning(f"{dog.name}: link lost ({e})")
            with dog.send_lock:
                dog.sock = None
            dog.nav_ready = False
            s.close()
            for gid in list(dog.goals):
                dog.end_goal(gid, "lost")
            time.sleep(1.0)

    def on_message(self, dog: Dog, h: dict, body: bytes) -> None:
        ch = h.get("ch")
        now = self.get_clock().now().to_msg()
        if ch == "pose":
            t = TransformStamped()
            t.header.stamp = now
            t.header.frame_id, t.child_frame_id = dog.frame(h["frame"]), dog.frame(h["child"])
            x, y, z, qx, qy, qz, qw = h["tf"]
            t.transform.translation.x, t.transform.translation.y, t.transform.translation.z = x, y, z
            t.transform.rotation.x, t.transform.rotation.y, t.transform.rotation.z, t.transform.rotation.w = qx, qy, qz, qw
            self.tfb.sendTransform(t)
        elif ch == "map":
            m = OccupancyGrid()
            m.header.stamp = now
            m.header.frame_id = dog.frame(h["frame"])
            m.info.resolution = float(h["resolution"])
            m.info.width, m.info.height = int(h["width"]), int(h["height"])
            p, q = m.info.origin.position, m.info.origin.orientation
            p.x, p.y, p.z, q.x, q.y, q.z, q.w = (float(v) for v in h["origin"])
            m.data = array.array("b", zlib.decompress(body))
            dog.map_pub.publish(m)
            dog.map_msg, dog.map_time = m, time.monotonic()
        elif ch == "dets":
            dog.det_pub.publish(String(data=json.dumps([{"class": c, "score": s} for c, s in h["items"]])))
        elif ch == "nav":
            if h["status"] != "accepted":
                dog.end_goal(int(h["id"]), h["status"], h.get("code"))
        elif ch == "hello":
            if bool(h.get("nav_ready")) != dog.nav_ready:
                self.get_logger().info(f"{dog.name}: Nav2 {'up' if h.get('nav_ready') else 'down'}")
            dog.nav_ready = bool(h.get("nav_ready"))
            m = h.get("mission")
            if m and m.get("id") != dog.mission_id:
                dog.mission_id = m.get("id")
                self.get_logger().info(f"{dog.name} asks for a {m.get('task')} mission ({m.get('id')})")
                self.mission_pub.publish(String(data=json.dumps({**m, "robot": dog.name})))

    def republish_maps(self) -> None:
        for dog in self.dogs:
            m = dog.map_msg
            if m is not None and dog.sock is not None and time.monotonic() - dog.map_time >= 1.0:
                m.header.stamp = self.get_clock().now().to_msg()
                dog.map_pub.publish(m)

    # -- NavigateToPose (executor threads) -------------------------------------------------
    def execute(self, dog: Dog, gh):
        p = gh.request.pose
        frame = p.header.frame_id
        prefix = dog.name + "/"
        done = threading.Event()
        with dog.lock:
            dog.next_id += 1
            gid = dog.next_id
            prev, dog.active = dog.active, gid
            dog.goals[gid] = {"status": None, "code": None, "done": done}
        if prev is not None:
            dog.end_goal(prev, "replaced")                  # the dog's Nav2 replaces it too
        sent = dog.send({"ch": "goal", "id": gid, "frame": frame[len(prefix):] if frame.startswith(prefix) else frame,
                         "x": p.pose.position.x, "y": p.pose.position.y, "yaw": yaw_of(p.pose.orientation)})
        if not sent:
            dog.end_goal(gid, "lost")
        cancel_at = None
        while not done.wait(0.1):
            if gh.is_cancel_requested and cancel_at is None:
                dog.send({"ch": "cancel", "id": gid})
                cancel_at = time.monotonic()
            if cancel_at is not None and time.monotonic() - cancel_at > 3.0:
                dog.end_goal(gid, "canceled")               # the dog did not answer
            if not rclpy.ok():
                dog.end_goal(gid, "lost")
        with dog.lock:
            g = dog.goals.pop(gid)
            if dog.active == gid:
                dog.active = None
        res = NavigateToPose.Result()
        if g["code"] is not None and hasattr(res, "error_code"):
            res.error_code = int(g["code"])
        if g["status"] == "succeeded":
            gh.succeed()
        elif g["status"] == "canceled" and gh.is_cancel_requested:
            gh.canceled()
        else:
            gh.abort()
        return res


def main(args=None):
    rclpy.init(args=args)
    node = DogLink()
    ex = MultiThreadedExecutor(num_threads=4 + 2 * len(node.dogs))
    ex.add_node(node)
    try:
        ex.spin()
    except (KeyboardInterrupt, ExternalShutdownException):
        pass
    finally:
        node.destroy_node()
        rclpy.try_shutdown()
