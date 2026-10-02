"""
What is running, and what to put in config/comres.yaml. Run it on the computer that will run the
COMRES node, with the robot(s) up (driver, SLAM Toolbox, Nav2, the detector):

    ros2 run comres_ros comres_discover [--seconds 5] [--vlm http://<gpu-box>:18765/v1]

It lists the occupancy-grid topics (with their frame, resolution, size and how they are
published), the detector-like topics and their message types, the NavigateToPose action servers,
the TF tree (frames and parents), the clock offset to the robots' stamps, and whether the VLM
server answers. Then it prints a config snippet from what it found.
"""

import argparse
import json
import time
import urllib.request

import rclpy
from rclpy.node import Node
from rclpy.qos import DurabilityPolicy, QoSProfile, ReliabilityPolicy

DETECTION_HINTS = ("detection", "yolo", "vision_msgs", "object")


def _qos_str(info) -> str:
    q = info.qos_profile
    return f"{q.reliability.name.lower()}/{q.durability.name.lower()}"


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--seconds", type=float, default=5.0, help="how long to listen")
    ap.add_argument("--vlm", default="", help="VLM server base URL to test, e.g. http://host:18765/v1")
    a, ros_args = ap.parse_known_args(argv)
    rclpy.init(args=ros_args)
    node = Node("comres_discover")
    from nav_msgs.msg import OccupancyGrid
    from tf2_ros import Buffer, TransformListener
    tf = Buffer()
    TransformListener(tf, node)

    t_end = time.monotonic() + 1.0
    while time.monotonic() < t_end:                     # let discovery settle
        rclpy.spin_once(node, timeout_sec=0.1)
    topics = node.get_topic_names_and_types()
    maps = [t for t, ty in topics if "nav_msgs/msg/OccupancyGrid" in ty]
    got = {}
    for t in maps:                                       # both kinds of publisher
        for q in (QoSProfile(depth=1, reliability=ReliabilityPolicy.RELIABLE, durability=DurabilityPolicy.TRANSIENT_LOCAL),
                  QoSProfile(depth=1, reliability=ReliabilityPolicy.BEST_EFFORT, durability=DurabilityPolicy.VOLATILE)):
            node.create_subscription(OccupancyGrid, t, lambda m, t=t: got.__setitem__(t, m), q)
    t_end = time.monotonic() + a.seconds
    while time.monotonic() < t_end:
        rclpy.spin_once(node, timeout_sec=0.1)

    print("\n== Occupancy grids (SLAM maps and costmaps)")
    if not maps:
        print("  none found: is SLAM Toolbox running, and on the same ROS_DOMAIN_ID?")
    for t in sorted(maps):
        pubs = node.get_publishers_info_by_topic(t)
        how = ", ".join(f"{p.node_name} ({_qos_str(p)})" for p in pubs) or "no publisher"
        m = got.get(t)
        if m is not None:
            age = node.get_clock().now().nanoseconds * 1e-9 - (m.header.stamp.sec + m.header.stamp.nanosec * 1e-9)
            print(f"  {t}: frame '{m.header.frame_id}', {m.info.resolution:.3f} m, {m.info.width} x {m.info.height}, "
                  f"stamp {age:+.1f} s old; published by {how}")
        else:
            print(f"  {t}: no message in {a.seconds:.0f} s; published by {how}")
        if "costmap" in t:
            print("     (a costmap: COMRES wants the SLAM map itself, not this)")

    print("\n== Detector-like topics")
    dets = [(t, ty) for t, ty in topics if any(h in (t + " ".join(ty)).lower() for h in DETECTION_HINTS)]
    for t, ty in sorted(dets):
        pubs = node.get_publishers_info_by_topic(t)
        print(f"  {t}: {', '.join(ty)}; " + (", ".join(f"{p.node_name} ({_qos_str(p)})" for p in pubs) or "no publisher"))
    if not dets:
        print("  none found (is the YOLO node running?)")

    print("\n== NavigateToPose action servers")
    try:
        from rclpy.action.graph import get_action_names_and_types
        acts = [n for n, ty in get_action_names_and_types(node) if "nav2_msgs/action/NavigateToPose" in ty]
    except Exception as e:                               # noqa: BLE001
        acts = []
        print(f"  could not list actions: {e}")
    for n in sorted(acts):
        print(f"  {n}")
    if not acts:
        print("  none found (is Nav2 up and active?)")

    print("\n== TF frames (frame: parent)")
    try:
        import yaml
        frames = yaml.safe_load(tf.all_frames_as_yaml()) or {}
    except Exception:                                    # noqa: BLE001
        frames = {}
    for f, d in sorted(frames.items()):
        print(f"  {f}: parent {d.get('parent')}, {d.get('rate', 0)} Hz")
    if not frames:
        print("  no TF received")

    if a.vlm:
        print(f"\n== VLM server {a.vlm}")
        try:
            with urllib.request.urlopen(a.vlm.rstrip("/") + "/models", timeout=5) as r:
                ids = [m.get("id") for m in json.loads(r.read()).get("data", [])]
            print(f"  answers; models: {ids}")
        except Exception as e:                           # noqa: BLE001
            print(f"  NOT reachable: {e}")

    print("\n== Suggested config (check every line)")
    slam = [t for t in maps if "costmap" not in t]
    robots = sorted({t.strip("/").split("/")[0] for t in slam if t.count("/") >= 2}) or ["go2_1"]
    m0 = got.get(slam[0]) if slam else None
    base = next((f for f in frames if f.endswith("base_link") or f.endswith("base_footprint")), "base_link")
    print(f"    robot_ids: {robots}")
    if slam:
        print(f"    map_topic: \"{slam[0]}\"      # with several robots: put {{robot}} where the robot name is")
    if m0 is not None:
        print(f"    world_frame: \"{m0.header.frame_id}\"      # one robot: its SLAM frame")
        print(f"    nav_frame: \"{m0.header.frame_id}\"")
    print(f"    base_frame: \"{base}\"")
    if acts:
        print(f"    nav_action: \"{acts[0]}\"")
    if dets:
        supported = ("vision_msgs/Detection2DArray", "vision_msgs/Detection3DArray", "yolo_msgs/DetectionArray",
                     "std_msgs/String")                  # comres_node.DETECTION_TYPES
        named = [(t, ty.replace("/msg/", "/")) for t, tys in dets for ty in tys]
        t, ty = next(((t, ty) for t, ty in named if ty in supported), named[0])
        print(f"    detection_topic: \"{t}\"")
        if ty in supported:
            print(f"    detection_type: \"{ty}\"")
        else:
            print(f"    # detection_type: {ty} is not supported yet (supported: {', '.join(supported)})")
    node.destroy_node()
    rclpy.shutdown()


if __name__ == "__main__":
    main()
