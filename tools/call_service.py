"""
Call a std_srvs/Trigger service and print its answer (for tests without the ros2 CLI):

    python tools/call_service.py /comres_node/start [--timeout 10]
"""

import argparse
import sys

import rclpy
from std_srvs.srv import Trigger


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("service")
    ap.add_argument("--timeout", type=float, default=10.0)
    a, ros_args = ap.parse_known_args()
    rclpy.init(args=ros_args)
    node = rclpy.create_node("call_service")
    cli = node.create_client(Trigger, a.service)
    if not cli.wait_for_service(timeout_sec=a.timeout):
        print(f"{a.service}: no such service", flush=True)
        return 2
    fut = cli.call_async(Trigger.Request())
    rclpy.spin_until_future_complete(node, fut, timeout_sec=a.timeout)
    res = fut.result()
    print(f"{a.service}: " + ("no answer" if res is None else f"success={res.success} message={res.message!r}"), flush=True)
    node.destroy_node()
    rclpy.try_shutdown()
    return 0 if res is not None else 1


if __name__ == "__main__":
    sys.exit(main())
