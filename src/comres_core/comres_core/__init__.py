"""
COMRES for real robots, without ROS.

algo/   the COMRES-VLM algorithm modules, unchanged (tools/sync_core.sh; algo/CORE_VERSION names
        the commit). They import each other as top-level modules (`import units`), so this
        package puts algo/ on sys.path.
grid.py the planning grid and the conversion of a ROS occupancy grid onto it
online  real-time COMRES: the simulator's COMRESStrategy driven by real maps, poses and goals
"""
import os
import sys

ALGO_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "algo")
if ALGO_DIR not in sys.path:
    sys.path.insert(0, ALGO_DIR)


def core_version() -> str:
    with open(os.path.join(ALGO_DIR, "CORE_VERSION")) as f:
        return f.read()
