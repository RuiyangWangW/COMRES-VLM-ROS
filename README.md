# COMRES on real robots (ROS 2 Jazzy)

COMRES — the VLM team planner of [COMRES-VLM](https://github.com/RuiyangWangW/COMRES-VLM) — on a
team of Unitree Go2 robots. Each robot runs its own SLAM, Nav2 and target detector; one central
node runs COMRES exactly as in the simulated experiments:

* every **T_H = 100 s** the robots' maps are merged and the VLM (Qwen3.8-27B on our server) is
  asked for a plan: one piece of the unknown space per robot and, in search, where the target
  probably is;
* in between, whenever a robot's target is **seen, reached or unreachable**, the COMRES selector
  picks its next frontier on that robot's own map (its live SLAM map on top of the last merged
  map), and the node sends it to the robot's Nav2;
* the VLM question runs in the background, so the robots never wait for it; the plan takes
  effect when the answer arrives;
* the search ends when a robot's detector reports the target.

## Layout

```
src/comres_core/       no ROS
  comres_core/algo/    the COMRES-VLM algorithm modules, unchanged; CORE_VERSION names the commit
  comres_core/grid.py  the planning grid in the world frame; SLAM occupancy grid -> planning grid
  comres_core/online.py  Mission: the simulator's COMRESStrategy driven in real time
  test/                pytest, runs anywhere (python -m pytest src/comres_core/test)
src/comres_ros/        ROS 2
  comres_ros/comres_node.py   the central node
  config/comres.yaml          robots, frames, topics, planning grid, hint, VLM server
  launch/comres.launch.py
tools/sync_core.sh     refresh algo/ from a COMRES-VLM checkout (refuses uncommitted changes)
```

`old_ros_ws` (the earlier LLM waypoint planner, ROS_CoX_ws) was the reference for the topics and
the Nav2 hand-off. What changed: COMRES instead of the GPT-4o waypoint prompt; live SLAM maps
instead of a prebuilt map with simulated sensing; one central node instead of nodes passing files;
any number of robots (nothing hard-coded to one robot); no API key in the code.

## What each robot must provide

| what | default (`{robot}` = robot id) |
|---|---|
| live SLAM map, `nav_msgs/OccupancyGrid` | `/{robot}/map` |
| TF from the shared world frame to the robot's map frame and base | `map` -> `{robot}/map` -> `{robot}/base_link` |
| Nav2 `NavigateToPose` action, goals in its map frame | `/{robot}/navigate_to_pose`, frame `{robot}/map` |
| its detector's output, `vision_msgs/Detection2DArray` (class name in `results[].hypothesis.class_id`) | `/{robot}/detections` |

The robots' map frames must be placed in one world frame, e.g. by static transforms from where
each robot starts (measured), or by a shared SLAM map (then `world_frame` and `nav_frame` may be
the same frame).

## The planning grid

COMRES plans on a grid of `cell_size` cells over `grid_x0, grid_y0, grid_width, grid_height` (m) in
the world frame, x east and y north (a larger row is further north, as in COMRES-VLM). Its
parameters are in cells and were tuned for a 20-cell sensing range, so set
`cell_size = (the robots' useful sensing range) / 20` (5 m -> 0.25 m). A grid cell is a wall if any
SLAM cell in it is occupied (>= `occ_threshold`), free if at least `free_fraction` of it is free,
else unknown.

## Build and run

```bash
source /opt/ros/jazzy/setup.bash
cd comres_ros_ws && colcon build && source install/setup.bash
# the VLM: COMRES-VLM serve_vlm.sh on the GPU box; set vlm_base_url to reach it
ros2 launch comres_ros comres.launch.py hint:="The target is ..." vlm_base_url:=http://<gpu-box>:18765/v1
ros2 service call /comres_node/start std_srvs/srv/Trigger     # starts the mission (the start cells are the team start)
ros2 service call /comres_node/stop std_srvs/srv/Trigger      # ends it early
```

`fake_vlm: true` answers with random valid plans (no server) for dry runs. RViz: `/comres_node/merged_map`
(the last merged map) and `/comres_node/targets` (the robots' current targets).

## Logs

Each mission writes `run_dir/<start time>/`: `meta.json` (settings, start cells and poses, the
algorithm's commit), `events.jsonl` (merges, plans, targets, goals, seen / reached / failed,
the end and why), `plans.jsonl` (every plan with the VLM's reply), `poses.jsonl` (every tick:
cells, poses, targets) and `maps/merge_<step>.npz` (each merged map).

## Tests

`python -m pytest src/comres_core/test` (numpy, scipy, scikit-image, pillow, openai; no ROS):
the grid conversion, and missions on a small simulated world with perfect sensing, the VLM in
fake mode, a slow VLM that must not stop the robots, and a failed goal that must be avoided.
The ROS node itself has not been run yet (no ROS on the machine it was written on).
