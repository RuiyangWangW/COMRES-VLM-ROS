# How COMRES runs on the robots, and how it was checked

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
  comres_ros/discover.py      `ros2 run comres_ros comres_discover`: what is running, and a config snippet
  config/comres.yaml          robots, topics, frames, planning grid, hint, VLM server
  launch/comres.launch.py
tools/sync_core.sh     refresh algo/ from a COMRES-VLM checkout (refuses uncommitted changes)
tools/shadow_check.py  the same decisions as the simulator, step for step, on recorded runs
tools/fake_dog.py, tools/fake_dog_test.sh   a fake Go2 for running the node without hardware
tools/fake_dog_manual_test.sh  ~/start and ~/stop against the fake Go2
tools/call_service.py  call ~/start or ~/stop without the ros2 command line
```

`old_ros_ws` (the earlier LLM waypoint planner, ROS_CoX_ws) was the reference for the topics and
the Nav2 hand-off. What changed: COMRES instead of the GPT-4o waypoint prompt; live SLAM maps
instead of a prebuilt map with simulated sensing; one central node instead of nodes passing files;
any number of robots (nothing hard-coded to one robot); no API key in the code.

## What each robot needs

* its driver (lidar scan, odometry and TF odom -> base, velocity commands from Nav2);
* **SLAM Toolbox** (or another SLAM) publishing the live map, `nav_msgs/OccupancyGrid`
  (default `/{robot}/map`; set `map_update_interval` to about 1 s so the map keeps up with the
  robot, as the simulated robot's map does every step);
* **Nav2** with the `NavigateToPose` action (default `/{robot}/navigate_to_pose`), planning on that
  map; a safe speed limit;
* **a detector** (the YOLO on the dog) publishing what it sees (default `/{robot}/detections`,
  `vision_msgs/Detection2DArray`; `detection_type` also takes `vision_msgs/Detection3DArray`,
  `yolo_msgs/DetectionArray` and `std_msgs/String` with class names).

The central computer runs the same ROS 2 version (Jazzy), sees the robots' topics (same
`ROS_DOMAIN_ID`), reaches the VLM server, and has its clock synchronised with the robots
(chrony / NTP). Python packages there: numpy, scipy, scikit-image, pillow, matplotlib, pyyaml
and openai (`pip install --user --break-system-packages openai`, or a venv made with
`--system-site-packages` so that it still sees ROS's packages).

**Find the names:** with the robot up, `ros2 run comres_ros comres_discover --vlm http://<gpu-box>:18765/v1`
lists the map topics (frame, resolution, how published), detector-like topics and their types,
the NavigateToPose servers, the TF tree and whether the VLM server answers, and prints a config
snippet.

**Frames.** With one robot nothing has to be set: the world frame and the Nav2 goal frame are the
robot's SLAM map frame (from its map messages) and the base frame is the first that exists of
`{robot}/base_footprint`, `{robot}/base_link`, `base_footprint`, `base_link`. With several robots,
set `world_frame` to a frame every robot's SLAM map frame is placed in (e.g. static transforms
from measured start poses). `~/start` refuses, saying what is missing, until every robot has a
map, its base and goal-frame transforms, and Nav2.

## The planning grid, and "north"

COMRES plans on a grid of `cell_size` cells, `grid_width` x `grid_height` metres, centred on the
team's start (`grid_auto_center`). Its parameters are in cells and were tuned for a 20-cell
sensing range, so set `cell_size = (the robots' useful sensing range) / 20` (5 m -> 0.25 m). A grid
cell is a wall if any SLAM pixel in it is occupied (>= 65; pixels grown by one so thin walls stay
closed at any angle), free if at least half of it is free (<= 25), else unknown. The node warns
when a SLAM map reaches outside the grid.

The VLM's picture is drawn with the grid's +y up and calls it **north** (+x east), as in the
simulated experiments, and hints use compass words. A SLAM map frame's +x is the robot's heading
at start-up, so either start the robot facing the direction the hint calls east, or set
`grid_yaw_deg` to the angle of building-east in the world frame.

## Running a mission

`ros2 launch comres_ros comres.launch.py`, then `ros2 service call /comres_node/start
std_srvs/srv/Trigger` (or `auto_start: true`). `~/start` answers "not ready: ..." with what is
missing until everything is up. `~/stop` ends the mission and cancels every robot's goal; called
before a mission, it keeps the node from starting one. One node runs one mission: restart the
node for the next. A search ends when a robot's detector reports a target class
`detection_frames` times in a row; exploration when no robot has a frontier left for
`no_frontier_ticks` ticks; both after `max_mission_seconds`.

## Same as the simulation

`comres_core.online.Mission` runs the simulator's COMRESStrategy unchanged, step for step: one tick
is one simulator step (sensing, the "seen = reached" check, the selector, the move, the arrival
test, then the plan delivery and every T_H steps the merge and the question). What differs is only
what the real world makes different:

* the planner's latency is real (its question runs in the background; the plan takes effect at
  the first tick after the answer);
* motion is Nav2's. Goals are moved up to 0.5 m off walls (`goal_clearance`), as Nav2 refuses goals
  in its obstacle margin; a goal is given up as unreachable when Nav2 aborts it or when neither
  Nav2's remaining path nor the straight-line distance shrinks for `goal_timeout`; a goal Nav2
  gives up at the target counts as reached, as the simulator never gives up a target it is at;
  a goal's outcome only counts for the target it was sent for;
* a robot's map between merges is its live SLAM map, with the last merge filling what it does not
  know; the 2-cell disc around the robot counts as seen where neither knows it;
* a robot that, staying where it reached a target, picks that same target 3 times (its frontier
  never clears: a SLAM map that lags, glass, a blind spot) avoids it for `arrival_hold` ticks
  instead of standing still. A robot moving along a frontier edge picks a new cell each time, so
  this never acts in the simulator.

In the simulator the robot moves one cell per step: with `tick_seconds` 1, a Nav2 speed of
`cell_size` m/s keeps that ratio (faster is fine, it only moves more per step).

`tools/shadow_check.py` checks this on recorded COMRES-VLM runs: the simulator replays a run (the
VLM's recorded replies, the recorded delays) while a Mission gets, every step, what the ROS node
would get from the robots; the Mission must choose the same target for every robot at every step,
merge the same maps and deliver the same plans at the same steps. On 2026-10-01 all 1,100
recorded runs (search and explore, every map size) were the same at every step.

`tools/fake_dog.py` is a fake Go2 (a map revealed by a 360-degree scan as "SLAM", a NavigateToPose
server with Nav2's feedback and error codes, a detector) to run the node without hardware;
`tools/fake_dog_test.sh` runs a mission against it and passes when the target is found.

## Logs

Each mission writes `run_dir/<start time>/`: `meta.json` (settings, start cells and poses, the
algorithm's commit), `events.jsonl` (merges, plans, targets, goals, seen / reached / failed,
the end and why), `plans.jsonl` (every plan with the VLM's reply), `plans/plan_<step>.png` and
`.txt` (the picture and text the VLM was given), `poses.jsonl` (every tick: cells, poses, targets)
and `maps/merge_<step>.npz` (each merged map).

## Tests

* `python -m pytest src/comres_core/test` (no ROS): the grid conversion (thin walls at any angle,
  uncertain cells, rotated frames) and missions on a small simulated world (two robots, one
  robot, a slow VLM that must not stop the robots, a failed goal that must be avoided, a frontier
  that never clears, a robot moving along an edge that must not count as stuck, goal outcomes
  for an older target ignored);
* `python tools/shadow_check.py <results .../comres.jsonl> --sim <COMRES-VLM>`: the same decisions
  as the simulator, step for step (above);
* `tools/fake_dog_test.sh` (ROS 2 Jazzy): the node end to end against the fake dog;
* `tools/fake_dog_manual_test.sh` (ROS 2 Jazzy): `~/start` before the robot is up, after, twice,
  and after the end; `~/stop` before any mission.
