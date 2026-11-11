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
* the search ends when a robot's detector reports the target;
* the robots reach the central node through a socket link, so they may run another ROS 2 than
  it does (the Go2: Foxy; here: Jazzy): `dog_bridge.py` on each robot, `dog_link` here.

## Layout

```
src/comres_core/       no ROS
  comres_core/algo/    the COMRES-VLM algorithm modules, unchanged; CORE_VERSION names the commit
  comres_core/grid.py  the planning grid in the world frame; SLAM occupancy grid -> planning grid
  comres_core/online.py  Mission: the simulator's COMRESStrategy driven in real time
  test/                pytest, runs anywhere (python -m pytest src/comres_core/test)
src/comres_ros/        ROS 2
  comres_ros/comres_node.py   the central node
  comres_ros/dog_link.py      `comres_dog_link`: each robot on this computer's ROS graph, through its bridge
  comres_ros/dog_bridge.py    the robot's end of the link (any ROS 2, Python 3.8); with dog_wire.py
                              (the messages) and comres_bt.xml (Nav2's tree for COMRES goals)
  comres_ros/discover.py      `ros2 run comres_ros comres_discover`: what is running, and a config snippet
  config/comres.yaml          robots, topics, frames, planning grid, hint, VLM server, the link
  launch/comres.launch.py     the node and the link
tools/sync_core.sh     refresh algo/ from a COMRES-VLM checkout (refuses uncommitted changes)
tools/shadow_check.py  the same decisions as the simulator, step for step, on recorded runs
tools/fake_dog.py, tools/fake_dog_test.sh   a fake Go2 for running the node without hardware
tools/dog_bridge_test.sh  the fake Go2 through the link (bridge and link in separate ROS domains)
tools/fake_dog_manual_test.sh  ~/start and ~/stop against the fake Go2
tools/call_service.py  call ~/start or ~/stop without the ros2 command line
tools/vlm_check.py     the VLM server answers the node's planning question (no ROS)
```

`old_ros_ws` (the earlier LLM waypoint planner, ROS_CoX_ws) was the reference for the topics and
the Nav2 hand-off. What changed: COMRES instead of the GPT-4o waypoint prompt; live SLAM maps
instead of a prebuilt map with simulated sensing; one central node instead of nodes passing files;
any number of robots (nothing hard-coded to one robot); no API key in the code.

## What each robot needs

* its driver (lidar, odometry and TF odom -> base, velocity commands from Nav2);
* **a SLAM** publishing the live map, `nav_msgs/OccupancyGrid`, latched (the Go2: RTAB-Map's 2D
  map on `/map`, `slam3d`), and TF map -> base;
* **Nav2** with the `NavigateToPose` action (`/navigate_to_pose`), planning on that map; a safe
  speed limit;
* **a detector** (the Go2: cpsl_detection's YOLO, `/detected_objects` markers labelled
  "<class> (<score>%)", each object once; `--det-type` also takes `vision_msgs/Detection2DArray`
  and `std_msgs/String` with class names);
* **the bridge**, `dog_bridge.py --robot <name>` (see its `--help` for other topic names).

**The link** (`dog_wire.py`). The robot listens on port 9100 and `dog_link` connects to it,
retrying until it is up. One TCP connection carries, from the robot, the map (zlib, at most once a
second), the base pose in the map frame (10 Hz), the detected classes and how each Nav2 goal ends;
to the robot, goals and cancellations. `dog_link` publishes them as comres_node expects:
`/{robot}/map` (again every second, as SLAM Toolbox does: RTAB-Map republishes only a changed
map), TF `{robot}/map -> {robot}/base_link`, `/{robot}/detections` (`std_msgs/String`, JSON) and
the action `/{robot}/navigate_to_pose`, which ends as the robot's Nav2 goal ends (a newer goal
replaces the running one, as in Nav2). Each side stamps what it receives with its own clock, so
the clocks need not agree. When the link drops, the bridge cancels the robot's goals.

The central computer runs ROS 2 Jazzy and reaches the robots (e.g. over Tailscale) and the VLM
server. Python packages there: numpy, scipy, scikit-image, pillow, matplotlib, pyyaml and openai
(`pip install --user --break-system-packages openai`, or a venv made with
`--system-site-packages` so that it still sees ROS's packages). With robots on the same ROS 2,
`dog_link:=false` skips the link, and `ros2 run comres_ros comres_discover` finds their topics.

**Frames.** With one robot nothing has to be set: the world frame and the Nav2 goal frame are the
robot's SLAM map frame (from its map messages: `{robot}/map` through the link) and the base frame
is the first that exists of `{robot}/base_footprint`, `{robot}/base_link`, `base_footprint`,
`base_link`. With several robots, set `world_frame` to a frame every robot's SLAM map frame is
placed in (e.g. static transforms from measured start poses). `~/start` refuses, saying what is
missing, until every robot has a map, its base and goal-frame transforms, and Nav2.

## The planning grid, and "north"

COMRES plans on a grid of `cell_size` cells, `grid_width` x `grid_height` metres, centred on the
team's start (`grid_auto_center`). Its parameters are in cells and were tuned for a 20-cell
sensing range, so set `cell_size = (the robots' useful sensing range) / 20` (5 m -> 0.25 m). A grid
cell is a wall if any SLAM pixel in it is occupied (>= 65; pixels grown by one so thin walls stay
closed at any angle), free if at least half of it is free (<= 25), else unknown. The node warns
when a SLAM map reaches outside the grid.

COMRES plans for a point robot, as in the simulator, so a one-cell gap between obstacles is a
passage to it and a frontier behind the gap a target; Nav2's robot needs `robot_radius` of room on
each side. `robot_clearance` (0.25 m: `nav2_go2.yaml`'s robot_radius) makes a wall of every cell
whose centre is nearer than that to an occupied SLAM pixel, as Nav2's inscribed radius rules out
those places for the robot's centre: on the Go2's map, every frontier COMRES can then reach, Nav2
can plan to (without it, 92 %).

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
`detection_frames` times in a row (1 for the Go2's YOLO, which reports each object once); any
mission ends on `~/stop` or after `max_mission_seconds`.

## Same as the simulation

`comres_core.online.Mission` runs the simulator's COMRESStrategy unchanged, step for step: one tick
is one simulator step (sensing, the "seen = reached" check, the selector, the move, the arrival
test, then the plan delivery and every T_H steps the merge and the question). What differs is only
what the real world makes different:

* the planner's latency is real (its question runs in the background; the plan takes effect at
  the first tick after the answer);
* motion is Nav2's. The robot is not a point: cells too near obstacles are walls
  (`robot_clearance`, above). A goal is given up as unreachable when Nav2 aborts it; Nav2 runs
  COMRES's goals with `comres_bt.xml` (one retry, no spinning), so that takes seconds, not minutes.
  A goal Nav2 gives up at the target counts as reached, as the simulator never gives up a target
  it is at; a goal's outcome only counts for the target it was sent for. `goal_clearance` (0 with
  `robot_clearance`) moves goals up to 0.5 m off walls when Nav2 would refuse them;
* a robot's map between merges is its live SLAM map, with the last merge filling what it does not
  know; the 2-cell disc around the robot counts as seen where neither knows it.

In the simulator the robot moves one cell per step: with `tick_seconds` 1, a Nav2 speed of
`cell_size` m/s keeps that ratio (faster is fine, it only moves more per step).

`tools/shadow_check.py` checks this on recorded COMRES-VLM runs: the simulator replays a run (the
VLM's recorded replies, the recorded delays) while a Mission gets, every step, what the ROS node
would get from the robots; the Mission must choose the same target for every robot at every step,
merge the same maps and deliver the same plans at the same steps. On 2026-10-01 all 1,100
recorded runs (search and explore, every map size) were the same at every step.

`tools/fake_dog.py` is a fake Go2 (a map revealed by a 360-degree scan as "SLAM", a NavigateToPose
server, a detector) to run the node without hardware;
`tools/fake_dog_test.sh` runs a mission against it and passes when the target is found.

## Logs

Each mission writes `run_dir/<start time>/`: `meta.json` (settings, start cells and poses, the
algorithm's commit), `events.jsonl` (merges, plans, targets, goals, seen / reached / failed,
the end and why), `plans.jsonl` (every plan with the VLM's reply), `plans/plan_<step>.png` and
`.txt` (the picture and text the VLM was given), `poses.jsonl` (every tick: cells, poses, targets)
and `maps/merge_<step>.npz` (each merged map).

## Tests

* `python -m pytest src/comres_core/test` (no ROS): the grid conversion (thin walls at any angle,
  uncertain cells, rotated frames, robot clearance) and missions on a small simulated world (two robots, one
  robot, a slow VLM that must not stop the robots, a failed goal that must be avoided, goal outcomes for an
  older target ignored);
* `python tools/shadow_check.py <results .../comres.jsonl> --sim <COMRES-VLM>`: the same decisions
  as the simulator, step for step (above);
* `tools/fake_dog_test.sh` (ROS 2 Jazzy): the node end to end against the fake dog;
* `tools/dog_bridge_test.sh` (ROS 2 Jazzy): the same through the link, the fake dog and its bridge
  in one ROS domain, dog_link and the node in another;
* `python3 tools/vlm_check.py`: the VLM server answers the node's planning question (three plans);
* `tools/fake_dog_manual_test.sh` (ROS 2 Jazzy): `~/start` before the robot is up, after, twice,
  and after the end; `~/stop` before any mission.
