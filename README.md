# COMRES on Go2 robots (ROS 2 Jazzy)

Three machines: the GPU server runs the VLM, the central computer runs COMRES, each dog runs its
driver, SLAM, YOLO and Nav2 and a bridge to the central computer (in its own ROS 2: Foxy on the Go2).

## 1. Central computer: install (once)

```bash
source /opt/ros/jazzy/setup.bash
git clone git@github.com:RuiyangWangW/COMRES-VLM-ROS.git ~/comres_ros_ws
cd ~/comres_ros_ws
rosdep install --from-paths src --ignore-src -y
pip install --user --break-system-packages openai
colcon build --symlink-install
```

Set in `src/comres_ros/config/comres.yaml`:

| setting | what to put |
|---|---|
| `robot_ids`, and `dog_link`'s `robots` and `hosts` | the dogs' names, and where each one's bridge listens (`address:9100`; its Tailscale address works) |
| `target_classes` | the YOLO class searched for when a dog does not name one, e.g. `["backpack"]` |
| `cell_size` | the dog's useful sensing range / 20 (5 m -> 0.25) |
| `grid_width`, `grid_height` | the area in metres, with margin (the dog starts in the middle) |
| `robot_clearance` | Nav2's `robot_radius` (0.25 m in `nav2_go2.yaml`) |

## 2. Each dog: install (once)

Copy `dog_bridge.py`, `dog_wire.py` and `comres_bt.xml` from `src/comres_ros/comres_ros/` into
`~/comres_bridge/` on the dog. On the Go2, `start_comres.sh` of I3T_CPSL_UnitreeGo2_codebase
(aliases `comres_explore`, `comres_search`) starts the rest; Nav2 uses MPPI, built once with
`colcon build --packages-select nav2_mppi_controller --cmake-args -DBUILD_TESTING=OFF`.

## 3. Central computer: every session

```bash
cd ~/comres_ros_ws
tools/comres_central.sh
```

It starts the VLM on the GPU server if it is not running (COMRES-VLM's `serve_vlm.sh`; about 6
minutes; `COMRES_VLM_SSH` names the server), opens the SSH tunnel to it, then runs the COMRES node
and its link to the dogs. Leave it running; Ctrl-C stops it all (and the VLM, if it started it).

## 4. Run a mission

On the dog's desktop (e.g. through NoMachine):

```bash
comres_explore                                    # explore, no prior information
comres_search "<hint>" [target class]             # search, with the prior in plain words, e.g.
comres_search "The backpack is in the room north of the start." backpack
```

One Terminator window starts the driver, sensors, 3D SLAM, Nav2, RViz and the bridge; the bridge
asks the central computer for the mission once Nav2 is up and there is a map, and COMRES starts it.
A search ends by itself when the YOLO sees the target;
`ros2 service call /comres_node/stop std_srvs/srv/Trigger` on the central computer stops any
mission (and cancels every goal). For the next mission, close the dog's window and run the command
again; the central side keeps running. Start each dog facing east (the hint's east), or set
`grid_yaw_deg`. Logs: `~/comres_runs/<start time>/`.

Without the dog's commands: run `dog_bridge.py` on the dog by hand and
`ros2 service call /comres_node/start std_srvs/srv/Trigger` starts the config's `task` and `hint`.

## Without a dog

```bash
python3 -m pytest src/comres_core/test
tools/fake_dog_test.sh ~/COMRES-VLM/benchmark/maps/bench_medium/bench_medium_branching_1.npy 50 2 59 69 240
tools/dog_bridge_test.sh        # through the dog link: two missions the bridge asks for
python3 tools/vlm_check.py      # the VLM answers a real planning question (through the tunnel)
```

Dry run with random plans instead of the VLM (no GPU server needed):

```bash
ros2 run comres_ros comres_node --ros-args --params-file src/comres_ros/config/comres.yaml -p fake_vlm:=true
```

How it works and how it was checked: [docs/how_it_works.md](docs/how_it_works.md).
