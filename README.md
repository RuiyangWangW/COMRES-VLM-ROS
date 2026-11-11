# COMRES on Go2 robots (ROS 2 Jazzy)

## 1. GPU server: start the VLM

```bash
cd ~/COMRES-VLM && source env.sh && ./serve_vlm.sh
```

## 2. Each dog

Run its driver, sensors, SLAM, the YOLO node and Nav2; on the Go2, `slam3d` on its desktop
(I3T_CPSL_UnitreeGo2_codebase), then `ros2 launch cpsl_nav nav2_go2.launch.py`. Then the link to
the central computer, in the dog's own ROS 2 (Foxy on the Go2): copy `dog_bridge.py`,
`dog_wire.py` and `comres_bt.xml` from `src/comres_ros/comres_ros/` into one folder on the dog
(once), and run

```bash
python3 ~/comres_bridge/dog_bridge.py --robot go2_1
```

It listens on port 9100; the central computer connects to it.

## 3. Central computer: install (once)

```bash
source /opt/ros/jazzy/setup.bash
git clone git@github.com:RuiyangWangW/COMRES-VLM-ROS.git ~/comres_ros_ws
cd ~/comres_ros_ws
rosdep install --from-paths src --ignore-src -y
pip install --user --break-system-packages openai
colcon build --symlink-install
```

## 4. Central computer: every session

```bash
source /opt/ros/jazzy/setup.bash && source ~/comres_ros_ws/install/setup.bash
cd ~/comres_ros_ws

# the VLM, through a tunnel to the GPU server (leave it running)
ssh -N -L 18765:127.0.0.1:18765 <user>@<gpu-server> &

# whether the VLM answers a real planning question
python3 tools/vlm_check.py
```

Set in `src/comres_ros/config/comres.yaml`:

| setting | what to put |
|---|---|
| `robot_ids`, and `dog_link`'s `robots` and `hosts` | the dogs' names, and where each one's bridge listens (`address:9100`; its Tailscale address works) |
| `target_classes` | the YOLO class of the target, e.g. `["backpack"]` |
| `hint` | the search hint |
| `cell_size` | the dog's useful sensing range / 20 (5 m -> 0.25) |
| `grid_width`, `grid_height` | the area in metres, with margin (the dog starts in the middle) |
| `robot_clearance` | Nav2's `robot_radius` (0.25 m in `nav2_go2.yaml`) |

Start each dog facing east (the hint's east), or set `grid_yaw_deg`.

## 5. Run a mission

```bash
# terminal 1: the COMRES node and its link to the dogs
ros2 launch comres_ros comres.launch.py config:=$HOME/comres_ros_ws/src/comres_ros/config/comres.yaml

# terminal 2
ros2 service call /comres_node/start std_srvs/srv/Trigger    # "not ready: ..." says what is missing
ros2 service call /comres_node/stop std_srvs/srv/Trigger     # stop at any time (cancels every goal)
```

The search ends by itself when the YOLO sees the target. For the next mission, restart terminal 1.
Logs: `~/comres_runs/<start time>/`.

Dry run with random plans instead of the VLM (no GPU server needed):

```bash
ros2 run comres_ros comres_node --ros-args --params-file src/comres_ros/config/comres.yaml -p fake_vlm:=true
```

## Without a dog

```bash
python3 -m pytest src/comres_core/test
tools/fake_dog_test.sh ~/COMRES-VLM/benchmark/maps/bench_medium/bench_medium_branching_1.npy 50 2 59 69 240
tools/dog_bridge_test.sh        # the same through the dog link (bridge and link in separate ROS domains)
```

How it works and how it was checked: [docs/how_it_works.md](docs/how_it_works.md).
