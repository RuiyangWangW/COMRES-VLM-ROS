# COMRES on Go2 robots (ROS 2 Jazzy)

## 1. GPU server: start the VLM

```bash
cd ~/COMRES-VLM && source env.sh && ./serve_vlm.sh
```

## 2. Each dog

Run its driver, SLAM Toolbox, Nav2 and the YOLO node, with the same `ROS_DOMAIN_ID` as the
central computer. In the SLAM Toolbox parameter file, set the map to update every second (the
default is 5 s), then start SLAM Toolbox with that file:

```yaml
slam_toolbox:
  ros__parameters:
    map_update_interval: 1.0
```

```bash
ros2 launch slam_toolbox online_async_launch.py slam_params_file:=<that file>
```

## 3. Central computer: install (once)

```bash
source /opt/ros/jazzy/setup.bash
git clone git@github.com:RuiyangWangW/COMRES-ROS.git ~/comres_ros_ws
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

# what the dog publishes, and whether the VLM answers
ros2 run comres_ros comres_discover --vlm http://127.0.0.1:18765/v1
```

Copy the names it prints into `src/comres_ros/config/comres.yaml`, and set there:

| setting | what to put |
|---|---|
| `robot_ids`, `map_topic`, `nav_action`, `detection_topic`, `detection_type` | from `comres_discover` |
| `target_classes` | the YOLO class of the target, e.g. `["backpack"]` |
| `hint` | the search hint |
| `cell_size` | the dog's useful sensing range / 20 (5 m -> 0.25) |
| `grid_width`, `grid_height` | the area in metres, with margin (the dog starts in the middle) |

Start each dog facing east (the hint's east), or set `grid_yaw_deg`.

## 5. Run a mission

```bash
# terminal 1
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
```

How it works and how it was checked: [docs/how_it_works.md](docs/how_it_works.md).
