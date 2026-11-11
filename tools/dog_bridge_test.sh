#!/bin/bash
# The COMRES node against tools/fake_dog.py through the dog link (no hardware; ROS 2 Jazzy sourced):
# the fake dog and dog_bridge.py run in one ROS domain, dog_link and comres_node in another, so all
# that reaches the node crosses the TCP link, as from the Go2 (ROS 2 Foxy).
#   tools/dog_bridge_test.sh [seconds]
# The world is the tests' corridor with four rooms (start at its west end, the target in the
# north-east room); the VLM's fake mode; writes to /tmp/comres_dog_bridge/.
# Pass: the node's log ends with "mission ended ...: target found".
set -e
WS=$(cd "$(dirname "$0")/.." && pwd)
SECS=${1:-240}
OUT=${COMRES_DOG_BRIDGE_OUT:-/tmp/comres_dog_bridge}
PY=${PYTHON:-python3}
PORT=9107
mkdir -p "$OUT"
export PYTHONPATH=$PYTHONPATH:$WS/src/comres_core:$WS/src/comres_ros:$WS/src/comres_core/test
export ROS_AUTOMATIC_DISCOVERY_RANGE=LOCALHOST
$PY -c "import numpy as np; from test_online import world; np.save('$OUT/world.npy', world())"
cat > "$OUT/params.yaml" <<YAML
comres_node:
  ros__parameters:
    robot_ids: ["go2_1"]
    task: "search"
    hint: "The target is in the far north-east room."
    T_H: 30
    detection_type: "std_msgs/String"
    detection_frames: 1
    fake_vlm: true
    auto_start: true
    run_dir: "$OUT/runs"
    max_mission_seconds: $((SECS - 20))
dog_link:
  ros__parameters:
    robots: ["go2_1"]
    hosts: ["127.0.0.1:$PORT"]
YAML
# the dog: domain 61
ROS_DOMAIN_ID=61 $PY "$WS/tools/fake_dog.py" --robot sim --world "$OUT/world.npy" --start 19 2 --target 30 52 \
    --seconds "$SECS" > "$OUT/dog.log" 2>&1 &
DOG=$!
ROS_DOMAIN_ID=61 timeout "$SECS" $PY -c "from comres_ros.dog_bridge import main; main()" --port $PORT \
    --map-topic /sim/map --nav-action /sim/navigate_to_pose --det-topic /sim/detections --det-type detection2d \
    > "$OUT/bridge.log" 2>&1 &
BRIDGE=$!
# the central computer: domain 62
ROS_DOMAIN_ID=62 timeout "$SECS" $PY -c "from comres_ros.dog_link import main; main()" \
    --ros-args --params-file "$OUT/params.yaml" > "$OUT/link.log" 2>&1 &
LINK=$!
sleep 3
ROS_DOMAIN_ID=62 timeout $((SECS - 5)) $PY -c "from comres_ros.comres_node import main; main()" \
    --ros-args --params-file "$OUT/params.yaml" > "$OUT/node.log" 2>&1 || true
kill $LINK $BRIDGE 2>/dev/null || true
wait $DOG $LINK $BRIDGE 2>/dev/null || true
tail -1 "$OUT/dog.log"
grep -c "goal [0-9]*:" "$OUT/bridge.log" | sed 's/^/goals through the bridge: /'
if grep -q "target found" "$OUT/node.log"; then echo "PASS: $(grep 'mission ended' "$OUT/node.log" | tail -1)"; else echo "FAIL (see $OUT/node.log)"; exit 1; fi
