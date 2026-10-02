#!/bin/bash
# The COMRES node against tools/fake_dog.py (no hardware): source a ROS 2 Jazzy setup first.
#   tools/fake_dog_test.sh <world.npy> <start_row> <start_col> <target_row> <target_col> [seconds] [hint]
# Uses the VLM's fake mode (random valid plans), T_H 30, and writes to /tmp/comres_fake_dog/.
# Pass: the node's log ends with "mission ended ...: target found".
set -e
WS=$(cd "$(dirname "$0")/.." && pwd)
WORLD=$1; SR=$2; SC=$3; TR=$4; TC=$5; SECS=${6:-300}; HINT=${7:-""}
OUT=${COMRES_FAKE_DOG_OUT:-/tmp/comres_fake_dog}
PY=${PYTHON:-python3}
mkdir -p "$OUT"
export PYTHONPATH=$PYTHONPATH:$WS/src/comres_core:$WS/src/comres_ros
cat > "$OUT/params.yaml" <<YAML
comres_node:
  ros__parameters:
    robot_ids: ["go2_1"]
    task: "search"
    hint: "$HINT"
    T_H: 30
    fake_vlm: true
    auto_start: true
    run_dir: "$OUT/runs"
    max_mission_seconds: $((SECS - 15))
YAML
$PY "$WS/tools/fake_dog.py" --world "$WORLD" --start "$SR" "$SC" --target "$TR" "$TC" --seconds "$SECS" > "$OUT/dog.log" 2>&1 &
DOG=$!
sleep 3
timeout $((SECS - 5)) $PY -c "from comres_ros.comres_node import main; main()" --ros-args --params-file "$OUT/params.yaml" > "$OUT/node.log" 2>&1 || true
wait $DOG || true
tail -1 "$OUT/dog.log"
if grep -q "target found" "$OUT/node.log"; then echo "PASS: $(grep 'mission ended' "$OUT/node.log" | tail -1)"; else echo "FAIL (see $OUT/node.log)"; exit 1; fi
