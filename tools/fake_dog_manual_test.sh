#!/bin/bash
# ~/start and ~/stop against tools/fake_dog.py (no hardware): source a ROS 2 Jazzy setup first.
#   tools/fake_dog_manual_test.sh <world.npy> <start_row> <start_col> <target_row> <target_col> [seconds]
# 1. ~/stop before any mission: the node answers that none ran and will not start one; ~/start
#    is then refused.
# 2. A second node: ~/start before the dog is up is refused (not ready); after, it starts;
#    ~/start again answers "already running"; once the target is found, ~/start answers
#    "mission over".
set -e
WS=$(cd "$(dirname "$0")/.." && pwd)
WORLD=$1; SR=$2; SC=$3; TR=$4; TC=$5; SECS=${6:-300}
OUT=${COMRES_FAKE_DOG_OUT:-/tmp/comres_fake_dog_manual}
PY=${PYTHON:-python3}
mkdir -p "$OUT"
export PYTHONPATH=$PYTHONPATH:$WS/src/comres_core:$WS/src/comres_ros
cat > "$OUT/params.yaml" <<YAML
comres_node:
  ros__parameters:
    robot_ids: ["go2_1"]
    task: "search"
    T_H: 30
    fake_vlm: true
    auto_start: false
    run_dir: "$OUT/runs"
    max_mission_seconds: $((SECS - 15))
YAML
call() { $PY "$WS/tools/call_service.py" "$1" | tee -a "$OUT/calls.log"; }
# exec: $! is then the node itself. SIGTERM, as a background job ignores SIGINT.
node() { exec $PY -c "from comres_ros.comres_node import main; main()" --ros-args --params-file "$OUT/params.yaml"; }
: > "$OUT/calls.log"
FAIL=0
expect() { if grep -q "$1" "$OUT/calls.log"; then echo "ok: $1"; else echo "MISSING: $1"; FAIL=1; fi; }

# 1. stop before start
node > "$OUT/node1.log" 2>&1 &
N1=$!
sleep 4
call /comres_node/stop
call /comres_node/start
kill -TERM $N1; wait $N1 || true
expect "no mission was running"
expect "mission over"

# 2. start before the dog, after it, again, and after the end
: > "$OUT/calls.log"
node > "$OUT/node2.log" 2>&1 &
N2=$!
sleep 4
call /comres_node/start
$PY "$WS/tools/fake_dog.py" --world "$WORLD" --start "$SR" "$SC" --target "$TR" "$TC" --seconds "$SECS" > "$OUT/dog.log" 2>&1 &
DOG=$!
sleep 5
call /comres_node/start
call /comres_node/start
for _ in $(seq 1 $((SECS / 5))); do
  grep -q "mission ended" "$OUT/node2.log" && break
  sleep 5
done
call /comres_node/start
kill -TERM $N2; wait $N2 || true
kill -TERM $DOG 2>/dev/null; wait $DOG || true
expect "success=False message='not ready"
expect "message='started'"
expect "already running"
expect "mission over"
grep -q "target found" "$OUT/node2.log" && echo "ok: target found" || { echo "MISSING: target found"; FAIL=1; }
[ $FAIL = 0 ] && echo PASS || { echo "FAIL (see $OUT)"; exit 1; }
