#!/bin/bash
# The central computer's side of COMRES in one command (ROS 2 Jazzy; this workspace built): the VLM
# on the GPU server (started there if it is not running), the SSH tunnel to it, and the COMRES node
# with its link to the dogs. Leave it running and start missions on a dog's desktop:
# comres_explore, or comres_search "<hint>". The node starts each one when the dog is ready, and
# the next after it ends. Ctrl-C stops it all, and the VLM if this script started it.
#   tools/comres_central.sh [config.yaml]
# COMRES_VLM_SSH: the GPU server (default ruiyangw@10.237.196.234), where COMRES-VLM's serve_vlm.sh
# serves the VLM on 127.0.0.1:18765.
WS=$(cd "$(dirname "$0")/.." && pwd)
CONFIG=${1:-$WS/install/comres_ros/share/comres_ros/config/comres.yaml}
SERVER=${COMRES_VLM_SSH:-ruiyangw@10.237.196.234}
PORT=18765
URL=http://127.0.0.1:$PORT/v1
source /opt/ros/jazzy/setup.bash
source "$WS/install/setup.bash"

STARTED_VLM=0
TUNNEL=
cleanup() {
    [ -n "$TUNNEL" ] && kill "$TUNNEL" 2>/dev/null
    if [ "$STARTED_VLM" = 1 ]; then
        ssh -o BatchMode=yes "$SERVER" 'tmux kill-session -t comres_vlm' && echo "stopped the VLM on $SERVER"
    fi
}
trap cleanup EXIT

if ! ssh -o BatchMode=yes -o ConnectTimeout=10 "$SERVER" 'tmux has-session -t comres_vlm 2>/dev/null'; then
    echo "starting the VLM on $SERVER (it takes about 6 minutes to load)"
    ssh -o BatchMode=yes "$SERVER" \
        'tmux new-session -d -s comres_vlm "cd ~/COMRES-VLM && source env.sh && ./serve_vlm.sh 2>&1 | tee ~/COMRES-VLM/tmp/serve_vlm.log"' || exit 1
    STARTED_VLM=1
fi
if ! ss -ltn | grep -q ":$PORT "; then
    ssh -N -o BatchMode=yes -o ExitOnForwardFailure=yes -o ServerAliveInterval=15 -L "$PORT:127.0.0.1:$PORT" "$SERVER" &
    TUNNEL=$!
fi
echo -n "waiting for the VLM at $URL "
until curl -s -m 5 "$URL/models" | grep -q '"id"'; do
    if [ -n "$TUNNEL" ] && ! kill -0 "$TUNNEL" 2>/dev/null; then
        echo; echo "the SSH tunnel to $SERVER ended" >&2
        exit 1
    fi
    echo -n .
    sleep 10
done
echo " up"
ros2 launch comres_ros comres.launch.py config:="$CONFIG" vlm_base_url:="$URL"
