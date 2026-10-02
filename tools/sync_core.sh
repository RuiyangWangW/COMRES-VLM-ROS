#!/bin/bash
# Copy the COMRES algorithm, unchanged, from the COMRES-VLM repo into comres_core/algo and record the
# commit it came from (algo/CORE_VERSION). Usage: tools/sync_core.sh [path to COMRES-VLM]
# (default: the folder this workspace sits in). Refuses a COMRES-VLM tree with uncommitted changes
# to these files, so CORE_VERSION always names exactly what was copied.
set -e
WS=$(cd "$(dirname "$0")/.." && pwd)
SRC=${1:-$(dirname "$WS")}
DST=$WS/src/comres_core/comres_core/algo
FILES="comres_strategy.py comres_planner.py comres_points.py units.py belief.py selector.py params.py
grid_map.py vlm_client.py waypoint_strategy.py experiment_config.py"
cd "$SRC"
for f in $FILES; do [ -f "$f" ] || { echo "missing $SRC/$f"; exit 1; }; done
if [ -n "$(git status --porcelain -- $FILES)" ]; then
  echo "uncommitted changes in $SRC to the algorithm files; commit them first:"; git status --short -- $FILES; exit 1
fi
mkdir -p "$DST"
for f in $FILES; do cp "$f" "$DST/$f"; done
{ echo "commit $(git rev-parse HEAD)"; echo "date   $(git log -1 --format=%cI)"; echo "files  $(echo $FILES | tr '\n' ' ')"; } > "$DST/CORE_VERSION"
echo "copied from $SRC at $(git rev-parse --short HEAD) into $DST"
