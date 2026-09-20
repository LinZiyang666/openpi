#!/bin/bash
# Serial ladder driver on weilandserver: for each "<suite>:<k>" in the list, launch the shards and wait until
# every shard JSON is complete (or its tmux session is gone), then move on. Markers: LADDER <suite> k=<k> UP/DONE/FAIL.
# usage: ladder_wls.sh "libero_10:5 libero_10:3 ..." [n_shards=5] [seed=195]
set -u
POINTS=${1:?points}; NS=${2:-5}; SEED=${3:-195}
HERE=$(cd "$(dirname "$0")" && pwd)
export PATH=/home/weiland/.local/bin:/usr/local/bin:/usr/bin:/bin
for P in $POINTS; do
  SUITE=${P%%:*}; K=${P##*:}
  bash "$HERE/launch_point_wls.sh" "$SUITE" "$K" "$NS" "$SEED" > /dev/null
  echo "LADDER $SUITE k=$K UP $(date +%H:%M:%S)"
  while :; do
    done_n=0; live=0
    for i in $(seq 0 $((NS-1))); do
      f=/data/cosmos/results/$SUITE/k$K/shard$i.json
      [ -f "$f" ] && grep -q '"complete": true' "$f" && done_n=$((done_n+1))
      tmux has-session -t "cosmos_${SUITE}_k${K}_s$i" 2>/dev/null && live=$((live+1))
    done
    [ "$done_n" -eq "$NS" ] && break
    if [ "$live" -eq 0 ]; then echo "LADDER $SUITE k=$K FAIL complete=$done_n/$NS $(date +%H:%M:%S)"; exit 1; fi
    sleep 60
  done
  echo "LADDER $SUITE k=$K DONE $(date +%H:%M:%S)"
done
echo "LADDER FINISHED $(date +%H:%M:%S)"
