#!/bin/bash
# timan107 ladder: for each "<suite>:<k>", run 10 client shards (one task each, GPUs 0-7 round-robin), shard i -> h100 replica port BASE+(i % NREP).
# usage: ladder_t107.sh "libero_spatial:5 libero_spatial:3 ..." [seed=195]
set -u
POINTS=${1:?points}; SEED=${2:-195}; NS=10; NREP=${COSMOS_NREP:-10}; BASE=${COSMOS_BASE_PORT:-23240}; HOST=${COSMOS_HOST:-149.165.153.233}
HERE=$(cd "$(dirname "$0")" && pwd)
ROOT=${COSMOS_RESULTS_ROOT:-/scratch/zixuans8/cosmos/results}
for P in $POINTS; do
  SUITE=${P%%:*}; K=${P##*:}
  for i in $(seq 0 $((NS-1))); do COSMOS_REMOTE=ws://$HOST:$((BASE + i % NREP)) bash "$HERE/run_shard_t107.sh" "$SUITE" "$K" "$i" "$i" $((i % 8)) "$SEED" > /dev/null; sleep 8; done
  echo "LADDER $SUITE k=$K UP $(date +%H:%M:%S)"
  while :; do
    done_n=0; live=0
    for i in $(seq 0 $((NS-1))); do
      f=$ROOT/$SUITE/k$K/shard$i.json
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
