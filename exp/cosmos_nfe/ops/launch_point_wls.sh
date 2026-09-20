#!/bin/bash
# Launch all shards of one (suite, k) point: <suite> <k> [n_shards=5] [seed=195] [trials=50]
# Tasks 0..9 are dealt round-robin over the shards (shard i gets i, i+S, ...).
set -u
SUITE=${1:?suite}; K=${2:?k}; NS=${3:-5}; SEED=${4:-195}; TRIALS=${5:-50}
HERE=$(cd "$(dirname "$0")" && pwd)
for i in $(seq 0 $((NS-1))); do
  TIDS=$(seq $i $NS 9 | paste -sd, -)
  bash "$HERE/run_shard_wls.sh" "$SUITE" "$K" "$i" "$TIDS" "$SEED" "$TRIALS"
  sleep 20   # stagger model loads
done
tmux ls | grep -c "cosmos_${SUITE}_k${K}_"
