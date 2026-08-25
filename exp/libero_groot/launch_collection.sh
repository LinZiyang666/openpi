#!/usr/bin/env bash
# Bring up an N-lane GR00T/LIBERO collection fleet on weilandserver.
#
# Collection is structurally single-connection (the collector hangs per-episode
# state off one runner), so parallelism is N server processes each driven by one
# client process. Lanes are carved with ``--episode-filter``: main.py only
# *skips* filtered-out episodes, never re-indexes, so every lane reproduces the
# init state and the global episode id an unsharded run would have used.
#
# Re-running is a resume: shards are recomputed against the HDF5 already on
# disk, so a crashed or re-balanced run picks up exactly what is missing.
#
# usage: launch_collection.sh <suite> <checkpoint> <out-dir> [lanes] [base-port]
set -eu
export HOME=/home/weiland

SUITE=${1:?suite}
CKPT=${2:?checkpoint}
OUT=${3:?out dir}
LANES=${4:-6}
BASE=${5:-8030}

PY=/home/weiland/gr00t_n15_venv/.venv/bin/python
REPO=/home/weiland/openpi
SHARDS=/data/libero_cache/shards/$SUITE
INIT=$REPO/exp/common/data/db_init/libero/$SUITE
GR00T_PATH=/home/weiland/gr00t_n15:/home/weiland/gr00t_n15/examples/Libero:$REPO:$REPO/src

[ -d "$INIT" ] || { echo "no B-pool init dir: $INIT"; exit 1; }
if ls "$INIT"/*.pruned_init >/dev/null 2>&1; then
  # _load_init_states prefers .pruned_init, so one stray file silently swaps
  # the frozen test set in for the collection pool.
  echo "REFUSING: $INIT contains .pruned_init (A pool) files"; exit 1
fi

echo "== tearing down any previous fleet =="
tmux kill-session -t lbwatch 2>/dev/null || true
for s in $(tmux ls 2>/dev/null | grep -oE '^(lbsrv|lbrun)[0-9]+'); do tmux kill-session -t "$s" 2>/dev/null || true; done
sleep 8

mkdir -p "$OUT" "$SHARDS"
# The watchdog heals whatever this file describes; without it a self-healer
# would have to guess the suite/checkpoint from a fleet that is already gone.
cat > /data/libero_cache/current_run.env <<EOF
SUITE=$SUITE
CKPT=$CKPT
OUT=$OUT
LANES=$LANES
BASE=$BASE
SHARDS=$SHARDS
EOF

echo "== starting $LANES servers =="
for i in $(seq 0 $((LANES-1))); do
  P=$((BASE+i))
  tmux new -s "lbsrv$P" -d "cd $REPO && PYTHONPATH=$GR00T_PATH OPENPI_MONITOR_LEVEL=BASIC $PY exp/libero_groot/serve_groot_libero.py --checkpoint $CKPT --port $P --collect-hdf5 $OUT --experiment groot_$SUITE 2>&1 | tee /tmp/lbsrv$P.log"
  sleep 10
done
for _ in $(seq 1 40); do
  up=0
  for i in $(seq 0 $((LANES-1))); do ss -tln | grep -q ":$((BASE+i)) " && up=$((up+1)); done
  [ "$up" = "$LANES" ] && break
  sleep 10
done
echo "servers listening: $up/$LANES"
[ "$up" = "$LANES" ] || { echo "ABORT: servers did not come up"; exit 1; }

echo "== sharding remaining work =="
cd "$REPO"
$PY exp/libero_groot/make_shards.py --num-tasks 10 --trials 50 --lanes "$LANES" \
  --done-dir "$OUT" --out-dir "$SHARDS" --prefix "$SUITE"

echo "== launching $LANES lanes =="
for i in $(seq 0 $((LANES-1))); do
  P=$((BASE+i))
  F=$SHARDS/${SUITE}_lane$i.json
  TASKS=$($PY -c "import json,sys;e=json.load(open(sys.argv[1]));print(' '.join(str(t) for t in sorted({x['task_id'] for x in e})))" "$F")
  [ -z "$TASKS" ] && { echo "lane$i: nothing left, skipping"; continue; }
  echo "lane$i -> port $P tasks=[$TASKS]"
  tmux new -s "lbrun$i" -d "cd $REPO && MUJOCO_EGL_DEVICE_ID=0 PYTHONPATH=. /home/weiland/miniconda3/bin/conda run -p /home/weiland/libero_sim --no-capture-output python examples/libero/main.py --host 127.0.0.1 --port $P --task-suite-name $SUITE --task-ids $TASKS --num-trials-per-task 50 --num-workers 1 --resize-size 256 --replan-steps 5 --init-states-dir $INIT --cuda-visible-devices 0 --episode-filter $F --save-episode-results --episode-results-path $OUT/results_lane$i.json 2>&1 | tee /tmp/lbrun$i.log"
  sleep 5
done
tmux new -s lbwatch -d "bash $REPO/exp/libero_groot/watchdog.sh 2>&1 | tee -a /tmp/lbwatch.log"
echo "FLEET-UP $SUITE lanes=$(tmux ls | grep -c '^lbrun') watchdog=$(tmux has-session -t lbwatch 2>/dev/null && echo up || echo DOWN) $(date +%H:%M:%S)"
