#!/usr/bin/env bash
# Drive the RIT calibration cohort from a sim box: one client process per task.
#
# One process per task rather than one sharded pool: a LIBERO client process is
# one websocket connection (--num-workers is an in-process EGL cap, not a
# connection count), and the cohort is exactly 15 episodes on each of 10 tasks,
# so the task axis is already the natural even shard. Processes are dealt round
# robin over the lane's servers.
#
# The init pool is the sampled cohort, so the client's trial index is the
# position inside the sampled file -- 15 trials covers fit+cal for that task
# and nothing else.
#
# usage: launch_shadow_clients.sh <suite> <host> <repo> <python> [base-port] [n-servers] [trials]
set -eu

SUITE=${1:?suite}
HOST=${2:?server host}
REPO=${3:?repo}
PY=${4:?python}
BASE=${5:-23210}
NSRV=${6:-4}
TRIALS=${7:-15}

POOL=$REPO/exp/libero_groot/data/rit/shadow/$SUITE/shadow_pool
[ -d "$POOL" ] || { echo "missing cohort pool $POOL"; exit 1; }

NGPU=$(nvidia-smi --query-gpu=index --format=csv,noheader | wc -l)
echo "== $NGPU GPUs, $NSRV servers, 10 client processes =="

for t in $(seq 0 9); do
  tmux kill-session -t "lgcli$t" 2>/dev/null || true
done
sleep 3

for t in $(seq 0 9); do
  P=$((BASE + t % NSRV))
  G=$((t % NGPU))
  tmux new -s "lgcli$t" -d "cd $REPO && \
    HOME=$HOME LIBERO_CONFIG_PATH=$HOME/.libero \
    MUJOCO_EGL_DEVICE_ID=$G PYTHONPATH=$REPO:$REPO/src \
    $PY examples/libero/main.py \
      --host $HOST --port $P \
      --task-suite-name $SUITE \
      --num-trials-per-task $TRIALS --num-workers 1 \
      --init-states-dir $POOL \
      --task-ids $t --replan-steps 5 --resize-size 256 \
      2>&1 | tee /tmp/lgcli$t.log; \
    echo LGCLI_EXIT=\${PIPESTATUS[0]} | tee -a /tmp/lgcli$t.log"
  sleep 2
done

sleep 20
echo "== launched =="
tmux ls 2>/dev/null | grep -c '^lgcli' | sed 's/^/client sessions: /'
