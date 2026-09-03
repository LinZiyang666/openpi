#!/bin/bash
# launch_client.sh <suite> <shard> <port> <task_ids...>  (server already running)
set -u
SUITE=$1; K=$2; PORT=$3; shift 3; TASKS="$*"
export PATH=/usr/local/bin:/usr/bin:/bin
export HOME=/home/weiland
R=/data/openpi_dispatch
B=/tmp/dsp_shared/rit_pareto/$SUITE
SLOG=/tmp/rit_srv_${SUITE}_${K}.log
CLOG=/tmp/rit_cli_${SUITE}_${K}.log
rm -f $CLOG
tmux new -s ritc_${SUITE}_$K -d "cd $R && export HOME=/home/weiland && until grep -q 'server listening on' $SLOG 2>/dev/null; do sleep 3; done; MUJOCO_EGL_DEVICE_ID=0 PYTHONPATH=. /home/weiland/miniconda3/bin/conda run --no-capture-output -p /home/weiland/libero_sim python examples/libero/main.py --host 127.0.0.1 --port $PORT --task-suite-name $SUITE --num-trials-per-task 15 --num-workers 1 --init-states-dir $B/shadow_pool --cohort-plan $B/cohort_plan.json --task-ids $TASKS 2>&1 | tee $CLOG; echo RIT_CLIENT_EXIT=\${PIPESTATUS[0]} | tee -a $CLOG; echo RIT_CLIENT_DONE | tee -a $CLOG"
echo "client shard $K port $PORT tasks [$TASKS]"
