#!/bin/bash
# launch_shadow.sh <suite> <shard> <port> <task_ids...>
# One collect server (single replica, non-concurrent) + one serial LIBERO client
# restricted to the given task ids; H5 -> /tmp/dsp_shared/rit_pareto/<suite>/h5/shard<k>.
set -u
SUITE=$1; K=$2; PORT=$3; shift 3; TASKS="$*"
export PATH=/usr/local/bin:/usr/bin:/bin
export HOME=/home/weiland
R=/data/openpi_dispatch
B=/tmp/dsp_shared/rit_pareto/$SUITE
SLOG=/tmp/rit_srv_${SUITE}_${K}.log
CLOG=/tmp/rit_cli_${SUITE}_${K}.log
mkdir -p $B/h5/shard$K
rm -f $SLOG $CLOG
tmux new -s rits_${SUITE}_$K -d "cd $R && export HOME=/home/weiland PYTHONPATH=$R/src:$R OPENPI_SERVER_GPU_MEMORY_LOCK=0 && /home/weiland/openpi/.venv/bin/python scripts/serve_policy.py --collect --collect_dir $B/h5/shard$K --env LIBERO --non-concurrent --port $PORT policy:checkpoint --policy.config pi05_libero --policy.dir /home/weiland/.cache/openpi/openpi-assets/checkpoints/pi05_libero_pytorch 2>&1 | tee $SLOG"
tmux new -s ritc_${SUITE}_$K -d "cd $R && export HOME=/home/weiland && until grep -q 'server listening on' $SLOG 2>/dev/null; do if grep -qiE 'Traceback|Killed|out of memory|CUDA error|Address already in use' $SLOG 2>/dev/null; then echo RIT_SERVER_FAILED; exit 1; fi; sleep 3; done; MUJOCO_EGL_DEVICE_ID=0 PYTHONPATH=. /home/weiland/miniconda3/bin/conda run --no-capture-output -p /home/weiland/libero_sim python examples/libero/main.py --host 127.0.0.1 --port $PORT --task-suite-name $SUITE --num-trials-per-task 15 --num-workers 1 --init-states-dir $B/shadow_pool --cohort-plan $B/cohort_plan.json --task-ids $TASKS 2>&1 | tee $CLOG; echo RIT_CLIENT_EXIT=\${PIPESTATUS[0]} | tee -a $CLOG; echo RIT_CLIENT_DONE | tee -a $CLOG"
echo "launched shard $K port $PORT tasks [$TASKS]"
