#!/usr/bin/env bash
# Drive the verification episodes from timan107: one LIBERO client process per task, official-index filters.
#
# usage: launch_verify_clients.sh <host> <port> <run_tag smoke|verify> <pool_dir> <filter.json> <trials>
#   pool_dir / filter.json / trials for smoke: smoke_pool / smoke_filter.json / 1
#                            for verify: verify_pool / verify_filter.json / 5
set -eu
HOST=${1:?server host}
PORT=${2:?port}
TAG=${3:?run tag}
POOL=${4:?pool dir}
FILTER=${5:?episode filter json}
TRIALS=${6:?trials per task}
REPO=/scratch/zixuans8/openpi_lg
export HOME=/home/zixuans8
export LIBERO_CONFIG_PATH=$HOME/.libero
PY="/home/zixuans8/miniconda3/bin/conda run --no-capture-output -p /scratch/zixuans8/libero_sim python"
[ -d "$POOL" ] || { echo "missing pool $POOL"; exit 1; }
[ -f "$FILTER" ] || { echo "missing filter $FILTER"; exit 1; }
NGPU=$(nvidia-smi --query-gpu=index --format=csv,noheader | wc -l)
for t in $(seq 0 9); do tmux kill-session -t "lotocli_${TAG}_$t" 2>/dev/null || true; done
sleep 2
for t in $(seq 0 9); do
  G=$((t % NGPU))
  tmux new -s "lotocli_${TAG}_$t" -d "cd $REPO && MUJOCO_EGL_DEVICE_ID=$G PYTHONPATH=$REPO:$REPO/src \
    $PY examples/libero/main.py --host $HOST --port $PORT --task-suite-name libero_10 \
    --num-trials-per-task $TRIALS --num-workers 1 --init-states-dir $POOL --episode-filter $FILTER \
    --task-ids $t --replan-steps 5 --resize-size 256 2>&1 | tee /tmp/lotocli_${TAG}_$t.log; \
    echo LOTOCLI_EXIT=\${PIPESTATUS[0]} | tee -a /tmp/lotocli_${TAG}_$t.log"
  sleep 2
done
sleep 5
tmux ls 2>/dev/null | grep -c "^lotocli_${TAG}_" | sed 's/^/client sessions: /'
