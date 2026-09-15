#!/usr/bin/env bash
# GR00T worker fleet on a simulator box: one agent per server endpoint, every
# worker launched with the GR00T rollout knobs (--resize-size 256
# --replan-steps 5). Otherwise identical to cache_prune/ops/launch_fleet.sh.
#
# usage: fw_groot_fleet.sh <server-host> <driver-port> <ports csv> <workers-per-endpoint> <gpu-ids csv> <prefix>
set -uo pipefail
HOST=${1:?host}; DRIVER_PORT=${2:?driver port}; PORTS=${3:?ports}; WORKERS=${4:?workers}; GPUS=${5:?gpu ids}; PREFIX=${6:?prefix}
R=/scratch/zixuans8/openpi_lg
export HOME=/home/zixuans8
export LIBERO_CONFIG_PATH=$HOME/.libero
export PYTHONPATH=$R/packages/openpi-client/src:$R/src:$R
export PATH=/scratch/zixuans8/dsp_bin:/usr/local/bin:/usr/bin:/bin
PY=/scratch/zixuans8/openpi/.venv/bin/python
POOLS=$R/exp/common/data/db_init/libero
CFG=$R/exp/ablation_study/cache_size/config
SERVERS=$(echo "$PORTS" | tr ',' '\n' | sed "s/^/$HOST:/" | paste -sd, -)
cd "$R" || exit 1
for P in $(echo "$PORTS" | tr ',' ' '); do
  S="fwgag$P"
  if tmux has-session -t "$S" 2>/dev/null; then echo "== $S already running"; continue; fi
  tmux new -s "$S" -d "cd $R && $PY -m exp.ablation_study.cache_size.run_size_eval \
    --role agent --task-suite libero_spatial \
    --servers $SERVERS --agent-server $HOST:$P \
    --driver-host $HOST --driver-port $DRIVER_PORT \
    --workers $WORKERS --gpu-ids $GPUS \
    --conda-env /scratch/zixuans8/libero_sim \
    --resize-size 256 --replan-steps 5 \
    --apool-record $CFG/apool_libero_spatial.yaml \
    --apool-record $CFG/apool_libero_10.yaml \
    --apool-dir libero_spatial=$POOLS/libero_spatial_apool,libero_10=$POOLS/libero_10_apool \
    --worker-prefix $PREFIX-p$P- 2>&1 | tee /tmp/$S.log; echo AGENT_EXIT=\${PIPESTATUS[0]} | tee -a /tmp/$S.log"
  echo "== started $S: $WORKERS workers -> $HOST:$P"
  sleep 2
done
sleep 5
tmux ls 2>/dev/null | grep -c '^fwgag' | sed 's/^/agent sessions: /'
