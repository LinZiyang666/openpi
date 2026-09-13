#!/usr/bin/env bash
# Bring up one lane's worker fleet on a simulator box.
#
# One agent per server endpoint, because an agent binds all of its workers to a
# single endpoint (the conductor schedules per endpoint). Each agent verifies
# both suites' frozen A-pools on this disk before spawning, and every worker
# sends a probe of its environment on its first pull, so the driver on the
# serving node can attest the fleet it never sees.
#
# The fleet is resident across waves: a finished driver tells the workers to
# shut down, the agent respawns them, and they retry the driver's address with
# backoff until the next wave binds it. Waves of either suite are served by the
# same fleet because the workers follow each task's suite.
#
# usage: launch_fleet.sh <server-host> <driver-port> <ports csv> <workers-per-endpoint> <gpu-ids csv> <prefix>
set -uo pipefail

HOST=${1:?server host the fleet dials}
DRIVER_PORT=${2:?driver pull port}
PORTS=${3:?comma list of server ports}
WORKERS=${4:?workers per endpoint}
GPUS=${5:?comma list of render GPU indices}
PREFIX=${6:?worker id prefix, e.g. timan107}

R=/scratch/zixuans8/openpi_lg
# The NFS home, not the tether agent's: LIBERO resolves its asset cache under
# ~/.cache/libero/assets and the agent's home carries only a partial copy.
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
  S="cpag$P"
  if tmux has-session -t "$S" 2>/dev/null; then
    echo "== $S already running"; continue
  fi
  tmux new -s "$S" -d "cd $R && $PY -m exp.ablation_study.cache_size.run_size_eval \
    --role agent --task-suite libero_spatial \
    --servers $SERVERS --agent-server $HOST:$P \
    --driver-host $HOST --driver-port $DRIVER_PORT \
    --workers $WORKERS --gpu-ids $GPUS \
    --conda-env /scratch/zixuans8/libero_sim \
    --apool-record $CFG/apool_libero_spatial.yaml \
    --apool-record $CFG/apool_libero_10.yaml \
    --apool-dir libero_spatial=$POOLS/libero_spatial_apool,libero_10=$POOLS/libero_10_apool \
    --worker-prefix $PREFIX-p$P- 2>&1 | tee /tmp/$S.log; \
    echo AGENT_EXIT=\${PIPESTATUS[0]} | tee -a /tmp/$S.log"
  echo "== started $S: $WORKERS workers -> $HOST:$P"
  sleep 2
done
sleep 5
tmux ls 2>/dev/null | grep -c '^cpag' | sed 's/^/agent sessions: /'
