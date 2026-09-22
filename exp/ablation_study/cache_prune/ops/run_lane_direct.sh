#!/usr/bin/env bash
# Evaluate one arm family (suite x regime) on a lane, without the freeze gate.
#
# Brings up this node's concurrent servers, then drives the family's arms from
# here while the lane's remote fleet (launch_fleet.sh on the simulator box)
# does the episodes. The driver hands each server its arms as bundle YAMLs at
# stage start, so the servers boot on the family's smallest library and grow
# only by what they are assigned. Resume is by journal: re-running the same
# family walks only the episodes it still lacks.
#
# usage: HOME=<node home> run_lane_direct.sh <lane> <host> <driver-port> <ports csv> <workers-per-endpoint> <suite> <regime>
set -uo pipefail

LANE=${1:?lane}
HOST=${2:?host the fleet dials}
DRIVER_PORT=${3:?driver port}
PORTS=${4:?server ports csv}
WORKERS=${5:?workers per endpoint}
SUITE=${6:?suite}
REGIME=${7:?regime}

: "${HOME:?set HOME to the node user home}"
export OMP_NUM_THREADS=4 MKL_NUM_THREADS=4 OPENBLAS_NUM_THREADS=4
unset CUDA_VISIBLE_DEVICES
R=/home/weiland/projects/openpi
P=/data/openpi/ablation_study/cache_prune/concurrent_v1
CFG=$P/config_direct
OUT=$P/direct/${LANE}/${SUITE}_${REGIME}
MATRIX=$CFG/matrix_${SUITE}_${REGIME}.yaml
CKPT=/data/openpi/checkpoints/pi05_libero_pytorch
PY=$R/.venv/bin/python
cd "$R" || exit 1
[ -f "$MATRIX" ] || { echo "missing $MATRIX"; exit 2; }
mkdir -p "$OUT"

# Boot library: the family's smallest (P09); the driver loads the rest on demand.
BOOT=$(grep -oE "yaml: .*_P09\.yaml" "$MATRIX" | head -1 | cut -d' ' -f2)
[ -n "$BOOT" ] || BOOT=$(grep -oE "yaml: .*\.yaml" "$MATRIX" | head -1 | cut -d' ' -f2)

echo "== [$LANE] $SUITE/$REGIME: restarting servers on $PORTS (boot $BOOT)"
for PT in $(echo "$PORTS" | tr ',' ' '); do
  tmux kill-session -t "cpsrv$PT" 2>/dev/null
  # Only this experiment's ports: the box is shared with other servers.
  pkill -TERM -f "[s]erve_policy.py --port $PT " 2>/dev/null
done
sleep 5
for PT in $(echo "$PORTS" | tr ',' ' '); do
  tmux new -s "cpsrv$PT" -d "cd $R && OMP_NUM_THREADS=4 MKL_NUM_THREADS=4 $PY scripts/serve_policy.py \
    --port $PT --replicas 1 --cache-config $BOOT \
    --stage1-device cuda:0 --stage2-device meta --stage3-device meta \
    policy:checkpoint --policy.config pi05_libero --policy.dir $CKPT 2>&1 | tee $OUT/server_$PT.log"
  sleep 3
done
for i in $(seq 1 60); do
  up=0
  for PT in $(echo "$PORTS" | tr ',' ' '); do
    grep -q "Loaded .* entries from" "$OUT/server_$PT.log" 2>/dev/null && ss -tln | grep -q ":$PT " && up=$((up+1))
  done
  [ "$up" -eq "$(echo "$PORTS" | tr ',' '\n' | wc -l)" ] && break
  sleep 5
done
echo "== [$LANE] servers listening: $up"
[ "$up" -gt 0 ] || { echo "no server came up"; exit 3; }

SERVERS=$(echo "$PORTS" | tr ',' '\n' | sed "s/^/$HOST:/" | paste -sd, -)
N=$(echo "$PORTS" | tr ',' '\n' | wc -l)
CAPS=$(yes "$WORKERS" | head -n "$N" | paste -sd, -)
echo "== [$LANE] driving $MATRIX on $SERVERS ($CAPS workers) $(date -Is)"
$PY -m exp.ablation_study.cache_size.run_size_eval --role driver \
  --arm-matrix "$MATRIX" --task-suite "$SUITE" \
  --servers "$SERVERS" --server-workers "$CAPS" --workers $((WORKERS * N)) \
  --trials 50 \
  --journal "$OUT/journal.jsonl" --per-step-out "$OUT/per_step.jsonl" \
  --apool-record "$R/exp/ablation_study/cache_size/config/apool_${SUITE}.yaml" \
  --bind-host 0.0.0.0 --bind-port "$DRIVER_PORT" \
  --eval-concurrency 2 --episode-timeout-s 1800 --min-full-hit 1 2>&1 | tee "$OUT/driver.log"
rc=${PIPESTATUS[0]}
echo "== [$LANE] DRIVER_EXIT=$rc $SUITE/$REGIME $(date -Is)"
exit "$rc"
