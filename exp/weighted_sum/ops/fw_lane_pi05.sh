#!/usr/bin/env bash
# Fusion-weight ablation, Pi0.5 lane: five pure-cache servers on this node plus
# the run_size_eval driver, one suite per invocation; the remote fleet
# (cache_prune/ops/launch_fleet.sh on the simulator box) does the episodes.
#
# Servers boot on the suite's uniform arm and receive the other two arms as
# bundle YAMLs at stage start. Stage 2/3 sit on meta: every step of an
# always_hit arm is a FULL_HIT, which the driver's --min-full-hit 1 witnesses.
# Resume is by journal.
#
# usage: HOME=<node home> fw_lane_pi05.sh <host the fleet dials> <driver-port> <ports csv> <workers-per-endpoint> <suite> [extra driver args]
set -uo pipefail

HOST=${1:?host}
DRIVER_PORT=${2:?driver port}
PORTS=${3:?server ports csv}
WORKERS=${4:?workers per endpoint}
SUITE=${5:?suite}
shift 5

: "${HOME:?set HOME to the node user home}"
export OMP_NUM_THREADS=4 MKL_NUM_THREADS=4 OPENBLAS_NUM_THREADS=4
unset CUDA_VISIBLE_DEVICES
R=/home/weiland/openpi
CFG=${FW_CFG_DIR:-$R/exp/weighted_sum/config/fusion_ablation/pi05}/$SUITE
OUT=${FW_OUT:-/data/openpi/weighted_sum/fusion_ablation}/$SUITE
MATRIX=$CFG/matrix_$SUITE.yaml
BOOT=${FW_BOOT:-$CFG/fw_pi05_${SUITE}_uniform.yaml}
CKPT=/data/openpi/checkpoints/pi05_libero_pytorch
PY=$R/.venv/bin/python
cd "$R" || exit 1
[ -f "$MATRIX" ] || { echo "missing $MATRIX"; exit 2; }
mkdir -p "$OUT"

echo "== [$SUITE] restarting servers on $PORTS (boot $BOOT)"
for PT in $(echo "$PORTS" | tr ',' ' '); do
  tmux kill-session -t "fwsrv$PT" 2>/dev/null
  pkill -TERM -f "[s]erve_policy.py --port $PT " 2>/dev/null
done
sleep 5
for PT in $(echo "$PORTS" | tr ',' ' '); do
  tmux new -s "fwsrv$PT" -d "cd $R && OMP_NUM_THREADS=4 MKL_NUM_THREADS=4 $PY scripts/serve_policy.py \
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
echo "== [$SUITE] servers listening: $up"
[ "$up" -gt 0 ] || { echo "no server came up"; exit 3; }

SERVERS=$(echo "$PORTS" | tr ',' '\n' | sed "s/^/$HOST:/" | paste -sd, -)
N=$(echo "$PORTS" | tr ',' '\n' | wc -l)
CAPS=$(yes "$WORKERS" | head -n "$N" | paste -sd, -)
echo "== [$SUITE] driving $MATRIX on $SERVERS ($CAPS workers) $(date -Is)"
$PY -m exp.ablation_study.cache_size.run_size_eval --role driver \
  --arm-matrix "$MATRIX" --task-suite "$SUITE" \
  --servers "$SERVERS" --server-workers "$CAPS" --workers $((WORKERS * N)) \
  --trials 50 \
  --journal "$OUT/journal.jsonl" --per-step-out "$OUT/per_step.jsonl" \
  --apool-record "$R/exp/ablation_study/cache_size/config/apool_${SUITE}.yaml" \
  --bind-host 0.0.0.0 --bind-port "$DRIVER_PORT" \
  --eval-concurrency ${FW_EVAL_CONC:-2} --episode-timeout-s 1800 --min-full-hit 1 "$@" 2>&1 | tee "$OUT/driver.log"
rc=${PIPESTATUS[0]}
echo "== [$SUITE] DRIVER_EXIT=$rc $(date -Is)"
exit "$rc"
