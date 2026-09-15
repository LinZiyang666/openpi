#!/usr/bin/env bash
# GR00T arm(s) under the conductor: N concurrent GR00T replicas on this node
# (separate processes, --allow-dynamic-bundles, no startup yaml) and the
# run_size_eval driver spreading one arm's 500 A-pool episodes over all of
# them; the remote fleet (fw_groot_fleet.sh on the simulator box) does the
# episodes with --resize-size 256 --replan-steps 5.
#
# usage: fw_groot_conductor.sh <suite> <host the fleet dials> <driver-port> <ports csv> <workers-per-endpoint> <arms csv>
set -uo pipefail
SUITE=${1:?suite}; HOST=${2:?host}; DRIVER_PORT=${3:?driver port}; PORTS=${4:?ports}; WORKERS=${5:?workers}; ARMS=${6:?arms}
R=${FW_REPO:-/data/openpi_lg}
PY=${FW_PY:-/home/weiland/openpi/.venv/bin/python}
ISLAND_PY=${GROOT_N15_PYTHON:-/home/weiland/gr00t_n15_venv/.venv/bin/python}
GROOT=${GROOT_N15_HOME:-/home/weiland/gr00t_n15}
case "$SUITE" in
  libero_spatial) CKPT=${FW_CKPT:-/home/weiland/ckpt_n15_libero_spatial} ;;
  libero_10)      CKPT=${FW_CKPT:-/home/weiland/ckpt_n15_libero_10} ;;
  *) echo "unknown suite $SUITE"; exit 2 ;;
esac
OUT=${FW_OUT:-/data/libero_cache/search/fusion_ablation_conductor}/$SUITE
MATRIX=${FW_MATRIX_DIR:-$R/exp/weighted_sum/config/fusion_ablation/groot}/matrix_${SUITE}.yaml
[ -f "$MATRIX" ] || { echo "missing $MATRIX"; exit 2; }
mkdir -p "$OUT"; cd "$R" || exit 1
export HOME=${HOME:-/home/weiland}

echo "== [$SUITE] restarting GR00T replicas on $PORTS ($CKPT)"
for PT in $(echo "$PORTS" | tr ',' ' '); do tmux kill-session -t "fwgsrv$PT" 2>/dev/null; done
sleep 5
for PT in $(echo "$PORTS" | tr ',' ' '); do
  tmux new -s "fwgsrv$PT" -d "cd $R && PYTHONPATH=$GROOT:$GROOT/examples/Libero:$R:$R/src HF_HUB_OFFLINE=1 OPENPI_MONITOR_LEVEL=BASIC \
    $ISLAND_PY exp/libero_groot/serve_groot_libero.py --checkpoint $CKPT --port $PT --denoising-steps 8 \
    --concurrent --allow-dynamic-bundles ${FW_SERVER_EXTRA:-} 2>&1 | tee $OUT/server_$PT.log"
  sleep 12
done
N=$(echo "$PORTS" | tr ',' '\n' | wc -l)
for i in $(seq 1 60); do
  up=0
  for PT in $(echo "$PORTS" | tr ',' ' '); do grep -q "SERVER-LISTENING" "$OUT/server_$PT.log" 2>/dev/null && up=$((up+1)); done
  [ "$up" -eq "$N" ] && break
  sleep 10
done
echo "== [$SUITE] replicas listening: $up/$N"
[ "$up" -eq "$N" ] || { echo "not all replicas came up"; exit 3; }

SERVERS=$(echo "$PORTS" | tr ',' '\n' | sed "s/^/$HOST:/" | paste -sd, -)
CAPS=$(yes "$WORKERS" | head -n "$N" | paste -sd, -)
echo "== [$SUITE] driving arms $ARMS on $SERVERS ($CAPS workers) $(date -Is)"
PYTHONPATH=$R:$R/src $PY -m exp.ablation_study.cache_size.run_size_eval --role driver \
  --arm-matrix "$MATRIX" $( [ "$ARMS" = all ] || echo --arms "$ARMS" ) --task-suite "$SUITE" \
  --servers "$SERVERS" --server-workers "$CAPS" --workers $((WORKERS * N)) \
  --trials 50 \
  --journal "$OUT/journal.jsonl" --per-step-out "$OUT/per_step.jsonl" \
  --apool-record "$R/exp/ablation_study/cache_size/config/apool_${SUITE}.yaml" \
  --bind-host 0.0.0.0 --bind-port "$DRIVER_PORT" \
  --eval-concurrency ${FW_EVAL_CONC:-2} --episode-timeout-s 1800 --min-full-hit 1 2>&1 | tee -a "$OUT/driver.log"
rc=${PIPESTATUS[0]}
echo "== [$SUITE] DRIVER_EXIT=$rc $(date -Is)"
for PT in $(echo "$PORTS" | tr ',' ' '); do tmux kill-session -t "fwgsrv$PT" 2>/dev/null; done
exit "$rc"
