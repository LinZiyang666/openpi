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
# Environment (all optional; defaults reproduce the fusion-ablation lane):
#   FW_CFG_DIR        config root, the suite is appended     (absolute path)
#   FW_MATRIX         arm matrix yaml, default $CFG/matrix_$SUITE.yaml (absolute path)
#   FW_BOOT           yaml the servers boot on              (absolute path)
#   FW_OUT            output root, the suite is appended
#   FW_STAGE2_DEVICE  serve_policy --stage2-device: meta (default) or cuda:0. A key
#                     builder that runs LLM layers inside the builder
#                     (cp1_llm_layer_extract) needs the language model resident.
#   FW_EVAL_CONC      driver --eval-concurrency (default 2)
#   FW_ROOT / FW_PY   repo root and interpreter (defaults: /home/weiland/openpi and
#                     its .venv); only tests point them elsewhere.
# Every launch writes its own server_<port>.<stamp>.log and driver.<stamp>.log so a
# relaunch never overwrites the previous attempt's evidence.
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
R=${FW_ROOT:-/home/weiland/openpi}
CFG=${FW_CFG_DIR:-$R/exp/weighted_sum/config/fusion_ablation/pi05}/$SUITE
OUT=${FW_OUT:-/data/openpi/weighted_sum/fusion_ablation}/$SUITE
MATRIX=${FW_MATRIX:-$CFG/matrix_$SUITE.yaml}
BOOT=${FW_BOOT:-$CFG/fw_pi05_${SUITE}_uniform.yaml}
STAGE2=${FW_STAGE2_DEVICE:-meta}
CKPT=/data/openpi/checkpoints/pi05_libero_pytorch
PY=${FW_PY:-$R/.venv/bin/python}
STAMP=$(date +%Y%m%d_%H%M%S)
# Paths are resolved before the cd below so a relative FW_* never silently
# points somewhere else once the script changes directory.
for V in CFG MATRIX BOOT; do
  case "${!V}" in /*) ;; *) echo "$V must be an absolute path: ${!V}"; exit 2 ;; esac
done
[ -f "$MATRIX" ] || { echo "missing $MATRIX"; exit 2; }
[ -f "$BOOT" ] || { echo "missing $BOOT"; exit 2; }
case "$STAGE2" in meta|cuda:0) ;; *) echo "FW_STAGE2_DEVICE must be meta or cuda:0, got $STAGE2"; exit 2 ;; esac
cd "$R" || exit 1
mkdir -p "$OUT"

echo "== [$SUITE] launch $STAMP: matrix $MATRIX boot $BOOT stage2 $STAGE2 out $OUT"
echo "== [$SUITE] restarting servers on $PORTS (boot $BOOT)"
for PT in $(echo "$PORTS" | tr ',' ' '); do
  tmux kill-session -t "fwsrv$PT" 2>/dev/null
  pkill -TERM -f "[s]erve_policy.py --port $PT " 2>/dev/null
done
sleep 5
for PT in $(echo "$PORTS" | tr ',' ' '); do
  tmux new -s "fwsrv$PT" -d "cd $R && OMP_NUM_THREADS=4 MKL_NUM_THREADS=4 $PY scripts/serve_policy.py \
    --port $PT --replicas 1 --cache-config $BOOT \
    --stage1-device cuda:0 --stage2-device $STAGE2 --stage3-device meta \
    policy:checkpoint --policy.config pi05_libero --policy.dir $CKPT 2>&1 | tee $OUT/server_$PT.$STAMP.log"
  sleep 3
done
for i in $(seq 1 60); do
  up=0
  for PT in $(echo "$PORTS" | tr ',' ' '); do
    grep -q "Loaded .* entries from" "$OUT/server_$PT.$STAMP.log" 2>/dev/null && ss -tln | grep -q ":$PT " && up=$((up+1))
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
  --eval-concurrency ${FW_EVAL_CONC:-2} --episode-timeout-s 1800 --min-full-hit 1 "$@" 2>&1 | tee "$OUT/driver.$STAMP.log"
rc=${PIPESTATUS[0]}
echo "== [$SUITE] DRIVER_EXIT=$rc $(date -Is)" | tee -a "$OUT/driver.$STAMP.log"
exit "$rc"
