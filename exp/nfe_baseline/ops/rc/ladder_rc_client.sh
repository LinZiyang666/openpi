#!/usr/bin/env bash
# Client-side RoboCasa365 lane driver (timan107 / timan108): every k of one teacher, main lane then pnp lane.
#
# usage: ladder_rc_client.sh <teacher pi05|groot_tp> <ks csv> <server host:port> <workers> <gpu-ids csv> <out_root>
#
# For each k: wait until the server's handshake reports k (probe_metadata.py,
# key nfe_num_steps stamped by both RoboCasa wrappers), run the main lane
# (8 tasks) then the pnp lane (5 pinned PickPlace tasks) through
# run_rc_client.sh, and require each lane's summary to be complete before
# moving on. A lane whose complete summary already exists is skipped (resume).
# Markers: RCLANE k=<k> UP / DONE / FAIL, RCLANE FINISHED.
set -u
TEACHER=${1:?teacher}; KS=${2:?ks csv}; SERVER=${3:?host:port}; W=${4:?workers}; GPUS=${5:?gpu ids}; OUT=${6:?out root}
HERE=$(cd "$(dirname "$0")" && pwd)
REPO=${NFE_RC_REPO:-/scratch/zixuans8/nfe/openpi_nfe}
PROBE_PY=${NFE_PROBE_PY:-/scratch/zixuans8/libero_sim/bin/python}
WAIT_SERVER_S=${NFE_WAIT_SERVER_S:-7200}
TAG=${NFE_TAG:-run}
PREFIX=${NFE_RC_PREFIX_STEM:-nfek}
export HOME=${NFE_HOME:-/home/zixuans8}
HOST=${SERVER%%:*}; PORT=${SERVER##*:}
IFS=',' read -r -a K_ARR <<< "$KS"

lane_complete() {  # <lane> <k>
  local f="$OUT/$TEACHER/$1/k$2/summary_${PREFIX}$2-teacher__l1s1_$TEACHER.json"
  [ -f "$f" ] && grep -q '"complete": true' "$f"
}

for K in "${K_ARR[@]}"; do
  if lane_complete main "$K" && lane_complete pnp "$K"; then echo "RCLANE k=$K already done; skip"; continue; fi
  waited=0
  while :; do
    got=$(cd "$REPO" && PYTHONPATH=$REPO/packages/openpi-client/src $PROBE_PY "$HERE/../../probe_metadata.py" "$HOST" "$PORT" 2>/dev/null | tail -1)
    [ "$got" = "$K" ] && break
    waited=$((waited+30))
    if [ "$waited" -ge "$WAIT_SERVER_S" ]; then echo "RCLANE FAIL k=$K server never reported k (last=$got)"; exit 1; fi
    sleep 30
  done
  echo "RCLANE k=$K UP $(date +%H:%M:%S)"
  for LANE in main pnp; do
    if lane_complete "$LANE" "$K"; then echo "RCLANE k=$K $LANE already complete; skip"; continue; fi
    NFE_RC_PREFIX="${PREFIX}${K}" NFE_TAG=$TAG bash "$HERE/run_rc_client.sh" "$TEACHER" "$K" "$SERVER" "$LANE" "$W" "$GPUS" "$OUT" "$TAG" \
      || { echo "RCLANE FAIL k=$K $LANE launch"; exit 1; }
    LOG="/tmp/nfe/nfercli_${TAG}_${TEACHER}_k${K}_${LANE}.log"
    while ! grep -q "RCCLI_EXIT=" "$LOG" 2>/dev/null; do sleep 60; done
    if ! grep -q "RCCLI_EXIT=0" "$LOG" || ! lane_complete "$LANE" "$K"; then
      echo "RCLANE FAIL k=$K $LANE exit=$(grep -o 'RCCLI_EXIT=[0-9]*' "$LOG" | tail -1) complete=$(lane_complete "$LANE" "$K" && echo yes || echo no) $(date +%H:%M:%S)"
      exit 1
    fi
    echo "RCLANE k=$K $LANE DONE $(date +%H:%M:%S)"
  done
  echo "RCLANE k=$K DONE $(date +%H:%M:%S)"
done
echo "RCLANE FINISHED $TEACHER $(date +%H:%M:%S)"
