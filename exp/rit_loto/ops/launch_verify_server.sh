#!/usr/bin/env bash
# One concurrent GR00T server on weilandserver serving the verification arm with the LOTO decision log.
#
# usage: launch_verify_server.sh <arm.yaml> <run_tag smoke|verify> <frozen_run.json> [port] [tmux-name]
set -eu
export HOME=/home/weiland
ARM=${1:?arm yaml}
TAG=${2:?run tag}
FROZEN=${3:?frozen run record}
PORT=${4:-23150}
NAME=${5:-srv0}
REPO=/data/openpi_lg
G=/home/weiland/gr00t_n15
PY=/home/weiland/gr00t_n15_venv/.venv/bin/python
CKPT=/home/weiland/ckpt_n15_libero_10
LOG_ROOT=/data/libero_cache/rit_loto/libero_10/verify_logs
mkdir -p "$LOG_ROOT"
if ss -tln | grep -q ":$PORT "; then echo "port $PORT is busy on this host; pick another"; exit 1; fi
if tmux has-session -t "$NAME" 2>/dev/null; then echo "tmux $NAME exists; pick another name"; exit 1; fi
tmux new -s "$NAME" -d "cd $REPO && PYTHONPATH=$G:$G/examples/Libero:$REPO:$REPO/src HF_HUB_OFFLINE=1 OPENPI_MONITOR_LEVEL=BASIC \
  $PY exp/libero_groot/serve_groot_libero.py --checkpoint $CKPT --port $PORT --denoising-steps 8 --concurrent \
  --cache-config $ARM --loto-log-out $LOG_ROOT --loto-run-tag $TAG --loto-frozen-record $FROZEN \
  2>&1 | tee /tmp/$NAME.log"
for _ in $(seq 1 90); do grep -q "SERVER-LISTENING" /tmp/$NAME.log 2>/dev/null && break; sleep 10; done
grep -m1 'serving stack' /tmp/$NAME.log || echo "server not up yet; tail /tmp/$NAME.log"
