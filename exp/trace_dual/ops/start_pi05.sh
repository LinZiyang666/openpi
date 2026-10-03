#!/bin/bash
# start_pi05.sh <port> <boot-yaml> <trace-dir> <log>  -- one single-replica pi0.5 trace server in tmux trsrv<port>
set -u
PORT=$1 YAML=$2 TR=$3 LOG=$4
R=/home/weiland/projects/openpi
CKPT=/home/weiland/.cache/openpi/openpi-assets/checkpoints/pi05_libero_pytorch
mkdir -p "$TR" "$(dirname "$LOG")"
tmux has-session -t "trsrv$PORT" 2>/dev/null && { echo "tmux trsrv$PORT exists"; exit 2; }
ss -ltnH "sport = :$PORT" | grep -q . && { echo "port $PORT busy"; exit 2; }
tmux new -s "trsrv$PORT" -d "cd $R && export HOME=/home/weiland OMP_NUM_THREADS=4 MKL_NUM_THREADS=4 && \
  $R/.venv/bin/python scripts/serve_policy.py --port $PORT --replicas 1 --cache-config $YAML \
  --trace-out $TR --trace-build-cache \
  policy:checkpoint --policy.config pi05_libero --policy.dir $CKPT 2>&1 | tee $LOG; echo SERVER_EXIT=\${PIPESTATUS[0]} | tee -a $LOG"
echo "started trsrv$PORT"
