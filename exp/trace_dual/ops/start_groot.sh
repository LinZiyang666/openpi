#!/bin/bash
# start_groot.sh <port> <ckpt> <boot-yaml> <trace-dir> <log>  -- one GR00T trace server in tmux trsrv<port>
set -u
PORT=$1 CKPT=$2 YAML=$3 TR=$4 LOG=$5
R=/home/weiland/projects/openpi
G=/home/weiland/projects/openpi_ext/third_party/gr00t_n15
mkdir -p "$TR" "$(dirname "$LOG")"
tmux has-session -t "trsrv$PORT" 2>/dev/null && { echo "tmux trsrv$PORT exists"; exit 2; }
ss -ltnH "sport = :$PORT" | grep -q . && { echo "port $PORT busy"; exit 2; }
tmux new -s "trsrv$PORT" -d "cd $R && export HOME=/home/weiland HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1 OPENPI_MONITOR_LEVEL=BASIC \
  PYTHONPATH=$G:$G/examples/Libero:$R:$R/src:$R/packages/openpi-client/src && \
  /home/weiland/projects/openpi_ext/envs/gr00t_n15_venv/.venv/bin/python exp/libero_groot/serve_groot_libero.py \
  --checkpoint $CKPT --port $PORT --denoising-steps 8 --concurrent --allow-dynamic-bundles \
  --cache-config $YAML --trace-out $TR --trace-build-cache 2>&1 | tee $LOG; echo SERVER_EXIT=\${PIPESTATUS[0]} | tee -a $LOG"
echo "started trsrv$PORT"
