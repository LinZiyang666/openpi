#!/usr/bin/env bash
# One GR00T server process for an online arm (single learner per arm, plan §3.7).
# Usage: launch_server_online.sh <port> <suite> <checkpoint> [tmux_name]
set -euo pipefail
PORT=$1; SUITE=$2; CKPT=$3; NAME=${4:-ort_srv_$PORT}
ROOT=${OPENPI_ROOT:-/data/openpi_lg}
PY=${GROOT_PY:-/home/weiland/gr00t_n15_venv/.venv/bin/python}
STATE=$ROOT/exp/online_rit/data/$SUITE/state
mkdir -p "$STATE"
tmux new-session -d -s "$NAME" "cd $ROOT && export PYTHONPATH=/home/weiland/gr00t_n15:/home/weiland/gr00t_n15/examples/Libero:$ROOT:$ROOT/src && \
  $PY exp/libero_groot/serve_groot_libero.py --checkpoint $CKPT --port $PORT --concurrent \
  --allow-dynamic-bundles --online-state-dir $STATE 2>&1 | tee $STATE/server_$PORT.log"
echo "started $NAME on :$PORT (state -> $STATE)"
