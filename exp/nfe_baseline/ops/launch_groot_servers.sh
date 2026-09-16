#!/bin/bash
# GR00T N1.5 LIBERO teacher-only servers at k denoising steps, one process per port (weilandserver).
#
# usage: launch_groot_servers.sh <suite libero_spatial|libero_10> <k> <base_port> <n> [repo] [gr00t root] [python] [ckpt dir]
#
# Host defaults are weilandserver's; h100 passes its own repo / gr00t / python /
# ckpt dir and NFE_HOME=/home/exouser. The production entry point already takes
# --denoising-steps; without --cache-config it serves the policy alone through
# the locked teacher factory and runs no schedule guard, so k=1 is fine.
# Sessions are nfesrv<port>, logs /tmp/nfe/nfesrv<port>.log; readiness is
# SERVER-LISTENING in the log. Idempotent: a port whose session already exists
# or that is already listening is left alone (tether exec on h100 runs a
# command twice; the per-port claim is an atomic mkdir lock); stop_servers.sh
# releases the locks and is the way to restart.
set -u
SUITE=${1:?suite}
K=${2:?k}
BASE=${3:?base port}
N=${4:?n servers}
REPO=${5:-/data/openpi_nfe}
GROOT=${6:-/home/weiland/gr00t_n15}
PY=${7:-/home/weiland/gr00t_n15_venv/.venv/bin/python}
CKPT_DIR=${8:-/home/weiland}
case "$SUITE" in
  libero_spatial) CKPT=$CKPT_DIR/ckpt_n15_libero_spatial; [ -d "$CKPT" ] || CKPT=$CKPT_DIR/n15_libero_spatial ;;
  libero_10) CKPT=$CKPT_DIR/ckpt_n15_libero_10; [ -d "$CKPT" ] || CKPT=$CKPT_DIR/n15_libero_10 ;;
  *) echo "unknown suite $SUITE"; exit 1 ;;
esac
[ -d "$CKPT" ] || { echo "checkpoint dir $CKPT missing"; exit 1; }
export PATH=/usr/local/bin:/usr/bin:/bin
export HOME=${NFE_HOME:-/home/weiland}
mkdir -p /tmp/nfe
PYPATH=$GROOT:$GROOT/examples/Libero:$REPO:$REPO/src

for i in $(seq 0 $((N-1))); do
  P=$((BASE+i))
  if ! mkdir "/tmp/nfe/lock_nfesrv$P" 2>/dev/null; then echo "nfesrv$P claimed (lock present); leaving it"; continue; fi
  if tmux has-session -t "nfesrv$P" 2>/dev/null; then echo "nfesrv$P already exists; leaving it"; continue; fi
  if ss -tlnH "sport = :$P" | grep -q .; then
    echo "port $P busy (not ours); refusing"; ss -tlnp "sport = :$P"; rmdir "/tmp/nfe/lock_nfesrv$P"; exit 1
  fi
  rm -f "/tmp/nfe/nfesrv$P.log"
  tmux new -s "nfesrv$P" -d "cd $REPO && export HOME=$HOME && \
    PYTHONPATH=$PYPATH HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1 OPENPI_MONITOR_LEVEL=BASIC \
    $PY exp/libero_groot/serve_groot_libero.py --checkpoint $CKPT --port $P \
      --denoising-steps $K --concurrent 2>&1 | tee /tmp/nfe/nfesrv$P.log"
  sleep 8
done

echo "== waiting for $N listeners on $BASE..$((BASE+N-1)) ($SUITE k=$K) =="
for _ in $(seq 1 60); do
  up=0
  for i in $(seq 0 $((N-1))); do
    grep -q "SERVER-LISTENING" "/tmp/nfe/nfesrv$((BASE+i)).log" 2>/dev/null && up=$((up+1))
  done
  [ "$up" -eq "$N" ] && { echo "all $N listening"; break; }
  sleep 10
done
for i in $(seq 0 $((N-1))); do
  P=$((BASE+i))
  printf '%s: %s | %s\n' "$P" "$(grep -m1 'denoising steps:' /tmp/nfe/nfesrv$P.log 2>/dev/null || echo 'NO STEPS LINE')" \
    "$(grep -m1 'serving stack' /tmp/nfe/nfesrv$P.log 2>/dev/null || echo 'NOT UP')"
done
