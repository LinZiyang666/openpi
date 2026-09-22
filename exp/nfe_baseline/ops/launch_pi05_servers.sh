#!/bin/bash
# Pi0.5 teacher-only servers with num_steps pinned to k, one process per port (weilandserver).
#
# usage: launch_pi05_servers.sh <k> <base_port> <n> [repo] [python] [ckpt]
#
# Served with --cache (interceptor without a library) so the BatchingCoordinator
# batches stage 1/2/3 across connections (NFE_BATCH, default 16); no retrieval,
# every decision is a k-step MISS. One single-replica process per port: the ksweep wrapper's patch does not
# survive the replica supervisor's spawn, so scale-out is N processes here,
# not --replicas. Each process is a tmux session nfesrv<port> logging to
# /tmp/nfe/nfesrv<port>.log; readiness is the port listening plus the
# KSWEEP marker in the log. Ports already listening are refused, never taken;
# an existing nfesrv<port> session is left alone (idempotent re-invocation);
# the per-port claim is an atomic mkdir lock, released by stop_servers.sh.
set -u
K=${1:?k}
BASE=${2:?base port}
N=${3:?n servers}
REPO=${4:-/data/openpi_nfe}
PY=${5:-/home/weiland/projects/openpi/.venv/bin/python}
CKPT=${6:-/home/weiland/.cache/openpi/openpi-assets/checkpoints/pi05_libero_pytorch}
export PATH=/usr/local/bin:/usr/bin:/bin
export HOME=${NFE_HOME:-/home/weiland}
mkdir -p /tmp/nfe

for i in $(seq 0 $((N-1))); do
  P=$((BASE+i))
  if ! mkdir "/tmp/nfe/lock_nfesrv$P" 2>/dev/null; then echo "nfesrv$P claimed (lock present); leaving it"; continue; fi
  if tmux has-session -t "nfesrv$P" 2>/dev/null; then echo "nfesrv$P already exists; leaving it"; continue; fi
  if ss -tlnH "sport = :$P" | grep -q .; then
    echo "port $P busy (not ours); refusing"; ss -tlnp "sport = :$P"; rmdir "/tmp/nfe/lock_nfesrv$P"; exit 1
  fi
  rm -f "/tmp/nfe/nfesrv$P.log"
  tmux new -s "nfesrv$P" -d "cd $REPO && export HOME=$HOME && \
    export PYTHONPATH=$REPO/src:$REPO && export OPENPI_SERVER_GPU_MEMORY_LOCK=0 && \
    export BATCHING_MAX_BATCH_SIZE=${NFE_BATCH:-16} BATCHING_MAX_WAIT_MS=${NFE_BATCH_WAIT_MS:-10} && \
    $PY -m exp.nfe_baseline.serve_pi05_ksweep --denoising-steps $K --cache --port $P \
      policy:checkpoint --policy.config pi05_libero --policy.dir $CKPT 2>&1 | tee /tmp/nfe/nfesrv$P.log"
  sleep 3
done

echo "== waiting for $N listeners on $BASE..$((BASE+N-1)) (k=$K) =="
for _ in $(seq 1 90); do
  up=0
  for i in $(seq 0 $((N-1))); do
    ss -tlnH "sport = :$((BASE+i))" | grep -q . && up=$((up+1))
  done
  [ "$up" -eq "$N" ] && { echo "all $N listening"; break; }
  sleep 10
done
for i in $(seq 0 $((N-1))); do
  P=$((BASE+i))
  printf '%s: %s | %s\n' "$P" "$(grep -m1 'KSWEEP num_steps' /tmp/nfe/nfesrv$P.log 2>/dev/null || echo 'NO KSWEEP MARKER')" \
    "$(ss -tlnH "sport = :$P" | grep -q . && echo listening || echo NOT UP)"
done
