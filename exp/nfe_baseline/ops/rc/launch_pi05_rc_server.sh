#!/bin/bash
# RoboCasa365 Pi0.5 teacher-only server at k denoising steps, one process, batched (--cache) mode.
#
# usage: launch_pi05_rc_server.sh <k> <port>
# env:   NFE_HOME NFE_REPO NFE_PI05_PY NFE_RC_PI05_CKPT NFE_BATCH (defaults = weilandserver)
#
# ``pi05_robocasa`` config; norm stats come from the checkpoint's own assets/.
# Idempotent claim and readiness (port listening + KSWEEP marker) as the LIBERO
# launcher.
set -u
K=${1:?k}; P=${2:?port}
REPO=${NFE_REPO:-/data/openpi_nfe}
PY=${NFE_PI05_PY:-/home/weiland/openpi/.venv/bin/python}
CKPT=${NFE_RC_PI05_CKPT:-/home/weiland/ckpt_pi05_robocasa_pytorch}
export PATH=/usr/local/bin:/usr/bin:/bin
export HOME=${NFE_HOME:-/home/weiland}
mkdir -p /tmp/nfe
[ -e "$CKPT" ] || { echo "checkpoint $CKPT missing"; exit 1; }
if ! mkdir "/tmp/nfe/lock_nfesrv$P" 2>/dev/null; then echo "nfesrv$P claimed (lock present); leaving it"; exit 0; fi
if tmux has-session -t "nfesrv$P" 2>/dev/null; then echo "nfesrv$P already exists; leaving it"; exit 0; fi
if ss -tlnH "sport = :$P" | grep -q .; then echo "port $P busy (not ours); refusing"; rmdir "/tmp/nfe/lock_nfesrv$P"; exit 1; fi
rm -f "/tmp/nfe/nfesrv$P.log"
tmux new -s "nfesrv$P" -d "cd $REPO && export HOME=$HOME && \
  export PYTHONPATH=$REPO/src:$REPO && export OPENPI_SERVER_GPU_MEMORY_LOCK=0 && \
  export BATCHING_MAX_BATCH_SIZE=${NFE_BATCH:-16} BATCHING_MAX_WAIT_MS=${NFE_BATCH_WAIT_MS:-10} && \
  $PY -m exp.nfe_baseline.serve_pi05_ksweep --denoising-steps $K --cache --port $P \
    policy:checkpoint --policy.config pi05_robocasa --policy.dir $CKPT 2>&1 | tee /tmp/nfe/nfesrv$P.log"
for _ in $(seq 1 90); do
  ss -tlnH "sport = :$P" | grep -q . && break
  sleep 10
done
printf '%s: %s | %s\n' "$P" "$(grep -m1 'KSWEEP num_steps' /tmp/nfe/nfesrv$P.log 2>/dev/null || echo 'NO KSWEEP MARKER')" \
  "$(ss -tlnH "sport = :$P" | grep -q . && echo listening || echo NOT LISTENING)"
