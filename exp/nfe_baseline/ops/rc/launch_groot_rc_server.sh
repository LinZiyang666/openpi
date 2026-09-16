#!/bin/bash
# RoboCasa365 GR00T teacher-only server at k denoising steps, one process (h100 or weilandserver).
#
# usage: launch_groot_rc_server.sh <k> <port>
# env:   NFE_HOME NFE_REPO NFE_GROOT NFE_GROOT_PY NFE_RC_GROOT_CKPT (defaults = weilandserver)
#
# Same idempotent claim (mkdir lock + tmux has-session) as the LIBERO launchers;
# readiness is SERVER-LISTENING in the log. The served step count is attested
# twice in the log: the G-A1 wrapper's KSWEEP num_inference_timesteps=<k>
# assertion and this line's KSWEEP num_steps=<k>.
set -u
K=${1:?k}; P=${2:?port}
REPO=${NFE_REPO:-/data/openpi_nfe}
GROOT=${NFE_GROOT:-/home/weiland/gr00t_n15}
PY=${NFE_GROOT_PY:-/home/weiland/gr00t_n15_venv/.venv/bin/python}
CKPT=${NFE_RC_GROOT_CKPT:-/home/weiland/ckpt_n15_robocasa_tp/gr00t_n1-5/foundation_model_learning/target_posttraining/atomic_seen/checkpoint-60000}
export PATH=/usr/local/bin:/usr/bin:/bin
export HOME=${NFE_HOME:-/home/weiland}
mkdir -p /tmp/nfe
[ -d "$CKPT" ] || { echo "checkpoint dir $CKPT missing"; exit 1; }
if ! mkdir "/tmp/nfe/lock_nfesrv$P" 2>/dev/null; then echo "nfesrv$P claimed (lock present); leaving it"; exit 0; fi
if tmux has-session -t "nfesrv$P" 2>/dev/null; then echo "nfesrv$P already exists; leaving it"; exit 0; fi
if ss -tlnH "sport = :$P" | grep -q .; then echo "port $P busy (not ours); refusing"; rmdir "/tmp/nfe/lock_nfesrv$P"; exit 1; fi
rm -f "/tmp/nfe/nfesrv$P.log"
tmux new -s "nfesrv$P" -d "cd $REPO && export HOME=$HOME && \
  HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1 NO_ALBUMENTATIONS_UPDATE=1 \
  PYTHONPATH=$GROOT:$REPO/src:$REPO \
  $PY -m exp.nfe_baseline.serve_groot_rc_ksweep --denoising-steps $K --port $P --concurrent \
    --checkpoint $CKPT 2>&1 | tee /tmp/nfe/nfesrv$P.log"
for _ in $(seq 1 60); do
  grep -q "SERVER-LISTENING" "/tmp/nfe/nfesrv$P.log" 2>/dev/null && break
  sleep 10
done
printf '%s: %s | %s | %s\n' "$P" "$(grep -m1 -o 'KSWEEP num_inference_timesteps=[0-9]*' /tmp/nfe/nfesrv$P.log 2>/dev/null || echo 'NO KSWEEP ASSERT')" \
  "$(grep -m1 'serving stack' /tmp/nfe/nfesrv$P.log 2>/dev/null || echo 'NOT UP')" \
  "$(ss -tlnH "sport = :$P" | grep -q . && echo listening || echo NOT LISTENING)"
