#!/usr/bin/env bash
# One pi0.5 server process of the step-vs-warm-start line (h100 for RC, weilandserver for LIBERO).
#
# usage: serve_pi05.sh <env_id> <mode shadow|plain|full|warm> <arm_id> <port> <exec_steps | cache_yaml | ->
#   env:  SD_REPO SD_PY SD_CKPT SD_EXP SD_OUT SD_LAUNCH SD_HOME (defaults = weilandserver LIBERO)
#
# Single-connection by construction: serve_diag_pi05 passes --non-concurrent, so every process
# takes exactly one worker (run_diag --workers-per-server 1; LIBERO client --num-workers 1).
# Idempotent claim: mkdir lock + tmux has-session; a listening port is refused, never taken.
# Readiness = the STEP_DIAG marker in the log + the port listening. Rows land in
# $SD_OUT/<arm_id>/rows_<arm>_<launch>.jsonl (+ arrays_*/, manifest_<arm>.json).
set -u
ENV_ID=${1:?env_id}; MODE=${2:?mode}; ARM=${3:?arm_id}; P=${4:?port}; ARG=${5:--}
REPO=${SD_REPO:-/data/openpi_sdiag}
PY=${SD_PY:-/home/weiland/projects/openpi/.venv/bin/python}
EXP=${SD_EXP:-sdiag_v1}
OUT=${SD_OUT:-$REPO/exp/step_diag/data/server}
LAUNCH=${SD_LAUNCH:-$(date +%Y%m%dT%H%M%S)_$P}
export PATH=/usr/local/bin:/usr/bin:/bin
export HOME=${SD_HOME:-/home/weiland}
case "$ENV_ID" in
  pi05_rc) CFG=pi05_robocasa; CKPT=${SD_CKPT:-/home/weiland/ckpt_pi05_robocasa_pytorch} ;;
  pi05_libero_spatial|pi05_libero_10|pi05_libero_object|pi05_libero_goal)
    CFG=pi05_libero; CKPT=${SD_CKPT:-/home/weiland/.cache/openpi/openpi-assets/checkpoints/pi05_libero_pytorch} ;;
  *) echo "unknown env_id $ENV_ID"; exit 1 ;;
esac
case "$MODE" in
  plain) [ "$ARG" != "-" ] || { echo "plain needs exec_steps"; exit 1; }; MODEARG="--exec-steps $ARG" ;;
  full) MODEARG="" ;;
  shadow|warm) [ -f "$ARG" ] || { echo "cache yaml $ARG missing"; exit 1; }; MODEARG="--cache-config $ARG" ;;
  *) echo "mode must be shadow|plain|full|warm"; exit 1 ;;
esac
[ -e "$CKPT" ] || { echo "checkpoint $CKPT missing"; exit 1; }
CELL=$ARM
[ "$MODE" != "shadow" ] || CELL="shadow_$ENV_ID"
OUT="$OUT/pi05/$CELL"
mkdir -p /tmp/sdiag "$OUT"
NAME="sdsrv$P"
if ! mkdir "/tmp/sdiag/lock_$NAME" 2>/dev/null; then
  if tmux has-session -t "$NAME" 2>/dev/null && ss -tlnH "sport = :$P" | grep -q .; then
    echo "$NAME already listening"; exit 0
  fi
  echo "$NAME has a claim but is not ready; inspect /tmp/sdiag/$NAME.log" >&2; exit 1
fi
if tmux has-session -t "$NAME" 2>/dev/null; then
  ss -tlnH "sport = :$P" | grep -q . && exit 0
  echo "$NAME exists but is not listening" >&2; exit 1
fi
if ss -tlnH "sport = :$P" | grep -q .; then echo "port $P busy (not ours); refusing"; rmdir "/tmp/sdiag/lock_$NAME"; exit 1; fi
LOG=/tmp/sdiag/$NAME.log
rm -f "$LOG"
tmux new -s "$NAME" -d "cd $REPO && export HOME=$HOME PYTHONPATH=$REPO/src:$REPO OPENPI_SERVER_GPU_MEMORY_LOCK=0 && \
  $PY -m exp.step_diag.serve_diag_pi05 --mode $MODE --env-id $ENV_ID --arm-id $ARM --experiment-id $EXP \
    --diag-out $OUT --launch-id $LAUNCH --policy-dir $CKPT $MODEARG -- \
    --port $P policy:checkpoint --policy.config $CFG --policy.dir $CKPT 2>&1 | tee $LOG"
for _ in $(seq 1 90); do
  ss -tlnH "sport = :$P" | grep -q . && grep -q "STEP_DIAG arm=" "$LOG" 2>/dev/null && break
  sleep 10
done
printf '%s: %s | %s\n' "$P" "$(grep -m1 'STEP_DIAG arm=' "$LOG" 2>/dev/null || echo 'NO STEP_DIAG MARKER')" \
  "$(ss -tlnH "sport = :$P" | grep -q . && echo listening || echo NOT LISTENING)"

if ! ss -tlnH "sport = :$P" | grep -q . || ! grep -q "STEP_DIAG arm=" "$LOG"; then
  echo "server readiness failed: $LOG" >&2
  exit 1
fi
