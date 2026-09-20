#!/usr/bin/env bash
# One GR00T N1.5 server process of the step-vs-warm-start line (h100 for RC, weilandserver for LIBERO).
#
# usage: serve_groot.sh <env_id> <mode shadow|plain|full|warm> <arm_id> <port> <exec_steps | cache_yaml | ->
#   env:  SD_REPO SD_GROOT SD_GROOT_PY SD_CKPT SD_EXP SD_OUT SD_LAUNCH SD_HOME (defaults = weilandserver)
#
# RC (groot_rc): serve_groot_n15 without --concurrent = single connection per process (one worker).
# LIBERO (groot_libero_*): shadow only, rides the production --rit-shadow-out factory, which needs
# --concurrent; the served policy holds one DiagSession per connection and the infer lock
# serialises the model, so several LIBERO client processes may share one server.
# Same idempotent claim / readiness contract as serve_pi05.sh.
set -u
ENV_ID=${1:?env_id}; MODE=${2:?mode}; ARM=${3:?arm_id}; P=${4:?port}; ARG=${5:--}
REPO=${SD_REPO:-/data/openpi_sdiag}
GROOT=${SD_GROOT:-/home/weiland/gr00t_n15}
PY=${SD_GROOT_PY:-/home/weiland/gr00t_n15_venv/.venv/bin/python}
EXP=${SD_EXP:-sdiag_v1}
OUT=${SD_OUT:-$REPO/exp/step_diag/data/server}
LAUNCH=${SD_LAUNCH:-$(date +%Y%m%dT%H%M%S)_$P}
export PATH=/usr/local/bin:/usr/bin:/bin
export HOME=${SD_HOME:-/home/weiland}
case "$ENV_ID" in
  groot_rc) BENCH=rc; PYPATH=$GROOT:$REPO/src:$REPO; EXTRA=""
    CKPT=${SD_CKPT:-/home/weiland/ckpt_n15_robocasa_tp/gr00t_n1-5/foundation_model_learning/target_posttraining/atomic_seen/checkpoint-60000} ;;
  groot_libero_spatial|groot_libero_10) BENCH=libero; PYPATH=$GROOT:$GROOT/examples/Libero:$REPO:$REPO/src; EXTRA="--concurrent"
    SUITE=${ENV_ID#groot_libero_}
    CKPT=${SD_CKPT:-/home/weiland/ckpt_n15_libero_$SUITE} ;;
  *) echo "unknown env_id $ENV_ID"; exit 1 ;;
esac
case "$MODE" in
  plain) [ "$ARG" != "-" ] || { echo "plain needs exec_steps"; exit 1; }; MODEARG="--exec-steps $ARG" ;;
  full) MODEARG="" ;;
  shadow|warm) [ -f "$ARG" ] || { echo "cache yaml $ARG missing"; exit 1; }; MODEARG="--cache-config $ARG" ;;
  *) echo "mode must be shadow|plain|full|warm"; exit 1 ;;
esac
[ -d "$CKPT" ] || { echo "checkpoint dir $CKPT missing"; exit 1; }
CELL=$ARM
[ "$MODE" != "shadow" ] || CELL="shadow_$ENV_ID"
OUT="$OUT/groot_tp/$CELL"
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
tmux new -s "$NAME" -d "cd $REPO && export HOME=$HOME HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1 NO_ALBUMENTATIONS_UPDATE=1 \
  PYTHONPATH=$PYPATH && \
  $PY -m exp.step_diag.serve_diag_groot --benchmark $BENCH --mode $MODE --env-id $ENV_ID --arm-id $ARM \
    --experiment-id $EXP --diag-out $OUT --launch-id $LAUNCH $MODEARG -- \
    --checkpoint $CKPT --port $P $EXTRA 2>&1 | tee $LOG"
for _ in $(seq 1 90); do
  ss -tlnH "sport = :$P" | grep -q . && grep -q "STEP_DIAG arm=" "$LOG" 2>/dev/null && break
  sleep 10
done
printf '%s: %s | %s | %s\n' "$P" "$(grep -m1 'STEP_DIAG arm=' "$LOG" 2>/dev/null || echo 'NO STEP_DIAG MARKER')" \
  "$(grep -m1 -o 'STEP_DIAG num_inference_timesteps=[0-9]*' "$LOG" 2>/dev/null || echo 'live steps: see log')" \
  "$(ss -tlnH "sport = :$P" | grep -q . && echo listening || echo NOT LISTENING)"

if ! ss -tlnH "sport = :$P" | grep -q . || ! grep -q "STEP_DIAG arm=" "$LOG"; then
  echo "server readiness failed: $LOG" >&2
  exit 1
fi
