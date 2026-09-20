#!/bin/bash
# Keep the evaluation queue running while training lanes are still producing checkpoints: repeat the eval queue every
# <sleep> seconds until it reports QUEUE DONE (every selected eval job verified) or <max hours> elapsed.
# usage: LANE=<name> x0_eval_loop.sh <arms> <sleep s> <max hours> [x0_queue eval args, e.g. --task-names pusht --parallel 2]
set -u
ARMS=${1:?arms}; SLEEP=${2:?sleep s}; MAXH=${3:?max hours}; shift 3
source "$(dirname "$0")/dp_x0_env.sh"
CELLS=$X0_DATA/cells; TASKS=$X0_CODE/exp/dp_nfe/config/x0_multimodal/tasks.yaml
LANE=${LANE:-eval}; STATE=$X0_DATA/queue_eval_$LANE.json; LOG=/tmp/x0/queue_eval_$LANE.log
t0=$(date +%s)
while :; do
  line=$(python -m exp.dp_nfe.x0_queue eval --tasks $TASKS --cells $CELLS --runs $X0_DATA/runs --results $X0_DATA/results_trailing --state $STATE --arms $ARMS --dp-root $DP_ROOT "$@" 2>&1 | tee -a $LOG | grep -E "^QUEUE" | tail -1)
  echo "$(date +%H:%M:%S) $line"
  case "$line" in
    "QUEUE DONE"*) echo "EVAL_LOOP_DONE $(date +%H:%M:%S)" | tee -a $LOG; exit 0 ;;
  esac
  # failed jobs with nothing pending/blocked left: terminal, report and stop (the failures stay in the ledger)
  if echo "$line" | grep -q "failed=" && ! echo "$line" | grep -qE "pending=|blocked="; then echo "EVAL_LOOP_FAILED_TERMINAL $(date +%H:%M:%S) $line" | tee -a $LOG; exit 2; fi
  if [ $(( $(date +%s) - t0 )) -gt $(( MAXH * 3600 )) ]; then echo "EVAL_LOOP_TIMEOUT $(date +%H:%M:%S)" | tee -a $LOG; exit 3; fi
  sleep "$SLEEP"
done
