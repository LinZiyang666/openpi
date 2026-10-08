#!/usr/bin/env bash
# Commands for the coordinator after source deployment and fleet availability.
# This file is syntax-checked by sol; no sync/chain/collect is executed by sol.
set -euo pipefail
cd /home/weiland/projects/openpi
ACTION=${1:?usage: coordinator_commands.sh plan|sync|chain|collect local|h100 pilot|test-a [rep0..rep1]}
FLEET=${2:?local or h100}
PHASE=${3:?pilot or test-a}
REPLICATE=${4:-}
case "$ACTION" in plan|sync|chain|collect) ;; *) exit 2 ;; esac
case "$REPLICATE" in ''|rep0|rep1) ;; *) exit 2 ;; esac
case "$FLEET" in
  local)
    export SERVER_HOST=local WORKER_HOST=timan107
    export LOCAL_SERVER_CPUS=22-37,66-81 LOCAL_SERVER_COUNT=4
    export PORTS=${PORTS:-23140,23141,23142,23143} SYNC_PORT=${SYNC_PORT:-23197}
    PIN=@local
    ;;
  h100)
    export SERVER_HOST=h100 WORKER_HOST=timan108
    export PORTS=${PORTS:-23240,23241,23242,23243} SYNC_PORT=${SYNC_PORT:-23198}
    PIN=@h100
    ;;
  *) exit 2 ;;
esac
export WPS=${WPS:-4} OMP=1 MAX_ATTEMPTS=3 POLL_SECONDS=60
export OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1 PYTHONDONTWRITEBYTECODE=1 PYTHONPATH=.:src
unset OSCL_MANIFEST OSCL_TASKS OSCL_EPISODES OSCL_INIT_POOL OSCL_POOL_CONTRACT OSCL_POOL_CONTRACT_SHA256
unset OSCL_RECORD_LOOKS OSCL_EXECUTION_AUDIT OSCL_ROLLOUT_SEED
IMPL=/home/weiland/projects/openpi/exp/offline_search/rounds/r14/impl
case "$PHASE" in pilot) QUEUE="$IMPL/queue_pilot.txt" ;; test-a) QUEUE="$IMPL/queue_test_a.txt" ;; *) exit 2 ;; esac
CONTROL=(taskset -c 22-37,66-81 env CUDA_VISIBLE_DEVICES= /home/weiland/projects/openpi/.venv/bin/python -B
         -m exp.offline_search.closed_loop.ops.h100.control)
while IFS= read -r LINE; do
  [[ ${LINE%% *} == "$PIN" ]] || continue
  PAYLOAD=${LINE#* }
  RUN_ROOT=${PAYLOAD%%|*}
  if [[ $PHASE == test-a && -n $REPLICATE && $RUN_ROOT != *_$REPLICATE ]]; then
    continue
  fi
  read -r -a ARM_NAMES <<< "${PAYLOAD#*|}"
  if [[ $ACTION == plan && $PHASE == test-a && ! -d $RUN_ROOT ]]; then
    RUN_ROOT="$IMPL/test_a_staged/${RUN_ROOT##*/}"
  fi
  if [[ $ACTION == sync ]]; then
    "${CONTROL[@]}" sync --concurrent "$RUN_ROOT" "${ARM_NAMES[@]}"
  elif [[ $ACTION == plan ]]; then
    "${CONTROL[@]}" plan --work-dir "$IMPL/coordinator_plans/$FLEET/${RUN_ROOT##*/}" "$RUN_ROOT" "${ARM_NAMES[@]}"
  else
    "${CONTROL[@]}" "$ACTION" "$RUN_ROOT" "${ARM_NAMES[@]}"
  fi
done < "$QUEUE"
