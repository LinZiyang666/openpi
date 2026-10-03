#!/bin/bash
# run_group.sh <suite> <arm> <servers> <server-workers> <out-dir> [extra run_gtp args]
# run_gtp (driver + WorkerAgent in one process) on timan107 against the weilandserver trace servers.
set -uo pipefail
SUITE=${1:?suite}; ARM=${2:?arm}; SERVERS=${3:?servers}; SW=${4:?server-workers}; OUT=${5:?out}
shift 5
R=/scratch/zixuans8/openpi_trace
export HOME=/home/zixuans8
export LIBERO_CONFIG_PATH=$HOME/.libero
export PYTHONPATH=$R/packages/openpi-client/src:$R/src:$R
export PATH=/scratch/zixuans8/dsp_bin:/usr/local/bin:/usr/bin:/bin
PY=/scratch/zixuans8/openpi/.venv/bin/python
case "$ARM" in *groot*) RES=(--resize-size 256) ;; *) RES=() ;; esac
W=0; for n in $(echo "$SW" | tr ',' ' '); do W=$((W+n)); done
mkdir -p "$OUT"
cd "$R"
echo "RUN_GROUP_START $(date -Is) arm=$ARM servers=$SERVERS sw=$SW"
$PY -m exp.gate_threshold_pareto.run_gtp \
  --arm-matrix run_cfg/matrix_${ARM}.yaml \
  --phase eval \
  --task-suite "$SUITE" \
  --servers "$SERVERS" \
  --server-workers "$SW" \
  --workers "$W" \
  --trials 50 \
  --gpus 8 \
  --conda-env /scratch/zixuans8/libero_sim \
  --judge-type threshold \
  --eval-gate always_search \
  "${RES[@]}" \
  --replan-steps 5 \
  --journal "$OUT/journal.jsonl" \
  --per-step-out "$OUT/per_step.jsonl" \
  --apool-record exp/ablation_study/cache_size/config/apool_${SUITE}.yaml \
  --apool-dir $R/exp/common/data/db_init/libero/${SUITE}_apool \
  "$@"
rc=$?
echo "RUN_GROUP_EXIT=$rc $(date -Is)"
exit $rc
