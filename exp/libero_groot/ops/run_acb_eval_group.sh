#!/usr/bin/env bash
# Run one ActionCache-baseline (CP2) arm matrix of LIBERO x GR00T from a sim
# box: the same run_gtp driver and lane shape as run_eval_group.sh, with the
# CP2 contract switched on (--checkpoint cp2 validates every arm yaml against
# libs.cp2_contract_problems before any rollout) and the single warm tier this
# line uses (0.875 -> one Euler step).
#
# usage: run_acb_eval_group.sh <suite> <arm-matrix.yaml> <servers> <workers> <gpus> <run-dir> [extra run_gtp args]
set -uo pipefail

SUITE=${1:?suite}
MATRIX=${2:?arm_matrix.yaml}
SERVERS=${3:?host:port[,host:port...]}
WORKERS=${4:?workers}
GPUS=${5:?gpus}
RUN=${6:?run dir}
shift 6

R=/scratch/zixuans8/openpi_lg
# The NFS home, not the tether agent's (see run_eval_group.sh).
export HOME=/home/zixuans8
export LIBERO_CONFIG_PATH=$HOME/.libero
export PYTHONPATH=$R/packages/openpi-client/src:$R/src:$R
export PATH=/scratch/zixuans8/dsp_bin:/usr/local/bin:/usr/bin:/bin
PY=/scratch/zixuans8/openpi/.venv/bin/python

[ -f "$MATRIX" ] || { echo "missing arm matrix $MATRIX"; exit 2; }
mkdir -p "$RUN"

cd "$R"
$PY -m exp.gate_threshold_pareto.run_gtp \
  --arm-matrix "$MATRIX" \
  --phase eval \
  --checkpoint cp2 \
  --task-suite "$SUITE" \
  --servers "$SERVERS" \
  --workers "$WORKERS" \
  --trials 50 \
  --gpus "$GPUS" \
  --conda-env /scratch/zixuans8/libero_sim \
  --judge-type threshold \
  --eval-gate always_search --warm-tiers 0.875 \
  --resize-size 256 \
  --replan-steps 5 \
  --journal "$RUN/journal.jsonl" \
  --per-step-out "$RUN/per_step.jsonl" \
  --apool-record exp/ablation_study/cache_size/config/apool_${SUITE}.yaml \
  --apool-dir $R/exp/common/data/db_init/libero/${SUITE}_apool \
  "$@"
echo "ACB_EVAL_EXIT=$?"
