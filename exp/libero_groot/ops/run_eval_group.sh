#!/usr/bin/env bash
# Run one arm matrix of the LIBERO x GR00T RIT line from a sim box.
#
# The driver is ``run_gtp``: it is the runner the Pi0.5 K=3 line used, and it
# is the only one that accepts this arm shape (``threshold`` + declared warm
# tiers behind a hysteresis gate). ``run_conductor`` cannot: its
# ``verify_warm_sweep`` treats any directory containing a warm tier as the
# forced-warm-start sweep and demands ``always_warm_start`` on every cell.
#
# Driver and workers share this process: ``run_gtp`` starts a ConductorDriver
# and a WorkerAgent together, so the sim box is the only host that runs
# anything besides the servers. Resume is by journal, so re-running this after
# a crash walks only the arms with work left.
#
# usage: run_eval_group.sh <suite> <matrix anchor|hg> <servers> <workers> <gpus> [extra run_gtp args]
set -uo pipefail

SUITE=${1:?suite}
MATRIX=${2:?matrix}
SERVERS=${3:?host:port[,host:port...]}
WORKERS=${4:?workers}
GPUS=${5:?gpus}
shift 5

R=/scratch/zixuans8/openpi_lg
# The NFS home, not the tether agent's. LIBERO resolves its asset cache under
# ~/.cache/libero/assets and the agent's home carries only a partial copy; the
# 2026-09-04 false start was a whole group run against missing scenes.
export HOME=/home/zixuans8
export LIBERO_CONFIG_PATH=$HOME/.libero
export PYTHONPATH=$R/packages/openpi-client/src:$R/src:$R
export PATH=/scratch/zixuans8/dsp_bin:/usr/local/bin:/usr/bin:/bin
PY=/scratch/zixuans8/openpi/.venv/bin/python

case "$MATRIX" in
  anchor) GATE=always_search; WARM="" ;;
  hg)     GATE=score_hysteresis; WARM="--warm-tiers 0.75,0.5" ;;
  *) echo "matrix must be anchor|hg"; exit 2 ;;
esac

D=$R/exp/libero_groot/config/rit/$SUITE
OUT=$R/exp/libero_groot/data/rit/eval/${SUITE}_${MATRIX}
mkdir -p "$OUT"

cd "$R"
$PY -m exp.gate_threshold_pareto.run_gtp \
  --arm-matrix "$D/arm_matrix_${MATRIX}.yaml" \
  --phase eval \
  --task-suite "$SUITE" \
  --servers "$SERVERS" \
  --workers "$WORKERS" \
  --trials 50 \
  --gpus "$GPUS" \
  --conda-env /scratch/zixuans8/libero_sim \
  --judge-type threshold \
  --eval-gate "$GATE" $WARM \
  --resize-size 256 \
  --replan-steps 5 \
  --journal "$OUT/journal.jsonl" \
  --per-step-out "$OUT/per_step.jsonl" \
  --apool-record exp/ablation_study/cache_size/config/apool_${SUITE}.yaml \
  --apool-dir $R/exp/common/data/db_init/libero/${SUITE}_apool \
  "$@"
echo "LG_EVAL_EXIT=$?"
