#!/usr/bin/env bash
# Run a lane's matrices back to back, so a finished group does not leave the
# fleet idle waiting on a human.
#
# Each matrix is a separate ``run_gtp`` invocation because the runner validates
# one gate and one judge shape per run: the anchors are ungated and the sweep
# arms sit behind the hysteresis gate. Resume is by journal inside each group,
# so re-running this chain after a crash re-enters whichever group still has
# work and skips the episodes already recorded.
#
# usage: run_eval_chain.sh <suite> <servers> <workers> <gpus> [matrices...]
set -uo pipefail

SUITE=${1:?suite}
SERVERS=${2:?servers}
WORKERS=${3:?workers}
GPUS=${4:?gpus}
shift 4
MATRICES=${*:-anchor hg}

R=/scratch/zixuans8/openpi_lg
OPS=$R/exp/libero_groot/ops

for M in $MATRICES; do
  D=$R/exp/libero_groot/config/rit/$SUITE
  if [ ! -f "$D/arm_matrix_${M}.yaml" ]; then
    echo "== $SUITE/$M: no matrix yet, skipping"
    continue
  fi
  echo "===== $(date -Is) $SUITE matrix $M ====="
  bash "$OPS/run_eval_group.sh" "$SUITE" "$M" "$SERVERS" "$WORKERS" "$GPUS"
  rc=$?
  echo "===== $(date -Is) $SUITE matrix $M exit=$rc ====="
  # A non-zero group is not a reason to skip the rest: the journal keeps what
  # completed and the chain can be re-run, but leaving the remaining matrices
  # unattempted would idle the lane until someone notices.
done
echo "LG_CHAIN_DONE $SUITE"
