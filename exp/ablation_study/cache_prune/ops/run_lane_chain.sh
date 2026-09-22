#!/usr/bin/env bash
# Run a lane's arm families back to back, so a finished family does not leave
# the fleet idle. Each family is its own run_lane_direct.sh invocation with its
# own journal; a failed family does not stop the next one, and re-running the
# chain resumes every family from its journal.
#
# usage: HOME=<node home> run_lane_chain.sh <lane> <host> <driver-port> <ports csv> <workers-per-endpoint> <suite:regime> [...]
set -uo pipefail
LANE=${1:?}; HOST=${2:?}; DP=${3:?}; PORTS=${4:?}; W=${5:?}; shift 5
OPS=/home/weiland/projects/openpi/exp/ablation_study/cache_prune/ops
for fam in "$@"; do
  suite=${fam%%:*}; regime=${fam##*:}
  echo "===== $(date -Is) [$LANE] $suite/$regime ====="
  bash "$OPS/run_lane_direct.sh" "$LANE" "$HOST" "$DP" "$PORTS" "$W" "$suite" "$regime"
  echo "===== $(date -Is) [$LANE] $suite/$regime exit=$? ====="
done
echo "CHAIN_DONE $LANE"
