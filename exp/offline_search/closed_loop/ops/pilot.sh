#!/bin/bash
# pilot.sh <run-root> <suite: sp|l10> <arm> [<arm> ...]  -- R3 closed-loop PILOT: chain.sh restricted to the 5 trap
# tasks of the suite x ep_idx 0-19 (100 episodes per arm, ~4-6 min wall at 64 workers), then compare with
# ops/kpi.py --pilot against the R2 arm of the same cell (R2 pilot baseline: rounds/r03/h4_kpi/r02_kpi.md).
#
#   spatial (sp)  tasks 6,9,0,4,1      l10  tasks 0,4,6,8,7      episodes 0..19 (OSCL_TASKS / OSCL_EPISODES of chain.sh)
#
# Passed through from the environment (see chain.sh): PORTS (required), WPS, SERVER_CPUS, STAGE1_ONLY, NEED_MB,
# MAX_ATTEMPTS. Overrides: PILOT_TASKS / PILOT_EPISODES (comma lists), PILOT_DRY=1 prints the chain command only.
# Every arm must exist in <run-root>/arms.json with the suite given here, and its yaml must already be on timan107
# (ops/sync_remote.sh <run-root> <arms>; this script does not sync). Arms with state/<arm>.DONE are skipped by chain.sh,
# so pilot arms need their own names (r3p_*), never an R2 arm name. Ports / tmux names follow chain.sh (never reuse a
# port or tmux session you did not create).
set -u
RUN=${1:?run-root}; shift
SUITE=${1:?suite sp|l10}; shift
[ $# -ge 1 ] || { echo "usage: pilot.sh <run-root> <sp|l10> <arm> [<arm> ...]" >&2; exit 2; }
HERE=$(cd "$(dirname "$0")" && pwd)
R=/home/weiland/projects/openpi

case "$SUITE" in
  sp|spatial|libero_spatial) SUITE_FULL=libero_spatial; TASKS_DEFAULT=6,9,0,4,1 ;;
  l10|libero_10)             SUITE_FULL=libero_10;      TASKS_DEFAULT=0,4,6,8,7 ;;
  *) echo "pilot.sh: suite must be sp | l10, got '$SUITE'" >&2; exit 2 ;;
esac
export OSCL_TASKS=${PILOT_TASKS:-$TASKS_DEFAULT}
export OSCL_EPISODES=${PILOT_EPISODES:-0,1,2,3,4,5,6,7,8,9,10,11,12,13,14,15,16,17,18,19}
: "${PORTS:?set PORTS (free ports in 23100-23199, comma list; scan: ss -ltnH | grep -oE ':231[0-9]{2}\b')}"
[ -f "$RUN/arms.json" ] || { echo "pilot.sh: $RUN/arms.json missing (ops/emit_arms.py first)" >&2; exit 2; }

nt=$(echo "$OSCL_TASKS" | tr ',' '\n' | grep -c .); ne=$(echo "$OSCL_EPISODES" | tr ',' '\n' | grep -c .)
rc=0
for arm in "$@"; do
  s=$(python3 -c "import json,sys; a={r['arm']:r for r in json.load(open(sys.argv[1]))}.get(sys.argv[2]); print(a['suite'] if a else 'MISSING')" "$RUN/arms.json" "$arm")
  if [ "$s" = MISSING ]; then echo "pilot.sh: arm $arm not in $RUN/arms.json" >&2; rc=1
  elif [ "$s" != "$SUITE_FULL" ]; then echo "pilot.sh: arm $arm is a $s arm, not $SUITE_FULL" >&2; rc=1; fi
  [ -f "$RUN/state/$arm.DONE" ] && echo "pilot.sh: note: $RUN/state/$arm.DONE exists -> chain.sh will skip $arm" >&2
done
[ $rc -eq 0 ] || exit 2

echo "pilot.sh: suite=$SUITE_FULL tasks=$OSCL_TASKS episodes=$OSCL_EPISODES expect=$((nt * ne))/arm ports=$PORTS wps=${WPS:-16}" \
     "server_cpus=${SERVER_CPUS:-34-37,78-81} stage1_only=${STAGE1_ONLY:-1} arms=$*"
echo "pilot.sh: afterwards: $R/.venv/bin/python -m exp.offline_search.closed_loop.ops.kpi --run-root $RUN --run-root <R2 run root>" \
     "$* --ref <R2 run>:<R2 arm> --pilot --md $RUN/kpi_pilot.md --json $RUN/kpi_pilot.json"
if [ "${PILOT_DRY:-0}" = "1" ]; then
  echo "DRY: OSCL_TASKS=$OSCL_TASKS OSCL_EPISODES=$OSCL_EPISODES PORTS=$PORTS bash $HERE/chain.sh $RUN $*"
  exit 0
fi
exec bash "$HERE/chain.sh" "$RUN" "$@"
