#!/usr/bin/env bash
# One RoboCasa365 cell (teacher, arm, lane) of the step-vs-warm-start line, driver + agent on timan108.
#
# usage: run_rc_cell.sh <teacher pi05|groot_tp> <arm_id> <lane main|pnp> <servers host:port,...> \
#                       <tasks csv> <episodes> <episodes_map json | -> <config_sha> [tag]
#   env:  SD_RC_REPO SD_RC_PY SD_RC_ENV SD_EXP SD_BASE_SEED SD_HOME SD_GPUS
#
# Every server in <servers> runs single-connection, so one worker is bound per server
# (--workers-per-server 1) and the tasks are spread over the servers by the driver. Artifacts:
# $REPO/exp/step_diag/data/rc/<teacher>/<arm_id>/{journal,run_plan,summary,launch,per_step}_*; the
# log ends with SDCELL_EXIT=<code> (POSIX redirect: tmux on the timan boxes runs dash).
set -eu
TEACHER=${1:?teacher}; ARM=${2:?arm_id}; LANE=${3:?main|pnp}; SERVERS=${4:?servers}
TASKS=${5:?tasks csv}; EPS=${6:?episodes}; EPMAP=${7:--}; CFGSHA=${8:?config_sha}; TAG=${9:-run}
REPO=${SD_RC_REPO:-/scratch/zixuans8/step_diag/openpi}
PY=${SD_RC_PY:-/scratch/zixuans8/Isaac-GR00T/gr00t/eval/sim/robocasa365/robocasa365_uv/.venv/bin/python}
ENVCFG=${SD_RC_ENV:-$REPO/exp/step_diag/config/rc_timan.env}
EXP=${SD_EXP:-sdiag_v1}
SEED=${SD_BASE_SEED:-2000000}
GPUS=${SD_GPUS:-$(nvidia-smi --query-gpu=index --format=csv,noheader | paste -sd, -)}

PIN=$REPO/exp/robocasa365/config/pnp_pinned_objects.json
case "$LANE" in
  main) EXTRA=() ;;
  pnp) EXTRA=(--pinned-objects "$PIN") ;;
  *) echo "lane must be main|pnp"; exit 1 ;;
esac
[ "$EPMAP" = "-" ] || EXTRA+=(--episodes-map "$EPMAP")
NSRV=$(echo "$SERVERS" | tr ',' '\n' | grep -c .)
NAME="sdcell_${TAG}_${TEACHER}_${ARM}_${LANE}"
mkdir -p /tmp/sdiag
if tmux has-session -t "$NAME" 2>/dev/null; then echo "$NAME already running; leaving it"; exit 0; fi
CMD=(env "PYTHONPATH=$REPO/src:$REPO" "$PY" -m exp.step_diag.run_diag
  --teacher "$TEACHER" --servers "$SERVERS" --arm-id "$ARM" --experiment-id "$EXP" --lane "$LANE" --tasks "$TASKS"
  --episodes "$EPS" --base-seed "$SEED" --replan-steps 5 --config-sha "$CFGSHA" "${EXTRA[@]}" --env-config "$ENVCFG"
  --workers-per-server 1 --gpu-ids "$GPUS" --role all)
printf -v COMMAND '%q ' "${CMD[@]}"
printf -v WORKDIR '%q' "$REPO"
printf -v LOG '%q' "/tmp/sdiag/$NAME.log"
printf -v SCRIPT '%s' "cd $WORKDIR && $COMMAND > $LOG 2>&1; rc=\$?; echo SDCELL_EXIT=\$rc >> $LOG; exit \$rc"
# tmux's default shell may be dash; explicitly use bash for printf %q escapes.
printf -v LAUNCH_CMD '%q ' bash -lc "$SCRIPT"
tmux new -s "$NAME" -d "$LAUNCH_CMD"
echo "started $NAME ($NSRV server(s), 1 worker each) -> /tmp/sdiag/$NAME.log"
