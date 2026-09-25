#!/usr/bin/env bash
# One LIBERO cell of the self-start round (sdiag_libero_self, logs/step_diag_libero_selfstart_plan.log.md):
# one (env, arm) on a task subset, driver + one worker per server on a timan box.
#
# usage: run_lib_cell.sh <env_id> <arm_id> <servers host:port,...> <task ids csv> <config_sha> <tag>
#   env:  SD_LIB_REPO SD_LIB_PY SD_LIB_CONFIG SD_EXP SD_GPUS SD_TMUX_SOCKET
#         SD_OUT_ROOT (--out-root), SD_RUN_PREFIX (--run-prefix), SD_EPISODES (--episodes, smoke ids only)
#
# The initial states are the frozen pruned A pool <repo>/exp/common/data/db_init/libero/<suite>_apool.
# Artifacts: <out root>/<teacher>/<env_id>/<arm_id>/{journal,run_plan,summary,launch,per_step}_*; the log
# /tmp/sdiag/<session>.log ends with SDCELL_EXIT=<code> (POSIX redirect: tmux on the timan boxes runs dash).
set -eu
ENV_ID=${1:?env_id}; ARM=${2:?arm_id}; SERVERS=${3:?servers}; TASKS=${4:?task ids}; CFGSHA=${5:?config_sha}
TAG=${6:?tag}
REPO=${SD_LIB_REPO:-/scratch/zixuans8/step_diag/openpi_lib}
PY=${SD_LIB_PY:-/scratch/zixuans8/libero_sim/bin/python}
LIBCFG=${SD_LIB_CONFIG:-/home/zixuans8/.libero}
EXP=${SD_EXP:-sdiag_libero_self}
GPUS=${SD_GPUS:-0}
case "$ENV_ID" in
  pi05_libero_spatial|groot_libero_spatial) SUITE=libero_spatial ;;
  pi05_libero_10|groot_libero_10) SUITE=libero_10 ;;
  *) echo "not a LIBERO self-start environment: $ENV_ID"; exit 1 ;;
esac
POOL=$REPO/exp/common/data/db_init/libero/${SUITE}_apool
[ -d "$POOL" ] || { echo "missing pool $POOL"; exit 1; }
EXTRA=()
[ -z "${SD_OUT_ROOT:-}" ] || EXTRA+=(--out-root "$SD_OUT_ROOT")
[ -z "${SD_RUN_PREFIX:-}" ] || EXTRA+=(--run-prefix "$SD_RUN_PREFIX")
[ -z "${SD_EPISODES:-}" ] || EXTRA+=(--episodes "$SD_EPISODES")
# tmux refuses "." in session names (arm ids such as warm_t0.875)
NAME="sdlib_${TAG}_${ARM//./_}_${EXP}"
mkdir -p /tmp/sdiag
# SD_TMUX_SOCKET: a private tmux server, so another session ending the default server cannot end the cell
TMUXC=(tmux); [ -n "${SD_TMUX_SOCKET:-}" ] && TMUXC=(tmux -L "$SD_TMUX_SOCKET")
if "${TMUXC[@]}" has-session -t "=$NAME" 2>/dev/null; then echo "$NAME already running; leaving it"; exit 0; fi
CMD=(env "PYTHONPATH=$REPO:$REPO/src" "LIBERO_CONFIG_PATH=$LIBCFG" MUJOCO_GL=egl
  "$PY" -m exp.step_diag.run_libero_diag --env-id "$ENV_ID" --arm-id "$ARM" --experiment-id "$EXP"
  --servers "$SERVERS" --pool "$POOL" --config-sha "$CFGSHA" --tasks "$TASKS" --gpu-ids "$GPUS"
  ${EXTRA[@]+"${EXTRA[@]}"})
printf -v COMMAND '%q ' "${CMD[@]}"
printf -v WORKDIR '%q' "$REPO"
printf -v LOG '%q' "/tmp/sdiag/$NAME.log"
printf -v SCRIPT '%s' "cd $WORKDIR && $COMMAND > $LOG 2>&1; rc=\$?; echo SDCELL_EXIT=\$rc >> $LOG; exit \$rc"
# tmux's default shell may be dash; explicitly use bash for printf %q escapes.
printf -v LAUNCH_CMD '%q ' bash -lc "$SCRIPT"
"${TMUXC[@]}" new -s "$NAME" -d "$LAUNCH_CMD"
echo "started $NAME -> /tmp/sdiag/$NAME.log"
