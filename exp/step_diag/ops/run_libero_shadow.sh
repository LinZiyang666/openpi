#!/usr/bin/env bash
# usage: run_libero_shadow.sh <env_id> <servers csv> <pool_dir> <config_sha> [tag]
# One conductor worker per endpoint; supports both single-connection and concurrent servers.
set -euo pipefail
ENV_ID=${1:?env_id}; SERVERS=${2:?servers}; POOL=${3:?pool}; CFGSHA=${4:?config_sha}; TAG=${5:-shadow}
REPO=${SD_LIB_REPO:-/scratch/zixuans8/step_diag/openpi}
PY=${SD_LIB_PY:-/scratch/zixuans8/libero_sim/bin/python}
EXP=${SD_EXP:-sdiag_v1}
GPUS=${SD_GPUS:-0}
export LIBERO_CONFIG_PATH=${SD_LIB_CONFIG:-/home/zixuans8/.libero}
export MUJOCO_GL=egl
[ -d "$POOL" ] || { echo "missing pool $POOL"; exit 1; }
NAME="sdlib_${TAG}_${ENV_ID}"
mkdir -p /tmp/sdiag
if tmux has-session -t "$NAME" 2>/dev/null; then echo "$NAME already running"; exit 0; fi
CMD=(env "PYTHONPATH=$REPO:$REPO/src" "$PY" -m exp.step_diag.run_libero_diag
 --env-id "$ENV_ID" --servers "$SERVERS" --pool "$POOL" --config-sha "$CFGSHA"
 --experiment-id "$EXP" --gpu-ids "$GPUS")
printf -v COMMAND '%q ' "${CMD[@]}"
printf -v WORKDIR '%q' "$REPO"
printf -v LOG '%q' "/tmp/sdiag/$NAME.log"
printf -v SCRIPT '%s' "cd $WORKDIR && $COMMAND > $LOG 2>&1; rc=\$?; echo SDLIB_EXIT=\$rc >> $LOG; exit \$rc"
printf -v LAUNCH_CMD '%q ' bash -lc "$SCRIPT"
tmux new -s "$NAME" -d "$LAUNCH_CMD"
echo "started $NAME -> /tmp/sdiag/$NAME.log"
