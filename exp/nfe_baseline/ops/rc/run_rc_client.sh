#!/usr/bin/env bash
# One RoboCasa365 teacher-only evaluation cell: (teacher, k, lane) on this timan box.
#
# usage: run_rc_client.sh <teacher pi05|groot_tp> <k> <server host:port> <lane main|pnp> <workers> <gpu-ids csv> <out_root> [tag]
#
# Drives exp.robocasa365.run_ws_search with --cid teacher (the G-A1 / RIT
# teacher-only recipe): 13-task roster split into the main lane (8 tasks, no
# pins) and the pnp lane (5 PickPlace tasks, pinned objects), 50 episodes per
# task, layout/style 1/1, eval seeds 1,000,000 + idx, replan 5. Artifacts land
# in <out_root>/<teacher>/<lane>/k<k>/{journal,run_plan,summary}_nfek<k>-teacher__l1s1_<teacher>.json;
# the log ends with RCCLI_EXIT=<code>.
set -eu
TEACHER=${1:?teacher}; K=${2:?k}; SERVER=${3:?host:port}; LANE=${4:?main|pnp}
W=${5:?workers}; GPUS=${6:?gpu ids}; OUT=${7:?out root}; TAG=${8:-run}
REPO=${NFE_RC_REPO:-/scratch/zixuans8/nfe/openpi_nfe}
PY=${NFE_RC_PY:-/scratch/zixuans8/Isaac-GR00T/gr00t/eval/sim/robocasa365/robocasa365_uv/.venv/bin/python}
ENVCFG=${NFE_RC_ENV:-$REPO/exp/nfe_baseline/config/rc_timan.env}
PIN=$REPO/exp/robocasa365/config/pnp_pinned_objects.json
EPS=${NFE_RC_EPISODES:-50}
PREFIX=${NFE_RC_PREFIX:-nfek$K}
export HOME=${NFE_HOME:-/home/zixuans8}
MAIN_TASKS=CloseBlenderLid,CloseFridge,CoffeeSetupMug,OpenCabinet,OpenDrawer,OpenStandMixerHead,SlideDishwasherRack,TurnOnSinkFaucet
PNP_TASKS=PickPlaceCounterToCabinet,PickPlaceCounterToStove,PickPlaceDrawerToCounter,PickPlaceSinkToCounter,PickPlaceToasterToCounter
case "$LANE" in
  main) TASKS=${NFE_RC_TASKS:-$MAIN_TASKS}; EXTRA="" ;;
  pnp) TASKS=${NFE_RC_TASKS:-$PNP_TASKS}; EXTRA="--pinned-objects $PIN" ;;
  *) echo "lane must be main|pnp"; exit 1 ;;
esac
JD=$OUT/$TEACHER/$LANE/k$K
NAME="nfercli_${TAG}_${TEACHER}_k${K}_${LANE}"
mkdir -p "$JD" /tmp/nfe
tmux kill-session -t "$NAME" 2>/dev/null || true
tmux new -s "$NAME" -d "cd $REPO && PYTHONPATH=$REPO/src:$REPO $PY -m exp.robocasa365.run_ws_search \
  --teacher $TEACHER --server $SERVER --cid teacher --run-prefix $PREFIX --tasks $TASKS --episodes $EPS \
  --layout 1 --style 1 --base-seed 1000000 --replan-steps 5 $EXTRA --env-config $ENVCFG \
  --workers $W --gpu-ids $GPUS --role all --journal-dir $JD > /tmp/nfe/$NAME.log 2>&1; \
  echo RCCLI_EXIT=\$? >> /tmp/nfe/$NAME.log"
echo "started $NAME -> $JD"
