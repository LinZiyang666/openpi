#!/usr/bin/env bash
# LOTO shadow table of one suite on weilandserver (GR00T island venv), in tmux with a tee'd log.
#
# Modes:
#   parity  -> parity sample + parity_gate.json (needs noise_floor.json of the suite)
#   full    -> the whole table (needs a PASS parity_gate.json)
#   smoke N -> first N episodes, ungated, written as loto_table.smoke.jsonl
#
# usage: run_loto_table.sh <suite> parity|full|smoke [N] [extra args...]
set -eu
export HOME=/home/weiland
SUITE=${1:?suite}
MODE=${2:?parity|full|smoke}
shift 2
REPO=/data/openpi_lg
G=/home/weiland/gr00t_n15
PY=/home/weiland/gr00t_n15_venv/.venv/bin/python
case "$SUITE" in
  libero_spatial) CORPUS=/archive/libero_cache/build_spatial_w13/libero_spatial; CKPT=/home/weiland/ckpt_n15_libero_spatial ;;
  libero_10)      CORPUS=/archive/libero_cache/build_libero10_w13/libero_10;    CKPT=/home/weiland/ckpt_n15_libero_10 ;;
  *) echo "unknown suite $SUITE"; exit 2 ;;
esac
LIB=/data/libero_cache/libraries_w13/$SUITE/${SUITE}_w13_S3.pkl
TPL=$REPO/exp/libero_groot/config/rit/$SUITE/template.yaml
OUT=/data/libero_cache/rit_loto/$SUITE
mkdir -p "$OUT"
COMMON="--suite $SUITE --corpus-dir $CORPUS --library-pkl $LIB --template-yaml $TPL --checkpoint $CKPT --out-dir $OUT"
case "$MODE" in
  parity) ARGS="$COMMON --parity-only --noise-floor-record $OUT/noise_floor.json"; NAME=lotopar_$SUITE ;;
  full)   ARGS="$COMMON --parity-gate $OUT/parity_gate.json --orchestrator-check 200"; NAME=lototab_$SUITE ;;
  smoke)  N=${1:?smoke needs N}; shift; ARGS="$COMMON --limit-episodes $N --allow-ungated-smoke --orchestrator-check 5"; NAME=lotosmk_$SUITE ;;
  *) echo "mode must be parity|full|smoke"; exit 2 ;;
esac
tmux kill-session -t "$NAME" 2>/dev/null || true
tmux new -s "$NAME" -d "cd $REPO && PYTHONPATH=$G:$G/examples/Libero:$REPO:$REPO/src HF_HUB_OFFLINE=1 \
  $PY -m exp.rit_loto.build_loto_table $ARGS $* 2>&1 | tee /tmp/$NAME.log; echo LOTO_TABLE_EXIT=\${PIPESTATUS[0]} | tee -a /tmp/$NAME.log"
echo "started tmux $NAME ($MODE) -> /tmp/$NAME.log ; products in $OUT"
