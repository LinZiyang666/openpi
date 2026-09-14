#!/usr/bin/env bash
# Noise floor of one suite on weilandserver (GR00T island venv), in tmux with a tee'd log.
#
# usage: run_noise_floor.sh <suite> [tmux-name]
set -eu
export HOME=/home/weiland
SUITE=${1:?suite}
NAME=${2:-lotonf_$SUITE}
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
tmux kill-session -t "$NAME" 2>/dev/null || true
tmux new -s "$NAME" -d "cd $REPO && PYTHONPATH=$G:$G/examples/Libero:$REPO:$REPO/src HF_HUB_OFFLINE=1 \
  $PY -m exp.rit_loto.noise_floor --suite $SUITE --corpus-dir $CORPUS --library-pkl $LIB \
  --template-yaml $TPL --checkpoint $CKPT --out-dir $OUT 2>&1 | tee /tmp/$NAME.log; echo NOISE_FLOOR_EXIT=\${PIPESTATUS[0]} | tee -a /tmp/$NAME.log"
echo "started tmux $NAME -> /tmp/$NAME.log ; products in $OUT"
