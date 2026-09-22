#!/usr/bin/env bash
# Fusion-weight ablation, GR00T lane: the coarse grid's own harness
# (exp/libero_groot/orchestrate_search.py) over the three ablation cells of one
# suite. Servers come up on this node per cell with --cache-config; the LIBERO
# clients run on timan107 from the harness's own shards. Results land as one
# per-episode JSON per cell, the same shape as the r1/r2 grid results.
#
# usage: fw_lane_groot.sh <suite> [ports csv] [workers per slot]
set -uo pipefail

SUITE=${1:?suite}
PORTS=${2:-23160,23161,23162}
WORKERS=${3:-12}
R=${FW_REPO:-/data/openpi_lg}
PY=${FW_PY:-/home/weiland/projects/openpi/.venv/bin/python}
BASE=/data/libero_cache/search/$SUITE
case "$SUITE" in
  libero_spatial) CKPT=/home/weiland/ckpt_n15_libero_spatial ;;
  libero_10)      CKPT=/home/weiland/ckpt_n15_libero_10 ;;
  *) echo "unknown suite $SUITE"; exit 2 ;;
esac
cd "$R" || exit 1
ls "$BASE/abl"/*.yaml >/dev/null || { echo "no cells under $BASE/abl"; exit 2; }
RESULTS=${FW_RESULTS:-$BASE/abl_results}
mkdir -p "$RESULTS"
echo "== [$SUITE] $(date -Is) cells: $(ls $BASE/abl/*.yaml | xargs -n1 basename | paste -sd' ' -)"
PYTHONPATH=$R:$R/src $PY exp/libero_groot/orchestrate_search.py \
  --yaml-dir "$BASE/abl" --results-dir "$RESULTS" \
  --suite "$SUITE" --checkpoint "$CKPT" \
  --ports "$PORTS" --workers "$WORKERS" --gpus 8 \
  --tasks 10 --trials 50 --expect 500 "${@:4}" 2>&1 | tee "$RESULTS/orchestrate.log"
rc=${PIPESTATUS[0]}
echo "== [$SUITE] ORCHESTRATE_EXIT=$rc $(date -Is)"
exit "$rc"
