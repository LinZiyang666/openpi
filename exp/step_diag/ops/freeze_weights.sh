#!/usr/bin/env bash
# Freeze the action weights (w / sigma / mask, plan §2.0) of every environment from its RIT library,
# on the host that holds the libraries. usage: freeze_weights.sh <out_dir> [env_id=library_pkl ...]
#   default bindings (weilandserver / h100 W13 RC libraries); LIBERO libraries are bound per call.
set -eu
OUT=${1:?out dir}; shift
REPO=${SD_REPO:-/data/openpi_sdiag}
PY=${SD_PY:-/home/weiland/openpi/.venv/bin/python}
W13=${SD_W13:-/data/robocasa365_cache/cache_artifacts_w13}
BIND=("$@")
[ ${#BIND[@]} -gt 0 ] || BIND=("pi05_rc=$W13/pi05_spatial_pool_16_w13_full.pkl" "groot_rc=$W13/groot_tp_spatial_pool_16_w13_full.pkl")
mkdir -p "$OUT"
for b in "${BIND[@]}"; do
  ENV_ID=${b%%=*}; LIB=${b#*=}
  [ -f "$LIB" ] || { echo "library $LIB missing"; exit 1; }
  (cd "$REPO" && PYTHONPATH=$REPO/src:$REPO "$PY" -m exp.step_diag.analysis.analyze_shadow freeze-weights \
     --library "$LIB" --env-id "$ENV_ID" --out "$OUT/weights_$ENV_ID.npz")
done
