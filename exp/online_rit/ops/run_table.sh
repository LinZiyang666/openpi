#!/usr/bin/env bash
# Build the M1 assets for one suite on the serving box (GR00T island venv):
# scales -> parity gate (self-replay + two-noise floor, must PASS) -> full table.
# Usage: run_table.sh <suite> <corpus_dir> <library_pkl> <checkpoint> [limit_episodes]
set -euo pipefail
SUITE=$1; CORPUS=$2; LIB=$3; CKPT=$4; LIMIT=${5:-0}
ROOT=${OPENPI_ROOT:-/data/openpi_lg}
PY=${GROOT_PY:-/home/weiland/gr00t_n15_venv/.venv/bin/python}
OUT=$ROOT/exp/online_rit/data/$SUITE/offline
TPL=exp/libero_groot/config/rit/$SUITE/template.yaml
mkdir -p "$OUT"
cd "$ROOT"
export PYTHONPATH=/home/weiland/gr00t_n15:/home/weiland/gr00t_n15/examples/Libero:$ROOT:$ROOT/src
$PY -m exp.online_rit.library_prep scales --library-pkl "$LIB" --out-dir "$OUT"
COMMON=(--suite "$SUITE" --corpus-dir "$CORPUS" --library-pkl "$LIB" --template-yaml "$TPL" \
        --scales "$OUT/update_scales.npz" --checkpoint "$CKPT" --out-dir "$OUT")
if [ "$LIMIT" -gt 0 ]; then
  $PY -m exp.online_rit.build_disagreement_table "${COMMON[@]}" --limit-episodes "$LIMIT" --allow-ungated-smoke \
    2>&1 | tee "$OUT/build_table.smoke.log"
else
  $PY -m exp.online_rit.build_disagreement_table "${COMMON[@]}" --parity-only 2>&1 | tee "$OUT/parity.log"
  grep -q "PARITY_GATE=PASS" "$OUT/parity.log" || { echo "parity gate did not PASS; see $OUT/parity_gate.json"; exit 3; }
  $PY -m exp.online_rit.build_disagreement_table "${COMMON[@]}" --parity-gate "$OUT/parity_gate.json" \
    2>&1 | tee "$OUT/build_table.log"
fi
