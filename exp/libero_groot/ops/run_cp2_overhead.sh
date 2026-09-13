#!/usr/bin/env bash
# The real CP2 decision-overhead preflight for one suite (plan §3.11): replay
# the accepted cohort H5 through reconstruction -> stage 2 (outside the timed
# boundary) -> run_cp2_key_source -> check(CP2) against the verified library.
#
# usage: run_cp2_overhead.sh <suite> <ckpt> <library-pkl> <accepted-manifest> [repo] [out-dir]
set -uo pipefail

SUITE=${1:?suite}
CKPT=${2:?checkpoint}
LIB=${3:?cp2 library pkl}
ACC=${4:?accepted_shadow_manifest.json}
REPO=${5:-/data/openpi_lg}
OUT=${6:-$REPO/exp/libero_groot/data/actioncache/overhead_$SUITE}

G=/home/weiland/gr00t_n15
PY=/home/weiland/gr00t_n15_venv/.venv/bin/python
export HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1 NO_ALBUMENTATIONS_UPDATE=1
export PYTHONPATH=$G:$G/examples/Libero:$REPO/src:$REPO
mkdir -p "$OUT"
cd "$REPO" || exit 1

"$PY" exp/libero_groot/bench_cp2_overhead_groot.py --mode overhead \
  --suite "$SUITE" --checkpoint "$CKPT" --library-pkl "$LIB" \
  --accepted-manifest "$ACC" --out-dir "$OUT"
rc=$?
echo "[overhead] rc=$rc  verdict: $(grep -o '"verdict": "[a-z_]*"' "$OUT/overhead.json" 2>/dev/null || echo none)"
exit $rc
