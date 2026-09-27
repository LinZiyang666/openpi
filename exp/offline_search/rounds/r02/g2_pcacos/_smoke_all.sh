#!/usr/bin/env bash
# Smoke every batch.json entry of g2_pcacos on pi05_spatial / groot_l10, inf + cache (5 episodes each), in parallel
# inside the family's CPU range.  Usage: bash _smoke_all.sh [cells] [episodes] [procs]
set -u
cd /home/weiland/projects/openpi
CELLS=${1:-pi05_spatial_inf,pi05_spatial_cache,groot_l10_inf,groot_l10_cache}
EPS=${2:-5}
P=${3:-12}
OUT=/home/weiland/trace_runs/offline_search_store/derived/r02/g2_pcacos/smoke/run
mkdir -p $OUT/logs
.venv/bin/python - "$CELLS" > $OUT/jobs.txt <<'PY'
import json, sys
cells = sys.argv[1].split(",")
for e in json.load(open("exp/offline_search/rounds/r02/g2_pcacos/batch.json")):
    for c in cells:
        print(e["method"], c, json.dumps(e["kwargs"], separators=(",", ":")), sep="\t")
PY
run_one() {
  IFS=$'\t' read -r meth cell kw <<< "$1"
  tag=$(echo "$meth:$kw:$cell" | md5sum | cut -c1-10)
  taskset -c 9-17,53-61 .venv/bin/python -m exp.offline_search.harness.smoke --method "$meth" --kwargs "$kw" \
      --cell "$cell" --episodes $EPS --root /dev/shm/offline_search_store --out $OUT/$cell > $OUT/logs/$tag.log 2>&1
  echo "$tag rc=$? $cell $kw"
}
export -f run_one
export EPS OUT
tr '\n' '\0' < $OUT/jobs.txt | xargs -0 -P $P -I{} bash -c 'run_one "$@"' _ {}
