#!/usr/bin/env bash
# Full-cell vectorized reference (_ref_check.py) of every batch.json variant (V5: inf cells only; on cache cells it
# equals V4). Usage: bash _ref_all.sh [procs] [threads]
set -u
cd /home/weiland/projects/openpi
P=${1:-4}
TH=${2:-4}
OUT=/home/weiland/trace_runs/offline_search_store/derived/r02/g2_pcacos/ref
mkdir -p $OUT
.venv/bin/python - > $OUT/jobs.txt <<'PY'
import json
cells = [f"{m}_{s}_{a}" for m in ("pi05", "groot") for s in ("spatial", "l10") for a in ("inf", "cache")]
for e in json.load(open("exp/offline_search/rounds/r02/g2_pcacos/batch.json")):
    cls = e["method"].split(":")[1]
    for c in cells:
        if cls == "PcaCosV5Fresh" and c.endswith("_cache"):
            continue
        print(cls, c, json.dumps(e["kwargs"], separators=(",", ":")), sep="\t")
PY
run_one() {
  IFS=$'\t' read -r cls cell kw <<< "$1"
  tag=$(echo "$cls:$kw:$cell" | md5sum | cut -c1-10)
  OMP_NUM_THREADS=$TH OPENBLAS_NUM_THREADS=$TH taskset -c 9-17,53-61 .venv/bin/python \
      exp/offline_search/rounds/r02/g2_pcacos/_ref_check.py --cell $cell --cls $cls --kwargs "$kw" --out $OUT/$tag.json \
      > $OUT/$tag.log 2>&1
  echo "$tag rc=$? $cls $cell $kw"
}
export -f run_one
export OUT TH
tr '\n' '\0' < $OUT/jobs.txt | xargs -0 -P $P -I{} bash -c 'run_one "$@"' _ {}
