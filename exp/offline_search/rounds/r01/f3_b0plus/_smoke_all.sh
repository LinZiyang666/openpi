#!/bin/bash
# Smoke every batch.json variant on the 4 spatial cells (+ M8 on two l10 cells for latency); <= 12 procs pinned to
# the CPUs allotted to this agent. Env: CELLS, NAMES (regex on the variant name), EPIS, PROCS, CPUS. Text outputs -> smoke/<name>__<cell>.txt; npz -> derived dir.
cd /home/weiland/projects/openpi
FAM=exp/offline_search/rounds/r01/f3_b0plus
OUT=/home/weiland/trace_runs/offline_search_store/derived/r01/f3_b0plus/smoke
CPUS=${CPUS:-12-17,56-61}
EPIS=${EPIS:-5}
CELLS=${CELLS:-"pi05_spatial_inf pi05_spatial_cache groot_spatial_inf groot_spatial_cache"}
.venv/bin/python - "$FAM" "$CELLS" "${NAMES:-.}" > /tmp/f3_smoke_cmds.txt <<'PY'
import json, re, sys
fam, cells = sys.argv[1], sys.argv[2].split()
sys.path.insert(0, fam)
from exp.offline_search.harness import run
for e in json.load(open(f"{fam}/batch.json")):
    cls, _ = run.load_method_class(e["method"])
    name = run.build_method(cls, e["kwargs"]).name
    if not re.search(sys.argv[3], name):
        continue
    for c in cells:
        print(f"{e['method']}\t{json.dumps(e['kwargs'])}\t{c}\t{name}")
PY
while IFS=$'\t' read -r meth kw cell name; do
  printf '%s\0' "taskset -c $CPUS .venv/bin/python -m exp.offline_search.harness.smoke --method $meth --kwargs '$kw' --cell $cell --episodes $EPIS --root /dev/shm/offline_search_store --out $OUT/${cell} > $FAM/smoke/${name}__${cell}.txt 2>&1"
done < /tmp/f3_smoke_cmds.txt | xargs -0 -P ${PROCS:-12} -I{} bash -c '{}'
echo ALLDONE
