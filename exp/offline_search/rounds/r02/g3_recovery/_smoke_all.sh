#!/usr/bin/env bash
# Harness smokes of every G3 class (5 episodes; pi0.5 + GR00T, inf + cache, current + big) -> smoke/<tag>.txt
# usage: taskset -c 18-26,62-70 bash exp/offline_search/rounds/r02/g3_recovery/_smoke_all.sh [parallel=3]
set -u
cd /home/weiland/projects/openpi
D=exp/offline_search/rounds/r02/g3_recovery
OUT=/home/weiland/trace_runs/offline_search_store/derived/r02/g3_recovery/smoke
mkdir -p $D/smoke $OUT
B=$D/standins.py:VZSKernel
jobs=()
add() {   # tag method kwargs cell
  jobs+=("$1|$2|$3|$4")
}
for lib in current big; do
  BK="{\"lib\": \"$lib\"}"
  WK="{\"base\": \"$B\", \"base_kwargs\": $BK}"
  if [ $lib = current ]; then c1=pi05_spatial; c2=groot_l10; else c1=groot_spatial; c2=pi05_l10; fi
  add base_${lib}_${c1}_cache $B "$BK" ${c1}_cache
  add base_${lib}_${c2}_inf $B "$BK" ${c2}_inf
  add v6_${lib}_${c1}_cache $D/wrappers.py:StuckRecovery "$WK" ${c1}_cache
  add v6_${lib}_${c1}_inf $D/wrappers.py:StuckRecovery "$WK" ${c1}_inf
  add v6_${lib}_${c2}_cache $D/wrappers.py:StuckRecovery "$WK" ${c2}_cache
  add v6_${lib}_${c2}_inf $D/wrappers.py:StuckRecovery "$WK" ${c2}_inf
  add v6blend_${lib}_${c2}_cache $D/wrappers.py:StuckRecovery "{\"base\": \"$B\", \"base_kwargs\": $BK, \"recover\": false}" ${c2}_cache
  add v7iso_${lib}_${c1}_inf $D/wrappers.py:DriftCalibratedConfidence "$WK" ${c1}_inf
  add v7iso_${lib}_${c2}_cache $D/wrappers.py:DriftCalibratedConfidence "$WK" ${c2}_cache
  add v7bin10_${lib}_${c2}_inf $D/wrappers.py:DriftCalibratedConfidence "{\"base\": \"$B\", \"base_kwargs\": $BK, \"calib\": \"bin10\"}" ${c2}_inf
done
run1() {
  IFS='|' read -r tag m kw cell <<< "$1"
  .venv/bin/python -m exp.offline_search.harness.smoke --method "$m" --kwargs "$kw" --cell "$cell" --episodes 5 \
    --root /dev/shm/offline_search_store --out $OUT/$tag > $D/smoke/$tag.txt 2>&1
  echo "$tag $(tail -1 $D/smoke/$tag.txt)"
}
export -f run1
export D OUT
printf '%s\n' "${jobs[@]}" | xargs -d '\n' -P "${1:-3}" -I{} bash -c 'run1 "$1"' _ {}
