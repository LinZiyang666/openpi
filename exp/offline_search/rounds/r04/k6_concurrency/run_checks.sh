#!/bin/bash
set -euo pipefail
cd /home/weiland/projects/openpi
source=${1:?dev or installed}
base=exp/offline_search/rounds/r04/k6_concurrency
py=(taskset -c 26-29,70-73 env OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1 CUDA_VISIBLE_DEVICES= PYTHONDONTWRITEBYTECODE=1 PYTHONPATH=.:src /home/weiland/projects/openpi/.venv/bin/python)
"${py[@]}" "$base/prepare_checks.py" "$source"
for group in k2 k1 k4; do
  bash "$base/dev/${source}_${group}.sh" > "$base/results/$source/existing_${group}.log" 2>&1
  echo "PASS existing $group"
done
"${py[@]}" "$base/dev/${source}_test_overlay.py" --source "$source" --out "/tmp/k6_overlay_$source" > "$base/results/$source/overlay.log" 2>&1
echo 'PASS K5 overlay'
"${py[@]}" "$base/dev/${source}_check_parity.py" "$source" > "$base/results/$source/parity.log" 2>&1
echo 'PASS byte parity'
"${py[@]}" "$base/dev/${source}_run_random_replays.py" "$source" > "$base/results/$source/random_replays.log" 2>&1
echo 'PASS K5 four randomized replays'
