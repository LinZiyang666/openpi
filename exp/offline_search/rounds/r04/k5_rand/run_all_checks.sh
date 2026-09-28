#!/usr/bin/env bash
# CPU only. Remove no artifacts belonging to another task; each run gets a fresh prefix.
# Existing blind tests append logs, so the fixed /tmp directories must be unused.
set -euo pipefail
cd /home/weiland/projects/openpi
base=exp/offline_search/rounds/r04/k5_rand
for path in /tmp/k5_existing_installed /tmp/k5_installed_replays /tmp/k5_overlay_installed; do
  if [[ -e "$path" ]]; then
    echo "Existing verification artifacts at $path; preserve or move them before rerunning this complete recipe." >&2
    exit 1
  fi
done
py=(taskset -c 26-29,70-73 env OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1 CUDA_VISIBLE_DEVICES= PYTHONDONTWRITEBYTECODE=1 PYTHONPATH=.:src /home/weiland/projects/openpi/.venv/bin/python)
"${py[@]}" "$base/prepare_arms.py" --run-root /tmp/k5_arms_check > "$base/results/arms_check.log" 2>&1
"${py[@]}" "$base/make_existing_checks.py"
for group in k2 k1 k4; do
  bash "$base/dev/existing_$group.sh" > "$base/results/existing_$group.log" 2>&1
done
"${py[@]}" "$base/test_overlay.py" --source installed --out /tmp/k5_overlay_installed > "$base/results/overlay_installed.log" 2>&1
"${py[@]}" "$base/check_parity.py" installed > "$base/results/parity_installed.log" 2>&1
"${py[@]}" "$base/run_random_replays.py" installed > "$base/results/replays_installed.log" 2>&1
"${py[@]}" "$base/validate_estimator.py" > "$base/results/estimator_validation.log" 2>&1
"${py[@]}" "$base/check_replay_estimator.py" > "$base/results/replay_estimator.log" 2>&1
"${py[@]}" "$base/check_ledger.py" > "$base/results/ledger_check.log" 2>&1
"${py[@]}" "$base/final_audit.py" > "$base/results/final_audit.log" 2>&1
"${py[@]}" "$base/summarize_checks.py" > "$base/results/existing_summary.log" 2>&1
