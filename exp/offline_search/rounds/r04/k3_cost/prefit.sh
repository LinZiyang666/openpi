#!/bin/bash
# CPU-only; four same-library wrist metric + V7/guard fits. Existing artifacts are validated by later tests.
set -euo pipefail
cd /home/weiland/projects/openpi
K3_FITS=${K3_FITS:-/home/weiland/trace_runs/offline_search_store/derived/r04/k3_cost/fits}
K3_LOG=exp/offline_search/rounds/r04/k3_cost/results
mkdir -p "$K3_FITS"
for suite in spatial l10; do
  for lib in current big; do
    artifact=$K3_FITS/pi05_${suite}_${lib}.pkl
    [ -e "$artifact" ] && continue
    taskset -c 26-29,70-73 env OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1 \
      .venv/bin/python -m exp.offline_search.closed_loop.plugin \
      --os-method exp.offline_search.rounds.r04.k3_cost.method:WristMixedJudge \
      --os-kwargs "{\"lib\":\"$lib\",\"guards\":true,\"events\":\"none\"}" \
      --os-cell pi05_${suite}_cache --os-root /dev/shm/offline_search_store \
      --os-log-dir "$K3_LOG/prefit_${suite}_${lib}" --os-fit-artifact "$artifact" \
      > "$K3_LOG/prefit_${suite}_${lib}.log" 2>&1
  done
done
