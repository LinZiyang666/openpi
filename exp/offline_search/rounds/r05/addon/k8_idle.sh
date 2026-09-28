#!/bin/bash
# Task #40: re-measure K8 search latency on the idle machine (after the last R5 chain ends). Runs 5/6, new files only.
cd /home/weiland/projects/openpi
while tmux has-session -t relay_R5A5 2>/dev/null; do sleep 60; done
echo "K8_IDLE_START $(date -Is)"; uptime
K8=exp/offline_search/rounds/r04/k8_search_latency
CFGS="pi05_spatial_50_AWM pi05_spatial_50_MixedJudge pi05_spatial_50_BlindAWM pi05_spatial_500_AWM pi05_spatial_500_MixedJudge pi05_spatial_500_BlindAWM pi05_l10_50_AWM pi05_l10_50_MixedJudge pi05_l10_50_BlindAWM pi05_l10_500_AWM pi05_l10_500_MixedJudge pi05_l10_500_BlindAWM groot_spatial_50_AWM groot_spatial_500_AWM groot_l10_50_AWM groot_l10_500_AWM K7_r4k7_p_l10_500_tail1ug K7_r4k7_p_l10_50_tail1ug K7_r4k7_p_sp_500_tail1ug K7_r4k7_p_l10_500_ph2g"
for run in 5 6; do for c in $CFGS; do
  echo "START $c $run $(date +%T)"
  taskset -c 34-37,78-81 env OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1 CUDA_VISIBLE_DEVICES= PYTHONDONTWRITEBYTECODE=1 PYTHONPATH=.:src .venv/bin/python $K8/benchmark.py --config $c --run $run --profile || echo "ERROR $c $run"
done; done
echo "K8_IDLE_DONE $(date -Is)"
