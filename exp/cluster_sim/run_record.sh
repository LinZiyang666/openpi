#!/usr/bin/env bash
# Step 1 (after the 4090 is free): record the complete system's 500 LIBERO-10 episodes, then cut the replay library.
set -euo pipefail
cd /home/weiland/projects/openpi
export CLUSTER_SIM_RAW=/nvme/weiland/openpi/cluster_sim/raw
mkdir -p "$CLUSTER_SIM_RAW" /nvme/weiland/openpi/cluster_sim/logs
env -u CUDA_VISIBLE_DEVICES -u JAX_PLATFORMS OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1 PYTHONPATH=.:src \
  .venv/bin/python -m exp.cluster_sim.record_run run --group /data/mw_finetune/eval/cs_rec_pi05/group.json \
  --agent-island /scratch/zixuans8/openpi_r16_conductor_v1 --port 23159 --ports 23150,23151,23152 \
  2>&1 | tee -a /nvme/weiland/openpi/cluster_sim/logs/record_run.log
PYTHONPATH=.:src .venv/bin/python -m exp.cluster_sim.record split --raw "$CLUSTER_SIM_RAW" \
  --out /nvme/weiland/openpi/cluster_sim/library 2>&1 | tee -a /nvme/weiland/openpi/cluster_sim/logs/record_run.log
