#!/bin/bash
R=/home/weiland/trace_runs/dual_20260923
REPO=/home/weiland/projects/openpi; G=/home/weiland/projects/openpi_ext/third_party/gr00t_n15
cd $REPO
export HOME=/home/weiland OMP_NUM_THREADS=4 HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1
export PYTHONPATH=$G:$G/examples/Libero:$REPO:$REPO/src:$REPO/packages/openpi-client/src
/home/weiland/projects/openpi_ext/envs/gr00t_n15_venv/.venv/bin/python $R/ops/replay_groot.py 2>&1 | grep --line-buffered -v -i 'warn' | tee $R/audit/replay_groot.log
echo "REPLAY_EXIT=${PIPESTATUS[0]}" | tee -a $R/audit/replay_groot.log
