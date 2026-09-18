#!/bin/bash
# Source-able env for running exp.dp_nfe.* inside the official DP env on weilandserver / h100 (by hostname).
case "$(hostname)" in
  weilandserver*) export DP_BASE=/home/weiland/dp; export DP_DATA=/data/dp ;;
  *) export DP_BASE=/data/dp_h100; export DP_DATA=/data/dp_h100 ;;
esac
export DP_ROOT=$DP_BASE/diffusion_policy
export PATH=$DP_BASE/env/bin:/usr/local/bin:/usr/bin:/bin
export MUJOCO_PY_MUJOCO_PATH=$DP_BASE/mujoco210 LD_LIBRARY_PATH=$DP_BASE/mujoco210/bin:$DP_BASE/env/lib:/usr/lib/nvidia
export MUJOCO_GL=egl PYOPENGL_PLATFORM=egl
export PYTHONPATH=$DP_BASE/openpi_exp:$DP_ROOT
export X0_CODE=$DP_BASE/openpi_exp
export X0_DATA=$DP_DATA/x0_multimodal
mkdir -p $X0_DATA/subsets $X0_DATA/runs $X0_DATA/results_trailing $X0_DATA/diagnostics /tmp/x0
