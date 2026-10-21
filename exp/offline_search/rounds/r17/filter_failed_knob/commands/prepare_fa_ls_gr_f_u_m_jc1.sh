#!/usr/bin/env bash
set -euo pipefail
# Continuation of fa_ls_gr_f_u_m_j (stopped 10-20 23:51) on jupyter-ziyang10-02 with 1 replica / 14 GiB:
# two 7 GiB GR00T replicas overflowed the MIG slice (NVML INTERNAL ASSERT / CUDNN_STATUS_INTERNAL_ERROR).
cd /home/weiland/projects/openpi
export R17_H100_NODE=jupyter-ziyang10-02 R17_H100_HOST=149.165.153.233 R17_DRIVER_PORT=23192
/home/weiland/projects/openpi/.venv/bin/python -m exp.offline_search.rounds.r17.filter_failed.stage --run /data/mw_finetune/eval/fa_ls_gr_f_u_m_jc1
