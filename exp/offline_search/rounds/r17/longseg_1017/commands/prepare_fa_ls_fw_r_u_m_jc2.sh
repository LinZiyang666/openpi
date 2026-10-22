#!/usr/bin/env bash
set -euo pipefail
cd /home/weiland/projects/openpi
export R17_H100_NODE=jupyter-ziyang10-02 R17_H100_HOST=149.165.153.233 R17_DRIVER_PORT=23192
/home/weiland/projects/openpi/.venv/bin/python -m exp.offline_search.rounds.r17.filter_failed.stage --run /data/mw_finetune/eval/fa_ls_fw_r_u_m_jc2
