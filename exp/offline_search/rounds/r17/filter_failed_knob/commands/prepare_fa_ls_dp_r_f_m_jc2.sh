#!/usr/bin/env bash
set -euo pipefail
# Continuation of fa_ls_dp_r_f_m_j on jupyter-ziyang10-02 with the shared-EGL worker code (10-19).
cd /home/weiland/projects/openpi; export PYTHONPATH=.:src
export R17_H100_NODE=jupyter-ziyang10-02 R17_H100_HOST=149.165.153.233 R17_DRIVER_PORT=23192
 .venv/bin/python -m exp.offline_search.rounds.r17.filter_failed.stage --run /data/mw_finetune/eval/fa_ls_dp_r_f_m_jc2
