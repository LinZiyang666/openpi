#!/usr/bin/env bash
set -euo pipefail
cd /home/weiland/projects/openpi
export R17_H100_NODE=jupyter-ziyang10 R17_H100_HOST=149.165.153.233 R17_DRIVER_PORT=23193
/home/weiland/projects/openpi/.venv/bin/python -m exp.offline_search.rounds.r17.filter_failed_knob.launch --run /data/mw_finetune/eval/fa_lo_gr_r_f_m_j
