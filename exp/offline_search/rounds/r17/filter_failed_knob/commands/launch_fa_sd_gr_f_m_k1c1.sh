#!/usr/bin/env bash
set -euo pipefail
# continuation of fa_sd_gr_f_m_k1 after the 10-24 pod restart (resume.py --remaining --verify-closed)
cd /home/weiland/projects/openpi; export PYTHONPATH=.:src
export R17_H100_NODE=jupyter-ziyang10 R17_H100_HOST=149.165.153.233 R17_DRIVER_PORT=23193 R17_PUBLIC_PORT=23470
exec .venv/bin/python -m exp.offline_search.rounds.r17.filter_failed_knob.launch --run /data/mw_finetune/eval/fa_sd_gr_f_m_k1c1
