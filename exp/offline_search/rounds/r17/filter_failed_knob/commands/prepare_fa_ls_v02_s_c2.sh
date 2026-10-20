#!/usr/bin/env bash
set -euo pipefail
# h100-temp continuation of fa_ls_v02_s_c1 (stopped 10-19 19:37 by the machine owner; owner allowed resuming 10-20).
cd /home/weiland/projects/openpi; export PYTHONPATH=.:src
export R17_H100_NODE=h100-temp R17_H100_HOST=149.165.172.126 R17_DRIVER_PORT=23194
 .venv/bin/python -m exp.offline_search.rounds.r17.filter_failed.stage --run /data/mw_finetune/eval/fa_ls_v02_s_c2
