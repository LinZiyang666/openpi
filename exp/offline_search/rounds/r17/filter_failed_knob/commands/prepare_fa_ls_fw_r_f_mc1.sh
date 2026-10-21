#!/usr/bin/env bash
set -euo pipefail
# h100-temp continuation of fa_ls_fw_r_f_m (stopped 10-20 22:06 after ~120 episodes so the A100 and its MetaWorld env join the worker pool).
cd /home/weiland/projects/openpi; export PYTHONPATH=.:src
export R17_H100_NODE=h100-temp R17_H100_HOST=149.165.172.126 R17_DRIVER_PORT=23194
exec .venv/bin/python -m exp.offline_search.rounds.r17.filter_failed.stage --run /data/mw_finetune/eval/fa_ls_fw_r_f_mc1
