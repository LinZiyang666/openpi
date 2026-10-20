#!/usr/bin/env bash
set -euo pipefail
# h100-temp continuation of fa_ls_v02_s_c2 (587 retry-exhausted rows from server CUDA OOM at R3, 10-20 18:15-18:31).
cd /home/weiland/projects/openpi; export PYTHONPATH=.:src
export R17_H100_NODE=h100-temp R17_H100_HOST=149.165.172.126 R17_DRIVER_PORT=23194
exec .venv/bin/python -m exp.offline_search.rounds.r17.filter_failed.launch --run /data/mw_finetune/eval/fa_ls_v02_s_c3
