#!/usr/bin/env bash
set -euo pipefail
export R17_H100_NODE=h100-sean R17_H100_HOST=149.165.169.137 R17_DRIVER_PORT=23198 R17_PUBLIC_PORT=23280
cd /home/weiland/projects/openpi
/home/weiland/projects/openpi/.venv/bin/python -m exp.offline_search.rounds.r17.filter_failed.stage --run /data/mw_finetune/eval/fa_sd_dp_r_l_k5
