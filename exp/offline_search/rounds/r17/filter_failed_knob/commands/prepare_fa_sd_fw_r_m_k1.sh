#!/usr/bin/env bash
set -euo pipefail
export R17_H100_NODE='' R17_H100_HOST='' R17_DRIVER_PORT='' R17_PUBLIC_PORT=''
cd /home/weiland/projects/openpi
/home/weiland/projects/openpi/.venv/bin/python -m exp.offline_search.rounds.r17.filter_failed.stage --run /data/mw_finetune/eval/fa_sd_fw_r_m_k1
