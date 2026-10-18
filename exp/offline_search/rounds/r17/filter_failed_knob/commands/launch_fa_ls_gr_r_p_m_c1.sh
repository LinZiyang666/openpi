#!/usr/bin/env bash
set -euo pipefail
cd /home/weiland/projects/openpi; export PYTHONPATH=.:src
exec .venv/bin/python -m exp.offline_search.rounds.r17.filter_failed.launch --run /data/mw_finetune/eval/fa_ls_gr_r_p_m_c1
