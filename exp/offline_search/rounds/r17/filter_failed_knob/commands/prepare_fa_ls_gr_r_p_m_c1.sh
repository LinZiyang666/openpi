#!/usr/bin/env bash
# coordinator 10-18: continuation of fa_ls_gr_r_p_m (939 accepted kept; remaining 1,561), astra slot4 fix
set -euo pipefail
cd /home/weiland/projects/openpi; export PYTHONPATH=.:src
.venv/bin/python -m exp.offline_search.rounds.r17.filter_failed.resume --source /data/mw_finetune/eval/fa_ls_gr_r_p_m --name fa_ls_gr_r_p_m_c1 --remaining --verify-closed
.venv/bin/python -m exp.offline_search.rounds.r17.filter_failed.stage --run /data/mw_finetune/eval/fa_ls_gr_r_p_m_c1
