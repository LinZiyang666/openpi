#!/usr/bin/env bash
# coordinator 10-18 13:2x: continuation of fa_ls_gr_r_p_m (939 kept; the same 1,561 UIDs as c1, whose servers ran
# without the long-segment fix and was moved to trash). Definition already built on CPU with
# filter_failed.resume --remaining; the fixed runtime is written into the archive by formal_prestage.
set -euo pipefail
cd /home/weiland/projects/openpi; export PYTHONPATH=.:src
.venv/bin/python -m exp.offline_search.rounds.r17.filter_failed.stage --run /data/mw_finetune/eval/fa_ls_fw_r_u_l_lc1
