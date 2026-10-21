#!/usr/bin/env bash
set -euo pipefail
# H100 rerun of the 3,855 fa_ls_fw_f_u_m episodes lost to EMFILE (proxy soft nofile 1024 at ~580 workers, 10-21 11:xx);
# resume.py --infra-errors; Fast-WAM text table (bitwise equal on the H100), 44 GiB, server_threads 4.
cd /home/weiland/projects/openpi; export PYTHONPATH=.:src
unset R17_H100_NODE R17_H100_HOST R17_DRIVER_PORT
exec .venv/bin/python -m exp.offline_search.rounds.r17.filter_failed_knob.launch --run /data/mw_finetune/eval/fa_ls_fw_f_u_mc1
