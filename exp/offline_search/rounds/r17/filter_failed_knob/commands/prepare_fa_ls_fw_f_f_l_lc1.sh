#!/usr/bin/env bash
set -euo pipefail
# H100 continuation of fa_ls_fw_f_f_l_l (17,455 = remaining + 41 retry-exhausted): the source ran without the
# Fast-WAM text table (2x26 GB), H100 renderers filled the rest and the replicas hit CUDA OOM; stopped 10-22 00:46
# PARTIAL. resume.py --remaining --verify-closed; text table (bitwise equal on the H100), 44 GiB, server_threads 4.
cd /home/weiland/projects/openpi; export PYTHONPATH=.:src
unset R17_H100_NODE R17_H100_HOST R17_DRIVER_PORT
exec .venv/bin/python -m exp.offline_search.rounds.r17.filter_failed.stage --run /data/mw_finetune/eval/fa_ls_fw_f_f_l_lc1
