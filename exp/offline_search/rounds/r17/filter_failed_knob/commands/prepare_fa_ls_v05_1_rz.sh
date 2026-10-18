#!/usr/bin/env bash
# coordinator 10-18 14:3x: fa_ls_v05_1 failed at launch (0 accepted) because worker_resilience.install_worker
# required worker_memory.reservation_mib, absent from its 10-10 frozen worker payload. Fixed; reset and re-prepare.
set -euo pipefail
cd /home/weiland/projects/openpi; export PYTHONPATH=.:src
CUDA_VISIBLE_DEVICES='' .venv/bin/python -m exp.offline_search.rounds.r17.autoscale_1017.reset_zero --run /data/mw_finetune/eval/fa_ls_v05_1 --backup /data/mw_finetune/eval/fa_ls_v05_1.before_retry_1430 --remote-backup /data/mw_finetune/eval/fa_ls_v05_1.before_retry_1430 --remote-stage-backup /tmp/r17_formal/fa_ls_v05_1.before_retry_1430 --reprepare
