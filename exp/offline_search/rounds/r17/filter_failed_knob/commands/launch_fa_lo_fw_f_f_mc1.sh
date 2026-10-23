#!/usr/bin/env bash
set -euo pipefail
# sean lane 1 continuation of fa_lo_fw_f_f_m (long segment OFF · Fast-WAM · MetaWorld · full steps · knob off).
# The source ran R2 without the text table: both replicas pinned at their 4 CPU threads, GPU ~43%, ~95 eps/min
# (90,000 -> ~15 h). Stopped 10-23 PARTIAL; resume.py --remaining --verify-closed; text table (bitwise equal on
# the H100), R4, 70 GiB, server_threads 3 (sean has 20 cores and no local workers).
# Chain env: R17_H100_NODE=h100-sean R17_H100_HOST=149.165.169.137 R17_DRIVER_PORT=23197
cd /home/weiland/projects/openpi; export PYTHONPATH=.:src
exec .venv/bin/python -m exp.offline_search.rounds.r17.filter_failed_knob.launch --run /data/mw_finetune/eval/fa_lo_fw_f_f_mc1
