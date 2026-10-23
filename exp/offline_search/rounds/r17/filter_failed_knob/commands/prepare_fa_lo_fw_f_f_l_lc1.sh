#!/usr/bin/env bash
set -euo pipefail
# H100 continuation of fa_lo_fw_f_f_l_l (long segment OFF · Fast-WAM · LIBERO · full steps · levels). The source ran
# R2 without the text table: both replicas pinned at their 4 CPU threads, GPU ~50-60%, ~38 eps/min. Stopped 10-23
# PARTIAL; resume.py --remaining --verify-closed; text table (bitwise equal on the H100), R3, 60 GiB, server_threads 3
# (H100 also hosts ~40 LIBERO render workers, so R4 would not leave their VRAM headroom).
cd /home/weiland/projects/openpi; export PYTHONPATH=.:src
unset R17_H100_NODE R17_H100_HOST R17_DRIVER_PORT R17_PUBLIC_PORT
exec .venv/bin/python -m exp.offline_search.rounds.r17.filter_failed.stage --run /data/mw_finetune/eval/fa_lo_fw_f_f_l_lc1
