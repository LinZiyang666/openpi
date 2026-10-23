#!/usr/bin/env bash
set -euo pipefail
# jupyter-ziyang10-02 continuation of fa_lo_fw_r_f_m_j2: the pod container restarted 10-23 17:45 at 88,597/90,000
# (59 retry-exhausted); launcher SIGTERMed, resume.py --remaining --verify-closed -> 1,462 episodes, same serving
# (R1, B8, 15.5 GiB, text table). Pod setup.sh fastwam_nfs re-run and tunnels 24472/24473 -> 23474/23475 restarted.
cd /home/weiland/projects/openpi; export PYTHONPATH=.:src
export R17_H100_NODE=jupyter-ziyang10-02 R17_H100_HOST=149.165.153.233 R17_DRIVER_PORT=23192
exec .venv/bin/python -m exp.offline_search.rounds.r17.filter_failed.stage --run /data/mw_finetune/eval/fa_lo_fw_r_f_m_j2c1
