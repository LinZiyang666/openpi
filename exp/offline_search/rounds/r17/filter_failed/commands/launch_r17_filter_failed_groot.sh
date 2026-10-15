#!/usr/bin/env bash
set -euo pipefail
cd /home/weiland/projects/openpi
env SERVER_HOST=h100 WORKER_HOST=timan108 PORTS=23230,23231,23232,23233 WPS=10 OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1 CUDA_VISIBLE_DEVICES= PYTHONPATH=.:src /home/weiland/projects/openpi/.venv/bin/python -m exp.offline_search.rounds.r17.filter_failed.legacy chain /home/weiland/trace_runs/os_closed_loop/r17_filter_failed_groot ff1015_groot_spatial_n50_c ff1015_groot_spatial_n100_c ff1015_groot_spatial_n200_c ff1015_groot_spatial_n300_c ff1015_groot_spatial_n400_c ff1015_groot_spatial_n500_c ff1015_groot_l10_n50_c ff1015_groot_l10_n100_c ff1015_groot_l10_n200_c ff1015_groot_l10_n300_c ff1015_groot_l10_n400_c ff1015_groot_l10_n500_c
