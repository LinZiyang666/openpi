#!/usr/bin/env bash
# Long segment OFF (longseg_off_1022): success-filtered library . GR00T . LIBERO . reduced K1 . levels ir45/40/32/25 . 12000 episodes
# H100 residents (/data/oscl_h100/openpi_r16_q3_server); 4 GR00T replicas; Q3 --os-judge stall_only; no --os-blind/--os-policy-tail. See longseg_off_1022/HANDBACK.md.
set -euo pipefail
cd /home/weiland/projects/openpi
env -u CUDA_VISIBLE_DEVICES -u JAX_PLATFORMS OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1 PYTHONPATH=.:src .venv/bin/python -m exp.offline_search.rounds.r17.filter_failed_knob.longseg_off_1022.runner run --group /data/mw_finetune/eval/fa_lo_gr_r_f_l_q/group.json --agent-island /scratch/zixuans8/openpi_r16_conductor_v1 --port 23158 --ports 23153,23154,23155,23156 --groot-suite-servers 3,1 --no-local-workers
