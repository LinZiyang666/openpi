#!/usr/bin/env bash
# Long segment OFF (longseg_off_1022): success-filtered library . pi0.5 . LIBERO . reduced K1 . levels ir45/40/32/25 . 8000 episodes
# H100 residents (/data/oscl_h100/openpi_r16_q3_server); 3 pi0.5 replicas; Q3 --os-judge stall_only; no --os-blind/--os-policy-tail. See longseg_off_1022/HANDBACK.md.
set -euo pipefail
cd /home/weiland/projects/openpi
env -u CUDA_VISIBLE_DEVICES -u JAX_PLATFORMS OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1 PYTHONPATH=.:src .venv/bin/python -m exp.offline_search.rounds.r17.filter_failed_knob.longseg_off_1022.runner run --group /data/mw_finetune/eval/fa_lo_pi_r_f_l_q/group.json --agent-island /scratch/zixuans8/openpi_r16_conductor_v1 --port 23157 --ports 23150,23151,23152 --no-local-workers
