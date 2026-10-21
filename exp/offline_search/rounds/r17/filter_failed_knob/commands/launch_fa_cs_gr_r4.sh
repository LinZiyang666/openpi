#!/usr/bin/env bash
# Eight-group cache study (#135): original library · groot · LIBERO · reduced K1 · groups c/cr/ch/chr.
# 10-21 01:4x: libero_spatial is complete (8,000/8,000); its two residents idled. Four libero_10 residents (23150/23151 kept, 23154/23155 new).
# R16 conductor from this checkout (resident servers: the frozen R16 server checkout); workers on timan107 and
# the 4090 from the central scheduler. Resumes from the group's own journal.
set -euo pipefail
cd /home/weiland/projects/openpi
env -u CUDA_VISIBLE_DEVICES -u JAX_PLATFORMS OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1 PYTHONPATH=.:src .venv/bin/python -m exp.offline_search.rounds.r16.conductor_runner run --group /data/mw_finetune/eval/fa_cs_gr_r/group.json --agent-island /scratch/zixuans8/openpi_r16_conductor_v1 --port 23159 --ports 23150,23151,23154,23155 --groot-suite-servers 4,0
