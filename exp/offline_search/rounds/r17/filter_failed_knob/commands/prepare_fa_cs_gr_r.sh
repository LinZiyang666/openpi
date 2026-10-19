#!/usr/bin/env bash
# Eight-group cache study (#135): original library · groot · LIBERO · reduced K1 · groups c/cr/ch/chr (R16 conductor).
# CPU only, starts nothing: refuses a DONE group. The launch installs the scheduler package. See cache_study_1019/HANDBACK.md.
set -euo pipefail
cd /home/weiland/projects/openpi
env CUDA_VISIBLE_DEVICES= JAX_PLATFORMS=cpu PYTHONDONTWRITEBYTECODE=1 OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1 PYTHONPATH=.:src .venv/bin/python -m exp.offline_search.rounds.r17.filter_failed_knob.cache_study_1019.study prepare --name fa_cs_gr_r
