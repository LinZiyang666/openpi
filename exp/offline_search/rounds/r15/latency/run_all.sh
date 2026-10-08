#!/bin/bash
# Sequential, one pinned core (34), single-thread BLAS; outputs bench_<cell>_<size>.json next to this script.
cd /home/weiland/projects/openpi
for cell in pi05_l10 pi05_spatial; do for size in 50 100 200 300 400 500; do
  taskset -c 34 env CUDA_VISIBLE_DEVICES= OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1 PYTHONPATH=.:src PYTHONDONTWRITEBYTECODE=1 \
    .venv/bin/python -m exp.offline_search.rounds.r15.latency.bench --cell $cell --size $size --n 2048
done; done
echo ALL_DONE
