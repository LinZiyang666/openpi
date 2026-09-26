"""Profiling / diagnostic tools for the offline retrieval exploration (see README.md).

CPU only. The thread env is pinned before numpy is imported by any submodule so that the
multiprocessing workers (fork) are single-threaded; override by exporting the vars first.
"""
import os as _os

for _v in ("OMP_NUM_THREADS", "MKL_NUM_THREADS", "OPENBLAS_NUM_THREADS"):
    _os.environ.setdefault(_v, "1")
_os.environ.setdefault("CUDA_VISIBLE_DEVICES", "")
