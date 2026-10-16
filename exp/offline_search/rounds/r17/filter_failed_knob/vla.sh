#!/usr/bin/env bash
set -euo pipefail
FFK_REPO=/home/weiland/projects/openpi
cd /home/weiland/projects/openpi_r16_k1ref_server
exec env CUDA_VISIBLE_DEVICES= PYTHONDONTWRITEBYTECODE=1 OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1 PYTHONPATH=.:src .venv/bin/python -c '
import runpy, sys
import exp.offline_search.rounds.r17 as r17
r17.__path__.append("/home/weiland/projects/openpi/exp/offline_search/rounds/r17")
module="vla_runner" if len(sys.argv)>1 and sys.argv[1]=="run" else "vla"
if len(sys.argv)>1 and sys.argv[1]=="cpu-future":
    module="cpu_future"
    sys.argv=[sys.argv[0],"--vla",*sys.argv[2:]]
runpy.run_module("exp.offline_search.rounds.r17.filter_failed_knob."+module,run_name="__main__")
' "$@"
