#!/usr/bin/env bash
# Eight-group cache study (#135): original library · pi05 · LIBERO · reduced K1 · groups c/cr/ch/chr.
# R16 native conductor launch (dynamic fleet from LEGACY_AUTOSCALE.json); resumes from the group's own journal.
set -euo pipefail
export VLA_WORKER_HOST=timan107 OPENPI_AUTOSCALE_NODE_CONFIG='{"timan107":{"floor":25769803776,"critical":17179869184,"hysteresis":8589934592,"spawn_interval_s":2,"external_peak_floor":4294967296,"external_peak_floor_metaworld":1073741824},"timan1":{"external_peak_floor":4294967296,"external_peak_floor_metaworld":1073741824}}'
cd /home/weiland/projects/openpi_r16_k1ref_server
env -u CUDA_VISIBLE_DEVICES -u JAX_PLATFORMS OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1 PYTHONPATH=.:src .venv/bin/python -c 'import sys;import exp.offline_search.rounds.r17 as r;r.__path__.append("/home/weiland/projects/openpi/exp/offline_search/rounds/r17");from exp.offline_search.rounds.r17.filter_failed_knob.cache_study_1019.study import serve;serve()' run --group /data/mw_finetune/eval/fa_cs_pi_r/group.json --agent-island /scratch/zixuans8/openpi_r16_conductor_v1 --workers 64 --port 23159 --ports 23150,23151,23152
