#!/usr/bin/env bash
# Eight-group cache study (#135): original library · pi05 · LIBERO · reduced K1 · groups c/cr/ch/chr (R16 conductor).
# Stages the dynamic-worker release (LEGACY_AUTOSCALE) for timan107; CPU only, starts nothing. See cache_study_1019/HANDBACK.md.
set -euo pipefail
cd /home/weiland/projects/openpi
env CUDA_VISIBLE_DEVICES= JAX_PLATFORMS=cpu PYTHONDONTWRITEBYTECODE=1 OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1 PYTHONPATH=.:src OPENPI_AUTOSCALE_NODE_CONFIG='{"timan107":{"floor":25769803776,"critical":17179869184,"hysteresis":8589934592,"spawn_interval_s":2,"external_peak_floor":4294967296,"external_peak_floor_metaworld":1073741824},"timan1":{"external_peak_floor":4294967296,"external_peak_floor_metaworld":1073741824}}' .venv/bin/python -m exp.offline_search.rounds.r17.filter_failed_knob.cache_study_1019.study prepare --name fa_cs_pi_r
