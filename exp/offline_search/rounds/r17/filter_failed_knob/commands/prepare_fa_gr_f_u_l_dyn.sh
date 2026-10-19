#!/usr/bin/env bash
# R16 with dynamic workers (legacy autoscale release, owner 10-18): stage the release, then the native launch reads LEGACY_AUTOSCALE.json.
set -euo pipefail
cd /home/weiland/projects/openpi
bash /home/weiland/projects/openpi/exp/offline_search/rounds/r17/filter_failed_knob/commands/prepare_fa_gr_f_u_l.sh
env CUDA_VISIBLE_DEVICES= JAX_PLATFORMS=cpu PYTHONDONTWRITEBYTECODE=1 OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1 PYTHONPATH=.:src OPENPI_AUTOSCALE_NODE_CONFIG='{"timan107":{"floor":25769803776,"critical":17179869184,"hysteresis":8589934592,"spawn_interval_s":2,"external_peak_floor":4294967296,"external_peak_floor_metaworld":1073741824},"timan1":{"external_peak_floor":4294967296,"external_peak_floor_metaworld":1073741824}}' .venv/bin/python -m exp.offline_search.rounds.r17.filter_failed_knob.vla_workers prepare --job fa_gr_f_u_l --worker-host timan107
