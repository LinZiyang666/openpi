#!/usr/bin/env bash
# R16 native launch through vla_workers.run_r16_selected (dynamic fleet from LEGACY_AUTOSCALE.json); resumes from the group's own state.
set -euo pipefail
export VLA_WORKER_HOST=timan107 OPENPI_AUTOSCALE_NODE_CONFIG='{"timan107":{"floor":25769803776,"critical":17179869184,"hysteresis":8589934592,"spawn_interval_s":2,"external_peak_floor":4294967296,"external_peak_floor_metaworld":1073741824},"timan1":{"external_peak_floor":4294967296,"external_peak_floor_metaworld":1073741824}}'
exec bash /home/weiland/projects/openpi/exp/offline_search/rounds/r17/filter_failed_knob/commands/launch_fa_gr_r_f_l.sh
