#!/usr/bin/env bash
# COORDINATOR ONLY. Copies only study-owned inputs/tools/cache dependencies.
# No checkpoints, envs, running campaigns, ports or process changes.
set -euo pipefail
cat /proc/pressure/io
if [[ -n "$(ss -Htn state established '( sport = :23177 )')" ]]; then
  echo 'An authorized rsync transfer is active; wait for it to finish before H100 staging.' >&2
  exit 2
fi
tether exec --timeout 30s h100 -- df -h -l
tether exec --timeout 30s h100 -- mkdir -p /home/exouser/xgpu_1017/tools /home/exouser/xgpu_1017/concurrent/bundles /home/exouser/xgpu_1017/rootfs/data/mw_finetune /home/exouser/xgpu_1017/inputs_dp3 /home/exouser/xgpu_1017/inputs_mt50_real3 /home/exouser/xgpu_1017/inputs_cosmos3
tether exec --timeout 5m h100 -- rsync -a --bwlimit=16384 rsync://ziyanglin.com:23177/xr_tools/ /home/exouser/xgpu_1017/tools/
tether exec --timeout 5m h100 -- rsync -a --bwlimit=16384 rsync://ziyanglin.com:23177/xc_bundles/ /home/exouser/xgpu_1017/concurrent/bundles/
tether exec --timeout 5m h100 -- rsync -a --bwlimit=16384 rsync://ziyanglin.com:23177/xc_references/ /home/exouser/xgpu_1017/concurrent/references/
tether exec --timeout 5m h100 -- rsync -a --bwlimit=16384 rsync://ziyanglin.com:23177/xr_inputs/ /home/exouser/xgpu_1017/inputs_dp3/
tether exec --timeout 5m h100 -- rsync -a --bwlimit=16384 rsync://ziyanglin.com:23177/xc_mt50_inputs/ /home/exouser/xgpu_1017/inputs_mt50_real3/
tether exec --timeout 5m h100 -- rsync -a --bwlimit=16384 rsync://ziyanglin.com:23177/xc_cosmos_inputs/ /home/exouser/xgpu_1017/inputs_cosmos3/
cat /proc/pressure/io
tether exec --timeout 5m h100 -- rsync -a --bwlimit=16384 rsync://ziyanglin.com:23177/xc_all_cache/ /home/exouser/xgpu_1017/rootfs/data/mw_finetune/
