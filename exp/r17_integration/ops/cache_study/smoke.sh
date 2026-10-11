#!/usr/bin/env bash
set -euo pipefail
cd /home/weiland/projects/openpi
cache_model="${1:?choose dp, fastwam or cosmos}"
cache_mode="${2:---print}"
if (( $# >= 2 )); then shift 2; else shift; fi
cache_args=(smoke --study exp/r17_integration/ops/cache_study/yaml/STUDY.json
  --worker-node timan107 --replicas 2 --batch-wait-ms 5
  --worker-max-episodes 4 --worker-restart-limit 2 --retries 2 --seconds 7200)
case "$cache_model" in
  dp) cache_args+=(--model dp_libero --server-node local --workers 32 --max-batch 8 --mps
      --episode-seconds 1800
      --replica-startup-width 1 --driver-mode group-commit --worker-memory-mib 4096 --worker-vram-mib 576
      --cache-control /data/mw_finetune/eval/hs5dp_ctl_142812 --cache-control-runtime /data/mw_finetune/eval/hs5dp_ctl_142812
      --memory-budget-gib 40 --port 23400 --driver-port 23419);;
  fastwam) cache_args+=(--model fastwam_libero --server-node h100 --workers 48 --max-batch 16 --mps
      --episode-seconds 3600
      --replica-startup-width 2 --driver-mode group-commit --worker-memory-mib 4096 --worker-vram-mib 768
      --cache-control /data/mw_finetune/eval/hs6fw_res_152931 --cache-control-runtime /data/mw_finetune/eval/hs6fw_ctl_143332
      --worker-egl-devices 0,1,2,3,4,5,6,7 --memory-budget-gib 70 --port 23420 --driver-port 23439);;
  cosmos) cache_args+=(--model cosmos_libero --server-node h100 --workers 8 --max-batch 1
      --episode-seconds 1800
      --replica-startup-width 1 --driver-mode legacy --worker-memory-mib 4096 --worker-vram-mib 768
      --cache-control /data/mw_finetune/eval/formal_cosmos_libero_official2000_20261010_130639_a8ca2043
      --cache-control-runtime /data/mw_finetune/eval/formal_cosmos_libero_official2000_20261010_130639_a8ca2043
      --memory-budget-gib 20 --port 23440 --driver-port 23459);;
  *) printf 'unknown model: %s\n' "$cache_model" >&2; exit 2;;
esac
case "$cache_mode" in
  --print)
    if [[ "$cache_model" == dp ]]; then printf '%q ' flock -n /tmp/r17_gpu.lock; fi
    printf '%q ' /home/weiland/projects/openpi/.venv/bin/python -u -m exp.r17_integration.ops.cache_study "${cache_args[@]}" "$@" --run; printf '\n'; exit 0;;
  --prepare-only|--run) cache_args+=("$cache_mode");;
  *) printf 'choose --print, --prepare-only or --run\n' >&2; exit 2;;
esac
if [[ "$cache_model" == dp && "$cache_mode" == --run ]]; then
  exec flock -n /tmp/r17_gpu.lock /home/weiland/projects/openpi/.venv/bin/python -u -m exp.r17_integration.ops.cache_study "${cache_args[@]}" "$@"
fi
exec /home/weiland/projects/openpi/.venv/bin/python -u -m exp.r17_integration.ops.cache_study "${cache_args[@]}" "$@"
