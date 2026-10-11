#!/usr/bin/env bash
# Reviewable normal-launcher commands. Print by default; execute one lane only
# with --run. Start the two lanes in separate terminals after RAM admission.
set -euo pipefail
cd /home/weiland/projects/openpi
lane=${1:-}
mode=${2:---show}
if [[ "$lane" != h100 && "$lane" != local ]] || [[ "$mode" != --show && "$mode" != --run ]]; then
    echo 'Usage: bash formal_commands.sh h100|local [--show|--run]' >&2
    exit 2
fi
base=(bash exp/r17_integration/ops/formal_eval.sh --model fastwam_libero
      --checkpoint /data/mw_finetune/models/fastwam_libero --worker-node timan107
      --replicas 1 --workers 16 --max-batch 8 --batch-wait-ms 5
      --worker-max-episodes 4 --worker-memory-mib 6144 --worker-vram-mib 1024
      --memory-budget-gib 36 --official-50 --seconds 43200 --episode-seconds 1800
      --retries 2 --run)
run_command() {
    if [[ "$mode" == --run ]]; then
        "$@"
    else
        printf '%q ' "$@"
        printf '\n'
    fi
}
if [[ "$lane" == h100 ]]; then
    node=(--server-node h100 --port 23170 --driver-port 23179
          --worker-egl-device 1 --worker-egl-devices 1,2,3)
    control=R17_FW_B8W5_H100_official50_control
    run_command "${base[@]}" "${node[@]}" --run-name "$control"
    run_command "${base[@]}" "${node[@]}" --run-name R17_FW_B8W5_H100_C16_K10 \
        --suites libero_spatial libero_10 --commit-horizon 16 \
        --paired-control "/data/mw_finetune/eval/$control"
else
    node=(--server-node local --port 23160 --driver-port 23169
          --worker-egl-device 4 --worker-egl-devices 4,5,6)
    lock=(flock -w 600 /tmp/r17_gpu.lock env
          CUDA_VISIBLE_DEVICES=GPU-98d36ed2-a8e9-fb07-701c-41f337a6c1f9)
    control=R17_FW_B8W5_4090_spatial10_control
    run_command "${lock[@]}" "${base[@]}" "${node[@]}" --run-name "$control" \
        --suites libero_spatial libero_10
    for horizon in 10 20 32; do
        run_command "${lock[@]}" "${base[@]}" "${node[@]}" \
            --run-name "R17_FW_B8W5_4090_C${horizon}_K10" \
            --suites libero_spatial libero_10 --commit-horizon "$horizon" \
            --paired-control "/data/mw_finetune/eval/$control"
    done
    for steps in 2 1; do
        run_command "${lock[@]}" "${base[@]}" "${node[@]}" \
            --run-name "R17_FW_B8W5_4090_K${steps}_C5" \
            --suites libero_spatial libero_10 --denoising-steps "$steps" \
            --paired-control "/data/mw_finetune/eval/$control"
    done
fi
