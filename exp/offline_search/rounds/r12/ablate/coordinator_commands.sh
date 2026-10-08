#!/usr/bin/env bash
# Coordinator use only. Emission/CPU tests never invoke plan, sync or chain here.
set -euo pipefail
cd /home/weiland/projects/openpi
ACTION=${1:?usage: coordinator_commands.sh plan|sync|chain|collect B|A CELL [METHOD ...]}
POOL=${2:?pool required}
CELL=${3:?cell required}
shift 3
case "$ACTION" in plan|sync|chain|collect) ;; *) exit 2 ;; esac
case "$CELL" in pi05_l10_50|pi05_spatial_50|pi05_spatial_200|groot_spatial_50|groot_l10_50) ;; *) exit 2 ;; esac
if [[ "$POOL" == B ]]; then
    ROOT=/home/weiland/projects/openpi/exp/offline_search/rounds/r12/ablate/pilots/r12_ablate_$CELL
    if (( $# == 0 )); then set -- both wrist_sub wrist_refit agent_sub agent_refit; fi
elif [[ "$POOL" == A ]]; then
    ROOT=/home/weiland/trace_runs/os_closed_loop/r12_ablate_test_$CELL
    if (( $# == 0 )); then
        echo 'A requires the explicit methods selected by the coordinator after the B screen.' >&2
        exit 2
    fi
else
    exit 2
fi
ARMS=()
for METHOD in "$@"; do
    case "$METHOD" in both|wrist_sub|wrist_refit|agent_sub|agent_refit) ;; *) exit 2 ;; esac
    ARMS+=("${ROOT##*/}_$METHOD")
done
export SERVER_HOST=${SERVER_HOST:-local} WORKER_HOST=${WORKER_HOST:-timan107}
export LOCAL_SERVER_CPUS=22-37,66-81 LOCAL_SERVER_COUNT=${LOCAL_SERVER_COUNT:-4}
export PORTS=${PORTS:-23150,23151,23152,23153} WPS=${WPS:-16}
export OMP=1 OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1
export PYTHONDONTWRITEBYTECODE=1 PYTHONPATH=.:src
unset OSCL_MANIFEST OSCL_TASKS OSCL_EPISODES OSCL_INIT_POOL OSCL_RECORD_LOOKS OSCL_EXECUTION_AUDIT
P=(taskset -c 22-37,66-81 env CUDA_VISIBLE_DEVICES= /home/weiland/projects/openpi/.venv/bin/python -B)
"${P[@]}" -m exp.offline_search.closed_loop.ops.h100.control "$ACTION" "$ROOT" "${ARMS[@]}"
