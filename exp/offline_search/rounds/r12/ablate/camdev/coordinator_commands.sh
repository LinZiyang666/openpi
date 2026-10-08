#!/usr/bin/env bash
# Coordinator use only, after the selected fleet and ports are available.
set -euo pipefail
cd /home/weiland/projects/openpi
ACTION=${1:?usage: coordinator_commands.sh plan|sync|chain CELL [both|wrist_sub ...]}
CELL=${2:?cell required}
shift 2
case "$ACTION" in plan|sync|chain) ;; *) exit 2 ;; esac
case "$CELL" in pi05_l10_50|pi05_spatial_50|pi05_spatial_200|groot_spatial_50|groot_l10_50) ;; *) exit 2 ;; esac
ROOT=/home/weiland/projects/openpi/exp/offline_search/rounds/r12/ablate/camdev/r12_camdev_$CELL
if (( $# == 0 )); then set -- both wrist_sub; fi
ARMS=()
for METHOD in "$@"; do
    case "$METHOD" in both|wrist_sub) ;; *) exit 2 ;; esac
    ARMS+=("${ROOT##*/}_$METHOD")
done
export SERVER_HOST=${SERVER_HOST:-local} WORKER_HOST=${WORKER_HOST:-timan107}
export LOCAL_SERVER_CPUS=22-37,66-81 LOCAL_SERVER_COUNT=${LOCAL_SERVER_COUNT:-4}
case "$SERVER_HOST" in
    local) export PORTS=${PORTS:-23150,23151,23152,23153} ;;
    h100) export PORTS=${PORTS:-23250,23251,23252,23253} ;;
    *) exit 2 ;;
esac
case ",$PORTS," in
    *,23100,*|*,23101,*|*,23102,*|*,23103,*) echo 'Ports 23100-23103 are occupied by live runs.' >&2; exit 2 ;;
esac
export WPS=${WPS:-16} OMP=1 OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1
export PYTHONDONTWRITEBYTECODE=1 PYTHONPATH=.:src
unset OSCL_MANIFEST OSCL_TASKS OSCL_EPISODES OSCL_INIT_POOL OSCL_RECORD_LOOKS OSCL_EXECUTION_AUDIT
P=(taskset -c 22-37,66-81 env CUDA_VISIBLE_DEVICES= /home/weiland/projects/openpi/.venv/bin/python -B)
"${P[@]}" -m exp.offline_search.closed_loop.ops.h100.control "$ACTION" "$ROOT" "${ARMS[@]}"
