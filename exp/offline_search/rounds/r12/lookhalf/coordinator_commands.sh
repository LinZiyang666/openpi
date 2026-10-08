#!/usr/bin/env bash
# Coordinator entry point. This script was emitted, never launched by sol.
set -euo pipefail
cd /home/weiland/projects/openpi
ACTION=${1:?usage: coordinator_commands.sh plan|sync|chain|collect|report}
case "$ACTION" in plan|sync|chain|collect|report) ;; *) exit 2 ;; esac
export SERVER_HOST=local
export WORKER_HOST=${WORKER_HOST:-timan107}
export LOCAL_SERVER_CPUS=22-37,66-81
export LOCAL_SERVER_COUNT=${LOCAL_SERVER_COUNT:-4}
if [[ -z ${PORTS:-} ]]; then
    PORTS=$(taskset -c 22-37,66-81 env CUDA_VISIBLE_DEVICES= .venv/bin/python -B -c \
        'import os; print(",".join(str(23110+i) for i in range(int(os.environ["LOCAL_SERVER_COUNT"]))))')
fi
export PORTS
export WPS=${WPS:-16}
export OMP=1 OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1
export PYTHONDONTWRITEBYTECODE=1 PYTHONPATH=.:src
unset OSCL_MANIFEST OSCL_TASKS OSCL_EPISODES OSCL_INIT_POOL
P=(taskset -c 22-37,66-81 env CUDA_VISIBLE_DEVICES= /home/weiland/projects/openpi/.venv/bin/python -B)
"${P[@]}" - "$ACTION" <<'PY'
import os
from pathlib import Path
import sys
ports = [int(p) for p in os.environ['PORTS'].split(',')]
count = int(os.environ['LOCAL_SERVER_COUNT'])
if count < 1 or len(ports) != count or len(set(ports)) != count:
    raise SystemExit('PORTS must contain exactly LOCAL_SERVER_COUNT distinct ports')
if any(not 23104 <= p <= 23199 for p in ports):
    raise SystemExit('ports 23100-23103 are reserved for the existing look-half pilot')
if sys.argv[1] in ('sync', 'chain'):
    listeners = set()
    for path in ('/proc/net/tcp', '/proc/net/tcp6'):
        for line in Path(path).read_text().splitlines()[1:]:
            cols = line.split()
            if cols[3] == '0A':
                listeners.add(int(cols[1].rsplit(':', 1)[1], 16))
    if listeners.intersection(range(23100, 23104)):
        raise SystemExit('existing look-half pilot is still listening; guard sync/chain must wait for coordinator release')
PY
BASE=/home/weiland/projects/openpi/exp/offline_search/rounds/r12/lookhalf/pilots
run_root() {
    local root=$1
    shift
    if [[ "$ACTION" == report ]]; then
        "${P[@]}" -m exp.offline_search.rounds.r12.lookhalf.report "$root"
    else
        "${P[@]}" -m exp.offline_search.closed_loop.ops.h100.control "$ACTION" "$root" "$@"
    fi
}
run_root "$BASE/r12_lookhalf_fix1_pi05_l10_50" \
    r12_lh_pi05_l10_50_off r12_lh_pi05_l10_50_opus_m1 \
    r12_lh_pi05_l10_50_astra_consensus2 r12_lh_pi05_l10_50_astra_strict
run_root "$BASE/r12_lookhalf_fix1_pi05_spatial_50" \
    r12_lh_pi05_spatial_50_off r12_lh_pi05_spatial_50_opus_m1 \
    r12_lh_pi05_spatial_50_astra_consensus2 r12_lh_pi05_spatial_50_astra_strict
run_root "$BASE/r12_lookhalf_fix1_pi05_spatial_200" \
    r12_lh_pi05_spatial_200_off r12_lh_pi05_spatial_200_opus_m1 \
    r12_lh_pi05_spatial_200_astra_consensus2
run_root "$BASE/r12_lookhalf_fix1_groot_spatial_50" \
    r12_lh_groot_spatial_50_off r12_lh_groot_spatial_50_opus_m1 \
    r12_lh_groot_spatial_50_astra_consensus2
run_root "$BASE/r12_lookhalf_fix1_groot_l10_50" \
    r12_lh_groot_l10_50_off r12_lh_groot_l10_50_opus_m1 \
    r12_lh_groot_l10_50_astra_consensus2
