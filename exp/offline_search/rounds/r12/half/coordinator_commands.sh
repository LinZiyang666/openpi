#!/usr/bin/env bash
# Coordinator only, after final binding and B screening; never used by CPU work.
set -euo pipefail
cd /home/weiland/projects/openpi
ACTION=${1:?usage: coordinator_commands.sh plan|sync|chain|collect B|A CELL [half|off ...]}
POOL=${2:?pool required}
CELL=${3:?cell required}
shift 3
case "$ACTION" in plan|sync|chain|collect) ;; *) exit 2 ;; esac
case "$CELL" in pi05_l10_50|pi05_spatial_50|pi05_spatial_200|groot_spatial_50|groot_l10_50|pi05_l10_50_periodic_ir32) ;; *) exit 2 ;; esac
BASE=/home/weiland/projects/openpi/exp/offline_search/rounds/r12/half
if [[ "$POOL" == B ]]; then
    ROOT=$BASE/pilots/r12_half_$CELL
elif [[ "$POOL" == A ]]; then
    ROOT=/home/weiland/trace_runs/os_closed_loop/r12_test_half_$CELL
else
    exit 2
fi
export PYTHONDONTWRITEBYTECODE=1 PYTHONPATH=.:src
export OMP=1 OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1
P=(taskset -c 22-37,66-81 env CUDA_VISIBLE_DEVICES= /home/weiland/projects/openpi/.venv/bin/python -B)
STATUS=$("${P[@]}" - "$BASE" "$POOL" "$CELL" "$ROOT" <<'PY'
import json,sys
from pathlib import Path
from exp.offline_search.closed_loop.half_everywhere_config import load_table
base,pool,cell,root=Path(sys.argv[1]),sys.argv[2],sys.argv[3],Path(sys.argv[4])
records=json.loads((base/('pilot_roots.json' if pool=='B' else 'test_a_roots.json')).read_text())
match=[r for r in records if r['suffix']==cell]
if not match:
    skipped=json.loads((base/'skipped.json').read_text())
    if cell.removesuffix('_periodic_ir32') in {r['cell'] for r in skipped}:
        print('SKIPPED_NO_WRIST');sys.exit(0)
    raise ValueError('cell was not emitted')
if match[0]['provisional']:
    raise ValueError('provisional roots cannot launch')
record=json.loads((root/'ROOT.json').read_text())
load_table(record['camera_table'],cell=record['cell'])
if record['provisional']:
    raise ValueError('provisional definitions cannot launch')
print('READY')
PY
)
if [[ "$STATUS" == SKIPPED_NO_WRIST ]]; then
    echo "$CELL: skipped (final table has no wrist tasks)."
    exit 0
fi
if (( $# == 0 )); then
    set -- half
    if [[ "$POOL" == B && "$CELL" == groot_* ]]; then set -- half off; fi
fi
ARMS=()
for METHOD in "$@"; do
    case "$METHOD" in half|off) ;; *) exit 2 ;; esac
    ARMS+=("${ROOT##*/}_$METHOD")
done
if [[ "$CELL" == groot_* ]]; then
    export SERVER_HOST=${SERVER_HOST:-h100} WORKER_HOST=${WORKER_HOST:-timan108}
else
    export SERVER_HOST=${SERVER_HOST:-local} WORKER_HOST=${WORKER_HOST:-timan107}
fi
export LOCAL_SERVER_CPUS=22-37,66-81 LOCAL_SERVER_COUNT=${LOCAL_SERVER_COUNT:-4}
case "$SERVER_HOST" in
    local) export PORTS=${PORTS:-23170,23171,23172,23173} ;;
    h100) export PORTS=${PORTS:-23370,23371,23372,23373} ;;
    *) exit 2 ;;
esac
case ",$PORTS," in *,23100,*|*,23101,*|*,23102,*|*,23103,*) echo 'Reserved live-run ports refused.' >&2; exit 2 ;; esac
export WPS=${WPS:-16}
unset OSCL_MANIFEST OSCL_TASKS OSCL_EPISODES OSCL_INIT_POOL OSCL_RECORD_LOOKS OSCL_EXECUTION_AUDIT
"${P[@]}" -m exp.offline_search.closed_loop.ops.h100.control "$ACTION" "$ROOT" "${ARMS[@]}"
