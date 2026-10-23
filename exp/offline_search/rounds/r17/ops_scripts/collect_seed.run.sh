#!/bin/bash
# ROOT MODEL SERVER SEED LIBRARY_ROOT RUN [--collect-only] [library_campaign args...]
# MT50: full native inference, C5, no images; finish may run later on any CPU host.
set -euo pipefail
if [ "$#" -lt 6 ]; then
  echo "usage: $0 ROOT MODEL SERVER SEED LIBRARY_ROOT RUN [--collect-only] [--smoke] [args...]" >&2
  exit 2
fi
ROOT=$1; MODEL=$2; SERVER=$3; SEED=$4; LIBRARY_ROOT=$5; RUN=$6; shift 6
case "$MODEL" in pi05_mw|groot_mw|dp_mw|fastwam_mw) ;; *) echo "unknown MT50 model: $MODEL" >&2; exit 2;; esac
case "$SEED" in 9|10) ;; *) echo 'new MT50 libraries require seed 9 or 10' >&2; exit 2;; esac
case "$RUN" in ""|*[!a-zA-Z0-9_-]*) echo 'plain unique run name required' >&2; exit 2;; esac
M=$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)
K=$M/../knob
ARGS=(); COLLECT_ONLY=0; SMOKE=0
for arg in "$@"; do
  case "$arg" in
    --collect-only) COLLECT_ONLY=1;;
    --smoke) SMOKE=1; ARGS+=("$arg");;
    *) ARGS+=("$arg");;
  esac
done
cd -- "$ROOT"
ROOT=$PWD
case "$LIBRARY_ROOT" in /*) ;; *) LIBRARY_ROOT=$PWD/$LIBRARY_ROOT;; esac
DIR=$LIBRARY_ROOT/${MODEL}_mt50/campaigns/$RUN
LOGS=$LIBRARY_ROOT/${MODEL}_mt50/logs/$RUN
mkdir -p -- "$LOGS"
log(){ echo "$(date +%H:%M:%S) [collect seed$SEED] $*" | tee -a "$LOGS/collect.log"; }
SECONDS_LIMIT=172800; EPISODE_SECONDS=3600
if [ "$SMOKE" = 1 ]; then SECONDS_LIMIT=900; EPISODE_SECONDS=600; fi
PY=("$ROOT/.venv/bin/python" -u -m exp.r17_integration.ops.library_campaign)
log "$RUN prepare $MODEL on $SERVER"
env CUDA_VISIBLE_DEVICES='' JAX_PLATFORMS=cpu "${PY[@]}" \
  --server-node "$SERVER" --gpu cuda:0 --seconds "$SECONDS_LIMIT" --episode-seconds "$EPISODE_SECONDS" \
  "${ARGS[@]}" --model "$MODEL" --suite mt50 --library-seed "$SEED" --library-root "$LIBRARY_ROOT" \
  --no-capture-images --subset-counts 5,10,20,30,40,50 --driver-mode legacy \
  --run-name "$RUN" --prepare-only > "$LOGS/prepare.log" 2>&1
python3 "$K/check_direct.py" "$DIR"
# /data/r17_library is on NVMe; keep driver/ inside the transferable campaign.
if [ "$SERVER" = local ]; then bash "$M/pin_local.sh" "$RUN" & fi
bash "$M/launch_direct_root.sh" "$ROOT" "$DIR" "$LOGS/run.log"
"$ROOT/.venv/bin/python" - "$DIR" <<'PY'
import json
from pathlib import Path
import sys
receipt = json.loads((Path(sys.argv[1])/'RECEIPT.json').read_text())
if (receipt.get('status') != 'PASS' or receipt.get('all_owned_closed') is not True or
        receipt.get('worker_storage_cleanup_pending')):
    raise SystemExit('collection is not closed/PASS; inspect RECEIPT.json before finish')
PY
log "$RUN collection closed: $DIR"
if [ "$COLLECT_ONLY" = 1 ]; then
  log "finish on the allocated CPU host: library_campaign --model $MODEL --suite mt50 --finish $DIR"
  exit 0
fi
env CUDA_VISIBLE_DEVICES='' JAX_PLATFORMS=cpu OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1 \
  "${PY[@]}" --model "$MODEL" --suite mt50 --finish "$DIR" > "$LOGS/finish.log" 2>&1
log "$RUN finish PASS"
