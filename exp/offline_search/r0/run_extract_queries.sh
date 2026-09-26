#!/usr/bin/env bash
# run_extract_queries.sh -- launcher for extract_queries.py (offline_search R0-A).
#
#   OUT=<queries root> TAG=<name> bash run_extract_queries.sh [extract_queries.py args...]
#
# OUT defaults to the full store /home/weiland/trace_runs/offline_search_store/queries, TAG to "extract".
# Runs in the foreground (wrap it in tmux / setsid nohup yourself). Writes
#   $OUT/_run/$TAG.log                     full log (stdout + stderr)
#   $OUT/_run/$TAG.DONE | $TAG.ERROR       exactly one of them at the end (exit code + time inside)
# Stale markers for the same TAG are removed at start. --out is passed through from $OUT; do not repeat it.
set -u
REPO=/home/weiland/projects/openpi
PY=$REPO/.venv/bin/python
OUT=${OUT:-/home/weiland/trace_runs/offline_search_store/queries}
TAG=${TAG:-extract}
RUN=$OUT/_run
mkdir -p "$RUN"
rm -f "$RUN/$TAG.DONE" "$RUN/$TAG.ERROR"

export OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 CUDA_VISIBLE_DEVICES=""
export PYTHONUNBUFFERED=1 TZ=America/Chicago

echo "== $(date '+%F %T %Z') start: $PY extract_queries.py --out $OUT $*" >>"$RUN/$TAG.log"
"$PY" "$REPO/exp/offline_search/r0/extract_queries.py" --out "$OUT" "$@" >>"$RUN/$TAG.log" 2>&1
rc=$?
echo "== $(date '+%F %T %Z') exit $rc" >>"$RUN/$TAG.log"

if [ "$rc" -eq 0 ]; then
  echo "rc=0 $(date '+%F %T %Z') log=$RUN/$TAG.log" >"$RUN/$TAG.DONE"
else
  { echo "rc=$rc $(date '+%F %T %Z') log=$RUN/$TAG.log"; tail -n 40 "$RUN/$TAG.log"; } >"$RUN/$TAG.ERROR"
fi
exit "$rc"
