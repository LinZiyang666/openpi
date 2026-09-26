#!/usr/bin/env bash
# run_build_library_store.sh -- launcher for build_library_store.py (offline_search R0-B).
#
#   OUT=<library root> TAG=<name> bash run_build_library_store.sh [build_library_store.py args...]
#
# OUT defaults to the full store /home/weiland/trace_runs/offline_search_store/library, TAG to "build".
# Runs in the foreground (wrap it in tmux / setsid nohup yourself). Writes
#   $OUT/_run/$TAG.log                     full log (stdout + stderr)
#   $OUT/_run/$TAG.DONE | $TAG.ERROR       exactly one of them at the end (exit code + time inside)
#   $OUT/READY                             (by the python script) once all 8 libraries are complete
# Stale markers for the same TAG are removed at start. --out is passed through from $OUT; do not repeat it.
# CPU only; /archive (GR00T build h5) is read by a single sequential reader inside the script.
set -u
REPO=/home/weiland/projects/openpi
PY=$REPO/.venv/bin/python
OUT=${OUT:-/home/weiland/trace_runs/offline_search_store/library}
TAG=${TAG:-build}
RUN=$OUT/_run
mkdir -p "$RUN"
for m in "$RUN/$TAG.DONE" "$RUN/$TAG.ERROR"; do
  if [ -e "$m" ]; then rm "$m"; fi
done

export OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 NUMEXPR_NUM_THREADS=1 CUDA_VISIBLE_DEVICES=""
export HDF5_USE_FILE_LOCKING=FALSE PYTHONUNBUFFERED=1 PYTHONPATH=$REPO/src TZ=America/Chicago

# no `exit` inside this brace group: a brace group runs in the current shell, so an exit there would end
# the script before the marker below is written (bug in the first version of this launcher).
{
  echo "== $(date '+%F %T %Z') start: $PY build_library_store.py --out $OUT $*"
  "$PY" "$REPO/exp/offline_search/r0/build_library_store.py" --out "$OUT" "$@"
  rc=$?
  echo "== $(date '+%F %T %Z') exit $rc"
} >>"$RUN/$TAG.log" 2>&1

if [ "$rc" -eq 0 ]; then
  echo "rc=0 $(date '+%F %T %Z') log=$RUN/$TAG.log" >"$RUN/$TAG.DONE"
else
  { echo "rc=$rc $(date '+%F %T %Z') log=$RUN/$TAG.log"; tail -n 40 "$RUN/$TAG.log"; } >"$RUN/$TAG.ERROR"
fi
exit "$rc"
