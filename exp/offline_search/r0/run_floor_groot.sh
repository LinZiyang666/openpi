#!/bin/bash
# run_floor_groot.sh -- launch floor_resample_groot.py (teacher noise floor, GPU) with env, a free-memory
# check, a log file and DONE / ERROR markers.
# usage: run_floor_groot.sh [--suite spatial|l10|all] [--s 500] [--k 4] [--seed 20260926] [--out DIR]
# env:   GPU (index, default 0), MIN_FREE_MIB (default 20480)
# files: $OUT/_run/floor_groot_<suite>.{log,DONE,ERROR}; data in $OUT/groot_<suite>/resample{.npz,_summary.json}
# A rerun with the same --s/--k/--seed resumes from $OUT/groot_<suite>/resample.partial.npz.
set -u
REPO=/home/weiland/projects/openpi
HERE=$REPO/exp/offline_search/r0
SUITE=all; S=500; K=4; SEED=20260926; OUT=/home/weiland/trace_runs/offline_search_store/floor
while [ $# -gt 0 ]; do
  case "$1" in
    --suite) SUITE=$2; shift 2;; --s) S=$2; shift 2;; --k) K=$2; shift 2;;
    --seed) SEED=$2; shift 2;; --out) OUT=$2; shift 2;;
    *) echo "unknown arg $1" >&2; exit 2;;
  esac
done
RUN=$OUT/_run; mkdir -p "$RUN"
TAG=floor_groot_$SUITE
LOG=$RUN/$TAG.log; DONE=$RUN/$TAG.DONE; ERR=$RUN/$TAG.ERROR
for f in "$DONE" "$ERR"; do if [ -e "$f" ]; then rm "$f"; fi; done
GPU=${GPU:-0}; MIN_FREE_MIB=${MIN_FREE_MIB:-20480}
FREE=$(nvidia-smi -i "$GPU" --query-gpu=memory.free --format=csv,noheader,nounits 2>/dev/null | head -1 | tr -d ' ')
if [ -z "$FREE" ] || [ "$FREE" -lt "$MIN_FREE_MIB" ]; then
  echo "$(date '+%F %T %Z') GPU $GPU free ${FREE:-?} MiB < $MIN_FREE_MIB MiB; not running" | tee -a "$LOG" > "$ERR"
  exit 3
fi
export HOME=/home/weiland CUDA_VISIBLE_DEVICES=$GPU HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1
export OMP_NUM_THREADS=4 MKL_NUM_THREADS=4 OPENBLAS_NUM_THREADS=4
PY=/home/weiland/projects/openpi_ext/envs/gr00t_n15_venv/.venv/bin/python
G=/home/weiland/projects/openpi_ext/third_party/gr00t_n15
export PYTHONPATH=$G:$G/examples/Libero:$REPO:$REPO/src:$REPO/packages/openpi-client/src
cd "$REPO"
echo "$(date '+%F %T %Z') start suite=$SUITE S=$S K=$K seed=$SEED out=$OUT gpu=$GPU free=${FREE}MiB pid=$$" >> "$LOG"
"$PY" "$HERE/floor_resample_groot.py" --suite "$SUITE" --s "$S" --k "$K" --seed "$SEED" --out "$OUT" >> "$LOG" 2>&1
RC=$?
if [ $RC -eq 0 ]; then
  echo "$(date '+%F %T %Z') exit 0" | tee -a "$LOG" > "$DONE"
else
  echo "$(date '+%F %T %Z') exit $RC" | tee -a "$LOG" > "$ERR"
fi
exit $RC
