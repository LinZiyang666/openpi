#!/usr/bin/env bash
# Coordinator-only, cond_loop2.sh-style queue. Never overwrites its dev/test queues.
set -u
trap 'exit 130' INT
trap 'exit 143' TERM
QUEUE=${1:?usage: bash retest_loop.sh /absolute/queue_retest.txt [agent-island]}
ISLAND=${2:-/scratch/zixuans8/openpi_r16_conductor_v1}
REPO=/home/weiland/projects/openpi
OPS=/home/weiland/projects/openpi_ext/scratch/r16_ops
LOG="${QUEUE%/*}/retest_loop.log"
CPU=(taskset -c 2-9,46-53 env CUDA_VISIBLE_DEVICES= OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1 PYTHONPATH=.:src PYTHONDONTWRITEBYTECODE=1 .venv/bin/python)
for ((try=1; try<=40; try++)); do
  rc=0
  while IFS= read -r line || [[ -n "$line" ]]; do
    [[ -z "$line" || "$line" == \#* ]] && continue
    [[ "$line" == '@local /'* ]] || { echo "Invalid queue line: $line" >> "$LOG"; exit 2; }
    group=${line#@local }
    opts=(--workers 64 --port 23179)
    if [[ "$group" == */r16_conductor_retest_groot_k* ]]; then
      sps=${GROOT_SERVERS_PER_SUITE:-}
      if [[ -z "$sps" && -f "$OPS/groot_sps" ]]; then sps=$(cat "$OPS/groot_sps"); fi
      opts+=(--groot-servers-per-suite "${sps:-2}")
    fi
    # Invoke DONE groups too: existing runner authenticates/releases any leftover receipts.
    (cd "$REPO" && "${CPU[@]}" -m exp.offline_search.rounds.r16.retest_runner run \
      --group "$group" --agent-island "$ISLAND" "${opts[@]}") >> "$LOG" 2>&1 || { rc=$?; break; }
  done < "$QUEUE"
  [[ "$rc" -eq 0 ]] && exit 0
  echo "Attempt $try failed ($rc); retry in 120 s" >> "$LOG"
  sleep 120
done
exit "$rc"
