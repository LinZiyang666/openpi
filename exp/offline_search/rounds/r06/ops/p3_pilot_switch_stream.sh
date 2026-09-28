#!/bin/bash
# Switch the pilot's remaining arms to streaming (codex STREAMING.md): deploy, receiver 23171, health, stop file-mode
# lanes L1/L2 at their arm boundary, release claims of unfinished arms, start stream lanes L3 (23165) / L4 (23164).
set -uo pipefail
cd /home/weiland/projects/openpi
D=exp/offline_search/rounds/r06/p3_profiling
RUN=/home/weiland/trace_runs/os_closed_loop/r06_p3_pilot
RX=23171
P3PY=(taskset -c 6-9,50-53 env OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1 CUDA_VISIBLE_DEVICES='' PYTHONDONTWRITEBYTECODE=1 PYTHONPATH=.:src .venv/bin/python)
echo "SWITCH_START $(date -Is)"
bash "$D/deploy_client.sh" "$RUN/client_bundle_stream" > "$RUN/client_deploy_stream.log" 2>&1 || { echo DEPLOY_FAILED; exit 1; }
echo "DEPLOY_OK"
tmux has-session -t p3rx_r06_p3_pilot 2>/dev/null || tmux new-session -d -s p3rx_r06_p3_pilot "cd /home/weiland/projects/openpi && taskset -c 6-9,50-53 env OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1 CUDA_VISIBLE_DEVICES='' PYTHONDONTWRITEBYTECODE=1 PYTHONPATH=.:src .venv/bin/python -m exp.offline_search.rounds.r06.p3_profiling.stream_receiver --run-root '$RUN' --port '$RX' --ready-file '$RUN/state/p3_stream.ready.json' >> '$RUN/receiver.log' 2>&1"
for i in $(seq 1 30); do "${P3PY[@]}" -m exp.offline_search.rounds.r06.p3_profiling.stream_collect --run-root "$RUN" --health "127.0.0.1:$RX" >/dev/null 2>&1 && break; sleep 1; done
"${P3PY[@]}" -m exp.offline_search.rounds.r06.p3_profiling.stream_collect --run-root "$RUN" --health "127.0.0.1:$RX" || { echo RECEIVER_UNHEALTHY; exit 1; }
echo "RECEIVER_OK $(date -Is)"
touch "$RUN/STOP_L1" "$RUN/STOP_L2"
while tmux has-session -t p3pilot_L1 2>/dev/null || tmux has-session -t p3pilot_L2 2>/dev/null; do sleep 15; done
echo "FILE_LANES_STOPPED $(date -Is)"
for c in "$RUN"/state/claim_*; do
  a=${c##*/claim_}
  if ! ls "$RUN/state/$a".manifest_*.DONE >/dev/null 2>&1 && [ ! -e "$RUN/state/$a.GAVEUP" ]; then
    echo "RELEASE_CLAIM $a"; [ -e "$c/by" ] && rm "$c/by"; rmdir "$c"
  fi
done
for L in "L3 23165 30-33,74-77" "L4 23164 10-13,54-57"; do
  set -- $L
  tmux new -s p3pilot_$1 -d "CHAIN=$RUN/ops/chain_p3.stream.frozen.sh STREAM_ENV=P3_STREAM_PORT=$RX bash /home/weiland/.claude/jobs/a607dd74/tmp/p3_pilot_line.sh $1 $2 $3"
done
sleep 3; tmux ls | grep p3pilot
echo "SWITCH_DONE $(date -Is)"
