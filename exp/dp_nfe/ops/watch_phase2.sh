#!/bin/bash
# Phase-2 condition watch: emits a line only when (a) a lane/eval-loop exits or a traceback appears, or (b) the count of
# verified finals / complete eval summaries changes on either host. Silent otherwise.
last=""
while true; do
  w=$(tether exec --timeout 25s weilandserver -- bash -lc 'f=$(ls /data/dp/x0_multimodal/runs/*/checkpoints/final.done 2>/dev/null | wc -l); s=$(ls /data/dp/x0_multimodal/results_trailing/*/*/summary.json 2>/dev/null | wc -l); e=$(grep -ahcE "Traceback|QUEUE_EXIT|EVAL_LOOP_(FAILED|TIMEOUT|DONE)|QUEUE DONE" /tmp/x0/queue_train_*.log /tmp/x0/eval_loop_*.log 2>/dev/null | paste -sd+ | bc); t=$(tmux ls 2>/dev/null | grep -c "^x0"); echo "finals=$f summaries=$s alerts=${e:-0} tmux=$t"' 2>/dev/null | awk '!seen[$0]++' | tail -1)
  h=$(tether exec --timeout 25s h100 -- bash -lc 'export HOME=/home/exouser; f=$(ls /data/dp_h100/x0_multimodal/runs/*/checkpoints/final.done 2>/dev/null | wc -l); s=$(ls /data/dp_h100/x0_multimodal/results_trailing/*/*/summary.json 2>/dev/null | wc -l); e=$(grep -ahcE "Traceback|QUEUE_EXIT|EVAL_LOOP_(FAILED|TIMEOUT|DONE)|QUEUE DONE" /tmp/x0/queue_train_*.log /tmp/x0/eval_loop_*.log 2>/dev/null | paste -sd+ | bc); t=$(tmux ls 2>/dev/null | grep -c "^x0"); echo "finals=$f summaries=$s alerts=${e:-0} tmux=$t"' 2>/dev/null | awk '!seen[$0]++' | tail -1)
  out="$(TZ=America/Chicago date +%H:%M) WLS[$w] H100[$h]"
  key="$w|$h"
  if [ "$key" != "$last" ]; then echo "$out"; last="$key"; fi
  sleep 300
done
