#!/bin/bash
# Line P6A with GPU-aware retry; yields to the P3 smoke and pilot lanes: start only if >= 20 GB free AND (no smoke/pilot lane OR >= 30 GB free).
R=/home/weiland/trace_runs/os_closed_loop/r06_paper
for attempt in $(seq 1 50); do
  ok=0
  while true; do
    free=$(nvidia-smi --query-gpu=memory.free --format=csv,noheader,nounits | head -1)
    smoke=0; tmux ls 2>/dev/null | grep -qE "^(p3v2_smoke|p3pilot_L[0-9])" && smoke=1
    if [ "${free:-0}" -ge 20000 ] && { [ $smoke -eq 0 ] || [ "${free:-0}" -ge 30000 ]; }; then ok=$((ok+1)); else ok=0; fi
    [ $ok -ge 3 ] && break
    sleep 60
  done
  echo "ATTEMPT $attempt GPU_FREE_OK $(date -Is) free=${free}MiB smoke=$smoke"
  for f in $R/state/*.ERROR; do [ -e "$f" ] && rm -f "$f"; done; rm -f $R/state/CHAIN.ERROR
  for p in 23150 23151; do pid=$(ss -ltnp 2>/dev/null | grep ":$p " | grep -oP 'pid=\K[0-9]+' | head -1); [ -n "$pid" ] && ps -o args= -p $pid | grep -q 'os-tag r[0-9a-z_]*_2315[01]' && kill $pid; done
  bash /home/weiland/.claude/jobs/a607dd74/tmp/line_P6A.sh
  if ! ls $R/state/*.ERROR >/dev/null 2>&1; then echo "LOOP_DONE $(date -Is)"; break; fi
  echo "CHAIN_STOPPED_RETRY $(date -Is)"; sleep 120
done
