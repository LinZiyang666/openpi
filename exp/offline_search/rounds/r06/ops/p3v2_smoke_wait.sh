#!/bin/bash
# v2 client smoke with GPU-aware retry (priority over line P6A): needs >= 10 GB free for 3 consecutive minutes.
cd /home/weiland/projects/openpi
D=exp/offline_search/rounds/r06/p3_profiling
RUN=/home/weiland/trace_runs/os_closed_loop/r06_p3_v2_client_smoke
ARMS=$(tr '\n' ' ' < $RUN/arm_names.txt)
for attempt in $(seq 1 30); do
  ok=0
  while true; do
    free=$(nvidia-smi --query-gpu=memory.free --format=csv,noheader,nounits | head -1)
    if [ "${free:-0}" -ge 10000 ]; then ok=$((ok+1)); else ok=0; fi
    [ $ok -ge 3 ] && break
    sleep 60
  done
  echo "SMOKE_ATTEMPT $attempt $(date -Is) free=${free}MiB"
  for f in $RUN/state/*.ERROR; do [ -e "$f" ] && rm -f "$f"; done; rm -f $RUN/state/CHAIN.ERROR
  PORTS=23164 WPS=2 SERVER_CPUS=2-5,46-49 STAGE1_ONLY=0 P3_PHASE=smoke taskset -c 2-5,46-49 bash $D/chain_p3.sh $RUN $ARMS 2>&1 | tee -a $RUN/chain_console.log
  if ! ls $RUN/state/*.ERROR >/dev/null 2>&1; then echo "SMOKE_LOOP_DONE $(date -Is)"; break; fi
  sleep 120
done
