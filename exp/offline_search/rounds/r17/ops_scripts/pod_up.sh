#!/bin/bash
# pod_up.sh NODE H100PORT:PODPORT...: after a pod restart, reinstall the GR00T server env (setup.sh groot, trees on the
# NFS home are skipped) and restart the reverse tunnels H100 127.0.0.1:H100PORT -> pod 127.0.0.1:PODPORT.
# -02: 24472:23474 24473:23475 24474:23476   pod1 (jupyter-ziyang10): 24470:23470 24471:23471 24475:23472
N=$1; shift
for pair in "$@"; do
  timeout 60 tether exec $N -- bash -lc "cd /home/ziyang10/.tether-agent/zl_tunnel && HOME=/home/ziyang10/.tether-agent setsid nohup bash tunnel.sh ${pair%%:*} ${pair#*:} >> tunnel_sd_${pair%%:*}.log 2>&1 < /dev/null & echo tunnel ${pair}"
done
timeout 1800 tether exec --timeout 1790s $N -- bash -lc "cd /home/ziyang10/zl_r17 && HOME=/home/ziyang10 bash setup.sh groot rsync://none/x > setup_${N}_groot_$(date +%m%d%H%M).log 2>&1; tail -3 setup_${N}_groot_*.log | tail -3"
timeout 60 tether exec $N -- bash -lc 'ps -eo args | grep -c "[t]unnel.sh"; ss -tln | grep -E ":2347[0-6] " | wc -l'
