#!/bin/bash
# start_eval_server.sh — weilandserver srv0: 4 replicas behind :23150 (public 1:1 NAT), pi05_libero.
export PATH=/usr/local/bin:/usr/bin:/bin
export HOME=/home/weiland
tmux kill-session -t srv0 2>/dev/null
sleep 1
P=$(pgrep -f "[s]erve_policy.py --replicas 4"); [ -n "$P" ] && kill $P && sleep 3
ss -ltn | grep -E ":2315[0-4]" && echo "PORT BUSY" && exit 1
rm -f /tmp/srv0.log
tmux new -s srv0 -d 'cd /data/openpi_dispatch && export HOME=/home/weiland && export PYTHONPATH=/data/openpi_dispatch/src:/data/openpi_dispatch && export OPENPI_SERVER_GPU_MEMORY_LOCK=0 && /home/weiland/openpi/.venv/bin/python scripts/serve_policy.py --replicas 4 --replica-spawn-batch 2 --port 23150 policy:checkpoint --policy.config pi05_libero --policy.dir /home/weiland/.cache/openpi/openpi-assets/checkpoints/pi05_libero_pytorch 2>&1 | tee /tmp/srv0.log'
echo "srv0 launched"
