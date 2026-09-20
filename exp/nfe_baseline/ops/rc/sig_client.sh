#!/usr/bin/env bash
# Client box: forward every new "RCLANE k=<k> DONE" line of <lane log> to the server box's kdone_listener as "DONE k=<k>".
# usage: sig_client.sh <lane log> <server host> <port>
set -u
LOG=${1:?lane log}; HOST=${2:?host}; PORT=${3:?port}
sent=$(grep -c "^RCLANE k=[0-9]* DONE" "$LOG" 2>/dev/null)   # lines already there are history, not new signals
while :; do
  now=$(grep -c "^RCLANE k=[0-9]* DONE" "$LOG" 2>/dev/null)
  if [ "$now" -gt "$sent" ]; then
    for K in $(grep "^RCLANE k=[0-9]* DONE" "$LOG" | tail -n $((now-sent)) | sed -E 's/^RCLANE k=([0-9]*) DONE.*/\1/'); do
      for try in 1 2 3; do timeout 5 bash -c "echo 'DONE k=$K' > /dev/tcp/$HOST/$PORT" 2>/dev/null && { echo "SIG sent DONE k=$K $(date +%H:%M:%S)"; break; }; sleep 5; done
    done
    sent=$now
  fi
  sleep 3
done
