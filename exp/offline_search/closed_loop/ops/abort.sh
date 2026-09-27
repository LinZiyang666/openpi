#!/bin/bash
# abort.sh <run-root> <arm>  -- stop one arm cleanly: Ctrl-C its timan107 driver (run_gtp stops its WorkerAgent and
# workers; the journal keeps every finished episode, a relaunch resumes), then SIGTERM this arm's servers by PID.
# Kill the local chain first (tmux kill-session -t oscl_chain) if it is still running, or it will restart things.
set -u
RUN=${1:?run-root}; ARM=${2:?arm}
HERE=$(cd "$(dirname "$0")" && pwd)
timeout 120 tether exec timan107 -- bash -c "tmux -L oscl has-session -t oscl_$ARM 2>/dev/null && tmux -L oscl send-keys -t oscl_$ARM C-c && echo driver_interrupted || echo no_driver"
for d in "$RUN/runs/$ARM"/server_*; do
  [ -d "$d" ] || continue
  p=${d##*/server_}
  bash "$HERE/stop_server.sh" "$d" "${ARM}_$p" 120
done
echo "worker processes left on timan107 (should drop to 0 within ~30 s; they are reaped by run_gtp's agent):"
timeout 60 tether exec timan107 -- bash -c "pgrep -fc '[w]orker_entry' || true"
