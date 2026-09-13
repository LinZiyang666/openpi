#!/usr/bin/env bash
# Take a lane's fleet down on a simulator box, agents first.
#
# The agent owns its workers' process groups and signals them on SIGTERM, so
# stopping the agents is what stops the fleet; the sweep afterwards is for
# anything a crashed agent left behind. Only this experiment's sessions and
# processes are named -- the box is shared.
#
# usage: stop_fleet.sh
set -uo pipefail

for S in $(tmux ls 2>/dev/null | awk -F: '/^cpag/{print $1}'); do
  echo "== stopping $S"
  tmux send-keys -t "$S" C-c 2>/dev/null || true
done
sleep 8
pkill -TERM -f "[r]un_size_eval --role agent" 2>/dev/null || true
sleep 5
left=$(pgrep -fc "examples.libero.[w]orker_entry" || true)
if [ "${left:-0}" -gt 0 ]; then
  echo "== sweeping $left leftover worker process(es)"
  pkill -TERM -f "examples.libero.[w]orker_entry" 2>/dev/null || true
  sleep 5
  pkill -KILL -f "examples.libero.[w]orker_entry" 2>/dev/null || true
fi
for S in $(tmux ls 2>/dev/null | awk -F: '/^cpag/{print $1}'); do
  tmux kill-session -t "$S" 2>/dev/null || true
done
echo "workers left: $(pgrep -fc "examples.libero.[w]orker_entry" || echo 0)"
