#!/usr/bin/env bash
# Stop THIS line's server sessions only (sdsrv<port>), by port list; releases the claim locks.
# usage: stop_servers.sh <port> [port...]
set -u
for P in "$@"; do
  NAME="sdsrv$P"
  if tmux has-session -t "$NAME" 2>/dev/null; then
    PANE_PID=$(tmux list-panes -t "$NAME" -F '#{pane_pid}' | head -1)
    # the python is a child of the pane shell: signal the process group of the pane, never a pattern
    [ -n "$PANE_PID" ] && kill -TERM -- "-$(ps -o pgid= -p "$PANE_PID" | tr -d ' ')" 2>/dev/null
    sleep 5
    tmux kill-session -t "$NAME" 2>/dev/null || true
    echo "stopped $NAME"
  else
    echo "$NAME: no session"
  fi
  rmdir "/tmp/sdiag/lock_$NAME" 2>/dev/null || true
done
