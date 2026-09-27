#!/bin/bash
# stop_server.sh <log-dir> <tag> [timeout_s=90]  -- SIGTERM the server this line started (PID file written by
# start_server.sh), after checking the PID's cmdline still carries this tag's log dir; waits for SERVER_EXIT.
# Never pattern-kills: the PID comes from the file, the check reads /proc, the kill is a separate command.
set -u
LOGD=${1:?log-dir}; TAG=${2:?tag}; TO=${3:-90}
PIDF=$LOGD/server_$TAG.pid
LOG=$LOGD/server_$TAG.log
[ -f "$PIDF" ] || { echo "no pid file $PIDF"; exit 0; }
PID=$(cat "$PIDF")
if [ ! -d "/proc/$PID" ]; then echo "pid $PID already gone"; exit 0; fi
YAML=$(sed -n 2p "$LOGD/server_$TAG.meta" 2>/dev/null)
[ -n "$YAML" ] || { echo "no meta for $TAG; refusing"; exit 3; }
if ! tr '\0' ' ' < "/proc/$PID/cmdline" | grep -qF -- "$YAML"; then
  echo "pid $PID is not our server (cmdline lacks $YAML); refusing"; exit 3
fi
kill -TERM "$PID"
for _ in $(seq 1 "$TO"); do
  [ -d "/proc/$PID" ] || break
  sleep 1
done
if [ -d "/proc/$PID" ]; then echo "pid $PID still alive after ${TO}s"; exit 4; fi
for _ in 1 2 3 4 5; do grep -q "SERVER_EXIT=" "$LOG" 2>/dev/null && break; sleep 1; done
echo "stopped $TAG: $(grep -o 'SERVER_EXIT=[0-9]*' "$LOG" | tail -1)"
