#!/usr/bin/env bash
# Central worker scheduler on the A100 (tether node a100, user exouser, public 149.165.152.232), system python3.
#
#   deploy_a100.sh start  [port]   copy __init__/allocate/server.py, (re)start the restart loop, check /healthz
#   deploy_a100.sh stop   [port]   kill the loop's process group (loop + server)
#   deploy_a100.sh status [port]   loop processes, last log lines, /healthz over the public address
#
# Port defaults to 23500. tether only runs commands: the three files travel inside the exec argv (base64).
# On the A100: code in /home/exouser/scheduler/code_<port>, logs and loop.pid in /home/exouser/scheduler/run_<port>
# (server.log = stdout/stderr and restarts, allocations.jsonl = allocation changes).
set -euo pipefail
cmd=${1:?usage: deploy_a100.sh start|stop|status [port]}
port=${2:-23500}
node=a100
host=149.165.152.232
here=$(cd "$(dirname "$0")" && pwd)
base=/home/exouser/scheduler
code=$base/code_$port
run=$base/run_$port
pkg=$code/exp/offline_search/rounds/r17/scheduler

remote() { tether exec --safe --timeout 60s "$node" -- bash -c "$1"; }

stop_script() {
  cat <<EOF
if [ -f $run/loop.pid ]; then kill -TERM -- -\$(cat $run/loop.pid) 2>/dev/null || true; rm -f $run/loop.pid; sleep 1; fi
EOF
}

start_script() {
  stop_script
  echo "set -e; mkdir -p $pkg $run; touch $code/exp/__init__.py $code/exp/offline_search/__init__.py"
  for f in __init__.py allocate.py server.py; do
    echo "echo $(base64 -w0 "$here/$f") | base64 -d > $pkg/$f"
  done
  cat <<EOF
cd $code
setsid nohup bash -c 'while true; do python3 -u -m exp.offline_search.rounds.r17.scheduler.server --port $port --log $run/allocations.jsonl; echo "server exited \$? at \$(date -Is); restart in 5 s"; sleep 5; done' >> $run/server.log 2>&1 < /dev/null &
echo \$! > $run/loop.pid
echo "loop pid \$(cat $run/loop.pid)"
EOF
}

health() {
  curl -s --noproxy '*' --max-time 5 "http://$host:$port/healthz" && echo || echo "healthz: no answer from $host:$port"
}

case $cmd in
  start)
    remote "$(start_script)"
    sleep 3
    health
    ;;
  stop)
    remote "$(stop_script)"
    ;;
  status)
    remote "if [ -f $run/loop.pid ]; then pgrep -a -g \$(cat $run/loop.pid); else echo 'not running'; fi; tail -n 5 $run/server.log 2>/dev/null || true"
    health
    ;;
  *)
    echo "usage: deploy_a100.sh start|stop|status [port]" >&2
    exit 2
    ;;
esac
