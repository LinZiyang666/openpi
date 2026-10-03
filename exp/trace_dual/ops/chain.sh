#!/bin/bash
# chain.sh <arm> [<arm> ...]  -- run groups serially: servers up -> remote run_gtp -> verify 500 -> servers down.
# Event lines (for the Monitor) are prefixed "EV "; everything else is detail.
set -u
B=/home/weiland/trace_runs/dual_20260923
O=/scratch/zixuans8/trace_runs/dual_20260923
PORTS=${PORTS:-23100,23101,23102,23103}
WPS=${WPS:-16}
MAX_ATTEMPTS=${MAX_ATTEMPTS:-4}
LOG=$B/runs/chain.log
mkdir -p $B/runs $B/state
ev() { echo "EV $(date +%m-%d_%H:%M:%S) $*" | tee -a $LOG; }
note() { echo "   $(date +%H:%M:%S) $*" | tee -a $LOG; }
rx() { timeout 300 tether exec timan107 -- bash -c "$1" 2>/dev/null; }

suite_of() { case "$1" in *_sp_*) echo libero_spatial;; *_l10_*) echo libero_10;; esac; }
ckpt_of() { case "$1" in *_sp_*) echo /data/ckpt/n15_libero_spatial;; *_l10_*) echo /data/ckpt/n15_libero_10;; esac; }

servers_up() {  # $1 arm
  local arm=$1 p pids=()
  for p in $(echo $PORTS | tr , ' '); do
    if ss -ltnH "sport = :$p" | grep -q .; then continue; fi
    tmux has-session -t trsrv$p 2>/dev/null && tmux kill-session -t trsrv$p
    case $arm in
      *pi05*) $B/ops/start_pi05.sh $p $B/config/$arm.yaml $B/runs/$arm/trace $B/runs/$arm/server_$p.log >/dev/null ;;
      *groot*) $B/ops/start_groot.sh $p $(ckpt_of $arm) $B/config/$arm.yaml $B/runs/$arm/trace $B/runs/$arm/server_$p.log >/dev/null ;;
    esac
    sleep 8
  done
  local t=0
  while :; do
    local up=0
    for p in $(echo $PORTS | tr , ' '); do
      ss -ltnH "sport = :$p" | grep -q . && up=$((up+1))
      grep -q "SERVER_EXIT=" $B/runs/$arm/server_$p.log 2>/dev/null && { ev "SERVER_DIED_AT_BOOT arm=$arm port=$p $(grep -E 'Traceback|Error' $B/runs/$arm/server_$p.log | tail -1 | cut -c1-200)"; return 1; }
    done
    [ $up -eq $(echo $PORTS | tr , '\n' | wc -l) ] && return 0
    t=$((t+5)); [ $t -gt 1200 ] && { ev "SERVER_BOOT_TIMEOUT arm=$arm up=$up"; return 1; }
    sleep 5
  done
}

servers_down() {  # $1 arm
  local p pid
  for p in $(echo $PORTS | tr , ' '); do
    pid=$(ss -ltnpH "sport = :$p" | grep -o 'pid=[0-9]*' | head -1 | cut -d= -f2)
    [ -n "$pid" ] && kill -TERM $pid
  done
  for i in $(seq 1 60); do
    local alive=0
    for p in $(echo $PORTS | tr , ' '); do tmux has-session -t trsrv$p 2>/dev/null && alive=$((alive+1)); done
    [ $alive -eq 0 ] && break; sleep 3
  done
  for p in $(echo $PORTS | tr , ' '); do note "server $p: $(grep -E 'drained|SERVER_EXIT' $B/runs/$1/server_$p.log | tail -2 | tr '\n' ' ' | cut -c1-200)"; done
}

any_server_dead() {  # $1 arm -> echo ports whose server exited
  local p d=""
  for p in $(echo $PORTS | tr , ' '); do grep -q "SERVER_EXIT=" $B/runs/$1/server_$p.log 2>/dev/null && d="$d $p"; done
  echo $d
}

count_done() {  # $1 arm -> "unique_uids accepted_terminal success"
  rx "python3 - <<'PY'
import json
u={}
try:
    for l in open('$O/$1/journal.jsonl'):
        try: r=json.loads(l)
        except Exception: continue
        # accepted terminal outcome: done (success) or failed (normal task failure), never an errored row
        if r.get('accepted') and r.get('status') in ('done','failed') and not r.get('error'): u[r['task_uid']]=r
except FileNotFoundError: pass
print(len(u), sum(1 for r in u.values() if r.get('success')))
PY"
}

run_group() {  # $1 arm
  local arm=$1 suite=$(suite_of $1) servers sw a rc
  servers=$(echo $PORTS | tr , '\n' | sed 's/^/ziyanglin.com:/' | paste -sd,)
  sw=$(echo $PORTS | tr , '\n' | sed "s/.*/$WPS/" | paste -sd,)
  echo $arm > $B/state/current
  mkdir -p $B/runs/$arm
  ev "GROUP_START arm=$arm suite=$suite ports=$PORTS workers=$sw"
  servers_up $arm || return 1
  ev "SERVERS_READY arm=$arm gpu=$(nvidia-smi --query-gpu=memory.used --format=csv,noheader)"
  for a in $(seq 1 $MAX_ATTEMPTS); do
    rx "mkdir -p $O/$arm; tmux has-session -t trc_$arm 2>/dev/null || tmux new -s trc_$arm -d 'bash /scratch/zixuans8/openpi_trace/run_group.sh $suite $arm $servers $sw $O/$arm 2>&1 | tee -a $O/$arm/driver.log'"
    note "driver attempt $a launched"
    while :; do
      sleep 60
      local dead=$(any_server_dead $arm)
      if [ -n "$dead" ]; then
        ev "SERVER_DIED arm=$arm ports=$dead $(grep -hE 'Traceback|Error|exit 3' $B/runs/$arm/server_*.log | tail -1 | cut -c1-200)"
        rx "tmux send-keys -t trc_$arm C-c" ; sleep 30
        servers_down $arm
        for p in $dead; do mv $B/runs/$arm/server_$p.log $B/runs/$arm/server_$p.died_$(date +%H%M%S).log; done
        servers_up $arm || return 1
        ev "SERVERS_RESTARTED arm=$arm"
        break
      fi
      local st=$(rx "tmux has-session -t trc_$arm 2>/dev/null && echo RUN || echo GONE \$(grep -o 'RUN_GROUP_EXIT=[0-9]*' $O/$arm/driver.log | tail -1)")
      case "$st" in RUN) ;; GONE*) note "driver ended: $st"; break;; *) note "remote status unreadable";; esac
    done
    read n s <<< "$(count_done $arm)"
    note "journal: accepted_done=$n success=$s"
    if [ "${n:-0}" -ge 500 ]; then
      ev "GROUP_DONE arm=$arm done=$n success=$s"
      servers_down $arm
      return 0
    fi
    ev "GROUP_INCOMPLETE arm=$arm attempt=$a done=${n:-?} -> resume"
  done
  ev "GROUP_FAILED arm=$arm after $MAX_ATTEMPTS attempts"
  servers_down $arm
  return 1
}

for arm in "$@"; do
  run_group $arm || { ev "CHAIN_STOPPED at $arm"; echo none > $B/state/current; exit 1; }
done
echo none > $B/state/current
ev "CHAIN_DONE $*"
