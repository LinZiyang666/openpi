#!/usr/bin/env bash
# L1 self-healer for the weighted-sum search. Runs on weilandserver in its own
# tmux so it outlives the operator's session.
#
# The search scheduler is a single process driving all six slots, so its death
# stops the whole line -- and leaves two kinds of orphan behind: local cache
# servers still holding ~6 GB of VRAM each, and timan107 sim workers still
# hammering a port nobody serves. Restarting the scheduler is safe and
# idempotent: it resumes by artifact (a cell with a complete results file is
# skipped) and it kills the previous server and client fleet for a slot before
# claiming a cell.
#
# It heals only three things:
#   * keepwarm  -- iron rule; a cold 4090 miscomputes silently under load.
#   * the scheduler -- restarted while cells remain.
#   * orphan servers -- only when the scheduler is gone, never while it runs
#     (a running scheduler owns those sessions and would race the reaper).
set -u
export HOME=/home/weiland
REPO=/home/weiland/openpi
STATE=/data/libero_cache/search/current_search.env
INTERVAL=${1:-300}

log() { echo "[$(date '+%m-%d %H:%M:%S')] $*"; }

heal_keepwarm() {
  tmux has-session -t keepwarm 2>/dev/null && return
  log "HEAL keepwarm was dead -> restarting"
  tmux new -s keepwarm -d "cd $REPO && /home/weiland/.local/bin/uv run python /home/weiland/gtp_logs/gpu_keepwarm.py 2>&1 | tee -a /home/weiland/gtp_logs/keepwarm.log"
}

reap_orphans() {
  # Only reachable with the scheduler gone: nothing else owns these.
  for s in $(tmux ls 2>/dev/null | grep -oE '^libsrv[0-9]+'); do
    log "REAP orphan server $s"
    tmux kill-session -t "$s" 2>/dev/null
  done
  tether exec timan107 -- bash -lc \
    'export HOME=/home/zixuans8; n=0; for s in $(tmux ls 2>/dev/null | grep -oE "^lw[0-9]+_[0-9]+"); do tmux kill-session -t $s 2>/dev/null; n=$((n+1)); done; echo "reaped $n"' \
    2>/dev/null | tail -1
}

log "search watchdog up (interval ${INTERVAL}s)"
while true; do
  heal_keepwarm
  if [ -r "$STATE" ]; then
    # shellcheck disable=SC1090
    . "$STATE"
    done_n=$(ls "$RESULTS"/*.json 2>/dev/null | grep -vc partial)
    if tmux has-session -t libsearch 2>/dev/null; then
      :
    elif [ "${done_n:-0}" -ge "$TOTAL" ]; then
      log "all $TOTAL cells done; watchdog idle"
    else
      log "HEAL scheduler gone at $done_n/$TOTAL -> reaping orphans and restarting"
      reap_orphans
      sleep 10
      tmux new -s libsearch -d "cd $REPO && PYTHONPATH=$REPO:$REPO/src .venv/bin/python exp/libero_groot/orchestrate_search.py --yaml-dir $YAML_DIR --results-dir $RESULTS --suite $SUITE --checkpoint $CKPT --ports $PORTS --workers $WORKERS 2>&1 | tee -a $LOGFILE"
    fi
  fi
  sleep "$INTERVAL"
done
