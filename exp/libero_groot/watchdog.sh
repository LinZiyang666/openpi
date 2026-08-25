#!/usr/bin/env bash
# L1 self-healer for the LIBERO collection fleet. Runs on weilandserver in its
# own tmux, so it survives the controlling session dying, a WSL reboot, or a
# lost tether -- the layers above it (Monitor, cron) only observe and escalate.
#
# It heals three things and deliberately nothing else:
#   * keepwarm  -- iron rule; a cold 4090 miscomputes silently under load.
#   * servers   -- a dead port strands its lane.
#   * lanes     -- but ONLY a lane whose shard still has unfinished episodes.
#
# That last condition is what lets `l10chain` work: a lane that finished its
# shard must be allowed to stay down, or the lane count never reaches zero and
# the suite handoff never fires. "Finished" is decided from the HDF5 on disk,
# not from the exit status, because a lane can also die after its last flush.
set -u
export HOME=/home/weiland
REPO=/home/weiland/openpi
PY=/home/weiland/gr00t_n15_venv/.venv/bin/python
STATE=/data/libero_cache/current_run.env
INTERVAL=${1:-300}

log() { echo "[$(date '+%m-%d %H:%M:%S')] $*"; }

heal_keepwarm() {
  tmux has-session -t keepwarm 2>/dev/null && return
  log "HEAL keepwarm was dead -> restarting"
  tmux new -s keepwarm -d "cd $REPO && /home/weiland/.local/bin/uv run python /home/weiland/gtp_logs/gpu_keepwarm.py 2>&1 | tee -a /home/weiland/gtp_logs/keepwarm.log"
}

# Episodes of lane i's shard that have no HDF5 yet. With $2 set, the remainder
# is also written there as a fresh filter file.
lane_remaining() {  # $1=lane index [$2=resume shard out] -> prints count
  local SRC="$SHARDS/${SUITE}_lane$1.json"
  [ -f "$SRC" ] || { echo 0; return; }
  cd "$REPO" && $PY exp/libero_groot/make_shards.py --trials 50 \
    --from-shard "$SRC" --done-dir "$OUT" ${2:+--out "$2"}
}

start_server() {  # $1=port
  local P=$1
  log "HEAL server $P down -> restarting"
  tmux kill-session -t "lbsrv$P" 2>/dev/null
  tmux new -s "lbsrv$P" -d "cd $REPO && PYTHONPATH=/home/weiland/gr00t_n15:/home/weiland/gr00t_n15/examples/Libero:$REPO:$REPO/src OPENPI_MONITOR_LEVEL=BASIC $PY exp/libero_groot/serve_groot_libero.py --checkpoint $CKPT --port $P --collect-hdf5 $OUT --experiment groot_$SUITE 2>&1 | tee /tmp/lbsrv$P.log"
  for _ in $(seq 1 30); do ss -tln | grep -q ":$P " && return 0; sleep 5; done
  log "WARN server $P did not come up within 150s"
  return 1
}

start_lane() {  # $1=lane index
  local i=$1 P=$((BASE+i)) F="$SHARDS/${SUITE}_lane$1.resume.json"
  local TASKS
  # Restart from what is LEFT, never from the original shard: replaying an
  # episode whose HDF5 already exists writes a second file under the same
  # global episode id, and that duplicate trips the suite-handoff check.
  # The original shard is untouched -- it is the lane's ownership record, and
  # lane_remaining reads it to decide whether the lane is finished at all.
  lane_remaining "$i" "$F" >/dev/null
  TASKS=$($PY -c "import json,sys;e=json.load(open(sys.argv[1]));print(' '.join(str(t) for t in sorted({x['task_id'] for x in e})))" "$F")
  [ -z "$TASKS" ] && return 0
  log "HEAL lane$i (port $P, tasks=[$TASKS]) -> restarting"
  tmux new -s "lbrun$i" -d "cd $REPO && MUJOCO_EGL_DEVICE_ID=0 PYTHONPATH=. /home/weiland/miniconda3/bin/conda run -p /home/weiland/libero_sim --no-capture-output python examples/libero/main.py --host 127.0.0.1 --port $P --task-suite-name $SUITE --task-ids $TASKS --num-trials-per-task 50 --num-workers 1 --resize-size 256 --replan-steps 5 --init-states-dir $REPO/exp/common/data/db_init/libero/$SUITE --cuda-visible-devices 0 --episode-filter $F --save-episode-results --episode-results-path $OUT/results_lane$i.json 2>&1 | tee /tmp/lbrun$i.log"
}

log "watchdog up (interval ${INTERVAL}s)"
while true; do
  heal_keepwarm
  if [ -r "$STATE" ]; then
    # shellcheck disable=SC1090
    . "$STATE"
    for i in $(seq 0 $((LANES-1))); do
      P=$((BASE+i))
      rem=$(lane_remaining "$i" 2>/dev/null || echo 0)
      [ "${rem:-0}" -eq 0 ] && continue          # shard finished: leave it down
      ss -tln | grep -q ":$P " || start_server "$P"
      tmux has-session -t "lbrun$i" 2>/dev/null || start_lane "$i"
    done
  fi
  sleep "$INTERVAL"
done
