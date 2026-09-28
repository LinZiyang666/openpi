#!/bin/bash
# chain_p3.sh <run-root> <arm> [<arm> ...]  -- run arms one after another:
#   servers up on weilandserver (one single-replica server per port) -> run_arm.sh (run_gtp role=all) on timan107 ->
#   completion check (accepted & status in {done,failed} & no error, >= EXPECT unique uids) -> collect -> servers down.
# Coordinator only: use P3 deploy_client.sh; never sync_remote.sh (it overwrites the shared launcher).
# P3_PHASE=smoke|pilot|continuation and P3_SCHEDULE select manifests/client_env; P3_* env values override.
# Event lines for the Monitor are prefixed "EV " in <run-root>/runs/chain.log; markers <run-root>/state/<arm>.DONE |
# <arm>.ERROR and CHAIN.DONE | CHAIN.ERROR.
# env: PORTS        comma list of free ports in 23100-23199 (scan: ss -ltnH | grep -oE ':231[0-9]{2}\b'), required
#      WPS          timan107 workers per server (default 16; 4 servers x 16 = the recommended 64)
#      SERVER_CPUS  taskset list for the servers (default 34-37,78-81)
#      STAGE1_ONLY  1 = stage 2/3 weights on meta (default 1, see start_server.sh); 0 = full model. An arm whose
#                   arms.json row carries "full_model": true (emit_arms spec; mixed HIT/MISS arms) is always
#                   started with STAGE1_ONLY=0 and the full-model NEED_MB, whatever this env says
#      NEED_MB      free GPU MiB required per server before starting (default 3000 with STAGE1_ONLY=1, else
#                   9000 pi05 / 8000 groot; measured 2.2 / 1.9 GB and 7.6 / 5.7-6.7 GB)
#      OSCL_EPISODES / OSCL_TASKS  smoke subset (ep_idx / task_id lists); EXPECT follows automatically
#      OSCL_MANIFEST exact (task, init) JSON; takes precedence over Cartesian filters.
#                    Per-arm manifest field is used when this env is absent. Uploaded by chain at launch.
#      MAX_ATTEMPTS driver (re)launches per arm (default 3; run_gtp resumes from its journal)
set -u
RUN=${1:?run-root}; shift
HERE=/home/weiland/projects/openpi/exp/offline_search/closed_loop/ops
P3=/home/weiland/projects/openpi/exp/offline_search/rounds/r06/p3_profiling
R=/home/weiland/projects/openpi
TAG=$(basename "$RUN")
ISL=/scratch/zixuans8/openpi_trace/os_cl
O=$ISL/runs/$TAG
PORTS=${PORTS:?set PORTS}
WPS=${WPS:-16}
MAX_ATTEMPTS=${MAX_ATTEMPTS:-3}
export CPUS=${SERVER_CPUS:-6-9,50-53}
LOG=$RUN/runs/chain.log
STATE=$RUN/state
mkdir -p "$RUN/runs" "$STATE"
ev() { echo "EV $(date +%m-%d_%H:%M:%S) $*" | tee -a "$LOG"; }
note() { echo "   $(date +%H:%M:%S) $*" | tee -a "$LOG"; }
rx() { timeout 300 tether exec timan107 -- bash -c "$1" 2>/dev/null; }
py() { taskset -c 6-9,50-53 env OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1 CUDA_VISIBLE_DEVICES= PYTHONDONTWRITEBYTECODE=1 PYTHONPATH=.:src "$R/.venv/bin/python" "$@"; }
field() { py -c "import json,sys; a={r['arm']:r for r in json.load(open('$RUN/arms.json'))}[sys.argv[1]]; v=a[sys.argv[2]]; print(json.dumps(v) if isinstance(v,(dict,list)) else v)" "$1" "$2"; }
field_or() { py -c "import json,sys; a={r['arm']:r for r in json.load(open('$RUN/arms.json'))}[sys.argv[1]]; v=a.get(sys.argv[2], sys.argv[3]); print(json.dumps(v) if isinstance(v,(dict,list,bool)) else v)" "$1" "$2" "$3"; }

if [ -n "${OSCL_EPISODES:-}" ]; then
  ne=$(echo "$OSCL_EPISODES" | tr ',' '\n' | grep -c .)
  nt=10; [ -n "${OSCL_TASKS:-}" ] && nt=$(echo "$OSCL_TASKS" | tr ',' '\n' | grep -c .)
  EXPECT=$((ne * nt))
else
  EXPECT=500
fi
LEGACY_EXPECT=$EXPECT

selection_for() {
  local arm=$1 info
  if [ -f "$STATE/P3_STREAM_PORT" ]; then export P3_STREAM_PORT=$(cat "$STATE/P3_STREAM_PORT"); fi
  MANIFEST=${OSCL_MANIFEST:-$(field_or "$arm" manifest '')}
  local planned
  planned=$(py -m exp.offline_search.rounds.r06.p3_profiling.client_plan --run-root "$RUN" --arm "$arm" --field manifest) || return 1
  [ -z "$MANIFEST" ] && MANIFEST=$planned
  P3_ENVS=$(py -m exp.offline_search.rounds.r06.p3_profiling.client_plan --run-root "$RUN" --arm "$arm" --field env_prefix) || return 1
  EXPECT=$LEGACY_EXPECT
  DONE="$STATE/$arm.DONE"
  REMOTE_MANIFEST=""
  if [ -n "$MANIFEST" ]; then
    MANIFEST=$(realpath "$MANIFEST") || return 1
    info=$(py "$HERE/remote/count.py" --manifest-info "$MANIFEST" --model "$(field_or "$arm" model '')" --suite "$(field_or "$arm" suite '')") || return 1
    read -r EXPECT MANIFEST_SHA <<< "$info"
    DONE="$STATE/$arm.manifest_$MANIFEST_SHA.DONE"
    REMOTE_MANIFEST="$ISL/cfg/manifest_${TAG}_${arm}_${MANIFEST_SHA}.json"
  fi
}

prepare_selection() {
  local arm=$1 saved="$RUN/runs/$1/manifest.json"
  if [ -n "$MANIFEST" ]; then
    cp "$MANIFEST" "$saved.tmp" && mv "$saved.tmp" "$saved" || return 1
    # tether can only stage to /tmp; never put an unchecked caller path in a remote shell.
    local stage="/tmp/oscl_manifest_${MANIFEST_SHA}.json"
    timeout 300 tether push --force "$saved" "timan107:$stage" >/dev/null || return 1
    rx "cp '$stage' '$REMOTE_MANIFEST'" || return 1
    MANIFEST="$saved"
  fi
  if [ -n "$MANIFEST" ] || [ -f "$RUN/runs/$arm/selection.json" ]; then
    py - "$RUN/runs/$arm/selection.json" "$MANIFEST" <<'PY'
import json, os, sys
p, manifest = sys.argv[1:]
with open(p + '.tmp', 'w') as f:
    json.dump({'manifest': manifest or None}, f)
os.replace(p + '.tmp', p)
PY
  fi
}

server_args() {  # $1 arm -> NUL-separated plugin args on stdout
  py - "$RUN/arms.json" "$1" <<'PY'
import json, sys
a = {r["arm"]: r for r in json.load(open(sys.argv[1]))}[sys.argv[2]]
if a["mode"] == "stock":
    out = []
elif a["mode"] == "native":
    out = ["--os-method", "native", "--os-cell", a["cell"], *a.get("plugin_args", [])]
else:
    out = ["--os-method", a["method"], "--os-kwargs", json.dumps(a.get("kwargs") or {}), "--os-cell", a["cell"],
           *a.get("plugin_args", [])]
sys.stdout.write("".join(x + "\0" for x in out))
PY
}

servers_up() {  # $1 arm
  local arm=$1 model suite yaml mode p need free s1 seed_base args=() seed_args=() arm_env=()
  model=$(field "$arm" model); suite=$(field "$arm" suite); yaml=$(field "$arm" yaml); mode=$(field "$arm" mode)
  s1=${STAGE1_ONLY:-1}
  [ "$(field_or "$arm" full_model false)" = "true" ] && s1=0     # mixed HIT/MISS arm: stage 2/3 must be loaded
  if [ "$s1" = "1" ]; then need=${NEED_MB:-3000}
  else need=${NEED_MB:-$([ "$model" = groot ] && echo 8000 || echo 9000)}; fi
  free=$(nvidia-smi --query-gpu=memory.free --format=csv,noheader,nounits | head -1)
  local n; n=$(echo "$PORTS" | tr , '\n' | grep -c .)
  if [ "$free" -lt $((need * n)) ]; then ev "GPU_TIGHT arm=$arm free=${free}MiB need=$((need * n))MiB -> not starting"; return 1; fi
  mapfile -d '' args < <(server_args "$arm")
  seed_base=$(field_or "$arm" server_seed '')
  mapfile -d '' arm_env < <(py - "$RUN/arms.json" "$arm" <<'PY'
import json, sys
a = {r['arm']: r for r in json.load(open(sys.argv[1]))}[sys.argv[2]]
sys.stdout.write(''.join(f'{k}={v}\0' for k, v in a.get('server_env', {}).items()))
PY
)
  for p in $(echo "$PORTS" | tr , ' '); do
    seed_args=()
    if [ -n "$seed_base" ]; then
      seed_args=(--os-seed "$((seed_base * 65536 + p))")
      note "policy seed arm=$arm port=$p seed=$((seed_base * 65536 + p))"
    fi
    if ss -ltnH "sport = :$p" | grep -q .; then ev "PORT_BUSY arm=$arm port=$p"; return 1; fi
    if [ "$mode" = stock ]; then
      env "${arm_env[@]}" STOCK=1 STAGE1_ONLY=$s1 bash "$HERE/start_server.sh" "$model" "$suite" "$p" "$yaml" "$RUN/runs/$arm/server_$p" "${arm}_$p" >/dev/null || return 1
    else
      env "${arm_env[@]}" STAGE1_ONLY=$s1 bash "$HERE/start_server.sh" "$model" "$suite" "$p" "$yaml" "$RUN/runs/$arm/server_$p" "${arm}_$p" "${args[@]}" "${seed_args[@]}" >/dev/null || return 1
    fi
    sleep 10
  done
  [ "$s1" = "0" ] && note "arm $arm: full-model servers (STAGE1_ONLY=0, need ${need}MiB each)"
  local t=0
  while :; do
    local up=0
    for p in $(echo "$PORTS" | tr , ' '); do
      ss -ltnH "sport = :$p" | grep -q . && up=$((up+1))
      if grep -q "SERVER_EXIT=" "$RUN/runs/$arm/server_$p/server_${arm}_$p.log" 2>/dev/null; then
        ev "SERVER_DIED_AT_BOOT arm=$arm port=$p $(grep -E 'Error|Traceback' "$RUN/runs/$arm/server_$p/server_${arm}_$p.log" | tail -1 | cut -c1-200)"
        return 1
      fi
    done
    [ "$up" -eq "$n" ] && return 0
    t=$((t+5)); [ $t -gt 1200 ] && { ev "SERVER_BOOT_TIMEOUT arm=$arm up=$up"; return 1; }
    sleep 5
  done
}

servers_down() {  # $1 arm
  local p
  for p in $(echo "$PORTS" | tr , ' '); do
    note "server $p: $(bash "$HERE/stop_server.sh" "$RUN/runs/$1/server_$p" "${1}_$p" 120 2>&1 | tail -1)"
  done
  for p in $(echo "$PORTS" | tr , ' '); do tmux has-session -t "oscl$p" 2>/dev/null && tmux kill-session -t "oscl$p"; done
}

any_server_dead() {  # $1 arm
  local p d=""
  for p in $(echo "$PORTS" | tr , ' '); do
    grep -q "SERVER_EXIT=" "$RUN/runs/$1/server_$p/server_${1}_$p.log" 2>/dev/null && d="$d $p"
  done
  echo $d
}

run_arm() {  # $1 arm
  local arm=$1 suite servers sw a n s rows st envs
  suite=$(field "$arm" suite)
  servers=$(echo "$PORTS" | tr , '\n' | sed 's/^/ziyanglin.com:/' | paste -sd,)
  sw=$(echo "$PORTS" | tr , '\n' | sed "s/.*/$WPS/" | paste -sd,)
  echo "$arm" > "$STATE/current"
  mkdir -p "$RUN/runs/$arm"
  if [ -n "${P3_STREAM_PORT:-}" ]; then
    py -m exp.offline_search.rounds.r06.p3_profiling.stream_collect --run-root "$RUN" --arm "$arm" \
      --health "127.0.0.1:$P3_STREAM_PORT" --token-file "${P3_STREAM_TOKEN_FILE:-$RUN/state/p3_stream.token}" \
      >> "$LOG" 2>&1 || { ev "P3_RECEIVER_UNHEALTHY arm=$arm"; return 1; }
  fi
  prepare_selection "$arm" || return 1
  py -m exp.offline_search.rounds.r06.p3_profiling.client_plan --run-root "$RUN" --arm "$arm" > "$RUN/runs/$arm/p3_client_plan.json" || return 1
  ev "ARM_START arm=$arm suite=$suite ports=$PORTS workers=$sw expect=$EXPECT t107=[$(rx 'uptime | sed "s/.*average: //"; free -g | awk "/^Mem/{print \$7\"G\"}"' | tr '\n' ' ')]"
  servers_up "$arm" || return 1
  ev "SERVERS_READY arm=$arm gpu_used=$(nvidia-smi --query-gpu=memory.used --format=csv,noheader)"
  envs="$P3_ENVS"
  [ -n "${OSCL_EPISODES:-}" ] && envs="${envs}OSCL_EPISODES=$OSCL_EPISODES "
  [ -n "${OSCL_TASKS:-}" ] && envs="${envs}OSCL_TASKS=$OSCL_TASKS "
  [ -n "$REMOTE_MANIFEST" ] && envs="${envs}OSCL_MANIFEST=$REMOTE_MANIFEST "
  local replan; replan=$(field_or "$arm" replan_steps '')
  [ -n "$replan" ] && envs="${envs}OSCL_REPLAN_STEPS=$replan "
  local count_args="" collect_args=()
  if [ -n "$REMOTE_MANIFEST" ]; then
    count_args=" --manifest '$REMOTE_MANIFEST' --arm '$arm'"
    collect_args=(--manifest "$MANIFEST")
  fi
  for a in $(seq 1 "$MAX_ATTEMPTS"); do
    rx "mkdir -p $O/$arm; tmux -L oscl has-session -t oscl_$arm 2>/dev/null || tmux -L oscl new -s oscl_$arm -d '${envs}bash $ISL/run_arm_v2.sh $suite $arm $servers $sw $O/$arm >> $O/$arm/driver.log 2>&1'"
    note "driver attempt $a launched"
    while :; do
      sleep 60
      local dead; dead=$(any_server_dead "$arm")
      if [ -n "$dead" ]; then
        ev "SERVER_DIED arm=$arm ports=$dead"
        rx "tmux -L oscl send-keys -t oscl_$arm C-c"; sleep 30
        servers_down "$arm"
        for p in $dead; do
          for f in "$RUN/runs/$arm/server_$p"/server_*.log; do mv "$f" "${f%.log}.died_$(date +%H%M%S).log"; done
        done
        servers_up "$arm" || return 1
        ev "SERVERS_RESTARTED arm=$arm"
        break
      fi
      st=$(rx "tmux -L oscl has-session -t oscl_$arm 2>/dev/null && echo RUN || echo GONE \$(grep -o 'RUN_ARM_EXIT=[0-9]*' $O/$arm/driver.log | tail -1)")
      case "$st" in RUN) ;; GONE*) note "driver ended: $st"; break ;; *) note "remote status unreadable" ;; esac
    done
    read -r n s rows <<< "$(rx "/scratch/zixuans8/libero_sim/bin/python $ISL/count.py $O/$arm/journal.jsonl$count_args")"
    note "journal: complete=$n success=$s rows=$rows"
    if [ "${n:-0}" -ge "$EXPECT" ]; then
      ev "ARM_DONE arm=$arm complete=$n success=$s sr=$(py -c "print(round(${s:-0}/max(${n:-0},1),3))")"
      servers_down "$arm"
      py -m exp.offline_search.closed_loop.ops.collect --run-root "$RUN" "$arm" "${collect_args[@]}" >> "$LOG" 2>&1 \
        || { ev "COLLECT_FAILED arm=$arm"; return 1; }
      if [ -n "${P3_STREAM_PORT:-}" ]; then
        py -m exp.offline_search.rounds.r06.p3_profiling.stream_collect --run-root "$RUN" --arm "$arm" --collect >> "$LOG" 2>&1 \
          || { ev "P3_STREAM_VERIFY_FAILED arm=$arm"; return 1; }
      else
        py -m exp.offline_search.rounds.r06.p3_profiling.collect_client --run-root "$RUN" --arm "$arm" --cleanup >> "$LOG" 2>&1 \
          || { ev "P3_COLLECT_FAILED arm=$arm"; return 1; }
      fi
      touch "$DONE"
      return 0
    fi
    ev "ARM_INCOMPLETE arm=$arm attempt=$a complete=${n:-?} -> resume"
  done
  ev "ARM_FAILED arm=$arm after $MAX_ATTEMPTS attempts"
  servers_down "$arm"
  if [ -z "${P3_STREAM_PORT:-}" ]; then
    py -m exp.offline_search.rounds.r06.p3_profiling.collect_client --run-root "$RUN" --arm "$arm" >> "$LOG" 2>&1 || ev "P3_PREFIX_COLLECT_FAILED arm=$arm"
  else
    ev "P3_STREAM_PREFIX_RETAINED arm=$arm (receiver partials and any marked remote spill; no cleanup)"
  fi
  return 1
}

cd "$R" || exit 1
for arm in "$@"; do
  selection_for "$arm" || exit 2
  if [ -f "$DONE" ]; then note "skip $arm (DONE)"; continue; fi
  if ! run_arm "$arm"; then
    ev "CHAIN_STOPPED at $arm"
    echo "$arm" > "$STATE/$arm.ERROR"
    echo "stopped at $arm" > "$STATE/CHAIN.ERROR"
    echo none > "$STATE/current"
    exit 1
  fi
done
echo none > "$STATE/current"
touch "$STATE/CHAIN.DONE"
ev "CHAIN_DONE $*"
