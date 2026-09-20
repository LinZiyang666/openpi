#!/bin/bash
# Client-side RoboCasa-2024 Cosmos ladder v2 (timan107 / timan108): one flat job list over all (k, task, trial range),
# launched in order whenever a pool slot is free, so the pool never drains between ks (the servers take k per request)
# and a dead shard is relaunched as soon as a slot frees instead of after the round. A shard is skipped while its tmux
# session is alive or once its result JSON is complete; at most 3 launches per shard. Markers: RC24 k=<k> UP / DONE / FAIL,
# RC24 FINISHED.   usage: ladder_rc24_client2.sh <ks csv> <ws urls csv> <workers> <gpu ids csv> [out root] [tasks csv]
set -u
KS=${1:?ks}; URLS=${2:?ws urls csv}; W=${3:?workers}; GPUS=${4:?gpu ids}; ROOT=${5:-/scratch/zixuans8/cosmos/results_rc24}
TASKS=${6:-PnPCounterToCab,PnPCabToCounter,PnPCounterToSink,PnPSinkToCounter,PnPCounterToMicrowave,PnPMicrowaveToCounter,PnPCounterToStove,PnPStoveToCounter,OpenSingleDoor,CloseSingleDoor,OpenDoubleDoor,CloseDoubleDoor,OpenDrawer,CloseDrawer,TurnOnStove,TurnOffStove,TurnOnSinkFaucet,TurnOffSinkFaucet,TurnSinkSpout,CoffeeSetupMug,CoffeeServeMug,CoffeePressButton,TurnOnMicrowave,TurnOffMicrowave}
RANGES=${RC24_RANGES:-0-9,10-19,20-29,30-39,40-49}
HERE=$(cd "$(dirname "$0")" && pwd)
export HOME=${NFE_HOME:-/home/zixuans8}
IFS=',' read -r -a K_ARR <<< "$KS"; IFS=',' read -r -a U_ARR <<< "$URLS"; IFS=',' read -r -a G_ARR <<< "$GPUS"
IFS=',' read -r -a T_ARR <<< "$TASKS"; IFS=',' read -r -a R_ARR <<< "$RANGES"

complete() { [ -f "$1" ] && grep -q '"complete": true' "$1"; }
running() { tmux ls 2>/dev/null | grep -c "^rc24_k"; }
STALL_S=${RC24_STALL_S:-1500}
reap_stalled() {  # a shard whose log has not grown for STALL_S (e.g. CUDA/EGL init failed and it spins) is killed and relaunched later
  local s now; now=$(date +%s)
  for s in $(tmux ls 2>/dev/null | grep -o "^rc24_k[^:]*"); do
    local L="/tmp/cosmos/$s.log"; [ -f "$L" ] || continue
    if [ $((now - $(stat -c %Y "$L"))) -gt "$STALL_S" ]; then tmux kill-session -t "$s" 2>/dev/null; echo "RC24 STALLED $s killed $(date +%H:%M:%S)"; fi
  done
}
declare -A tries=(); declare -A done_k=()
i=0
while :; do
  reap_stalled
  left_total=0; launched=0
  for K in "${K_ARR[@]}"; do
    left_k=0
    for T in "${T_ARR[@]}"; do for R in "${R_ARR[@]}"; do
      OUT="$ROOT/k${K}/${T}_t${R}.json"; NAME="rc24_k${K}_${T}_t${R}"; key="k${K}_${T}_${R}"
      complete "$OUT" && continue
      left_k=$((left_k+1))
      tmux has-session -t "$NAME" 2>/dev/null && continue
      [ "${tries[$key]:-0}" -ge 3 ] && continue
      while [ "$(running)" -ge "$W" ]; do sleep 15; done
      tries[$key]=$(( ${tries[$key]:-0} + 1 ))
      [ -z "${done_k[up$K]:-}" ] && { echo "RC24 k=$K UP $(date +%H:%M:%S)"; done_k[up$K]=1; }
      # skip replicas whose port is down (a server host may be temporarily borrowed); try each URL once per launch
      for _try in $(seq 1 ${#U_ARR[@]}); do
        URL="${U_ARR[$((i % ${#U_ARR[@]}))]}"; HP=${URL#ws://}; UH=${HP%%:*}; UP=${HP##*:}
        timeout 3 bash -c "</dev/tcp/$UH/$UP" 2>/dev/null && break
        i=$((i+1)); URL=""
      done
      [ -z "$URL" ] && { echo "RC24 no replica reachable, waiting $(date +%H:%M:%S)"; sleep 60; continue; }
      bash "$HERE/run_shard_rc24.sh" "$T" "$K" "$R" "${G_ARR[$((i % ${#G_ARR[@]}))]}" "$URL" "$ROOT" >/dev/null
      i=$((i+1)); launched=$((launched+1)); sleep 2
    done; done
    if [ "$left_k" -eq 0 ] && [ -z "${done_k[$K]:-}" ]; then echo "RC24 k=$K DONE $(date +%H:%M:%S)"; done_k[$K]=1; fi
    left_total=$((left_total+left_k))
  done
  [ "$left_total" -eq 0 ] && break
  if [ "$launched" -eq 0 ] && [ "$(running)" -eq 0 ]; then echo "RC24 FAIL left=$left_total (tries exhausted) $(date +%H:%M:%S)"; exit 1; fi
  sleep 30
done
echo "RC24 FINISHED $(date +%H:%M:%S)"
