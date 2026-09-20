#!/bin/bash
# Client-side RoboCasa-2024 Cosmos ladder (timan107 / timan108): for each k, the 24 tasks x 5 trial ranges (0-9 .. 40-49,
# one test scene each) as 120 shard jobs through a pool of <workers> concurrent run_shard_rc24.sh processes, servers and
# GPUs assigned round-robin. A shard whose result JSON is already complete is skipped (resume); a shard that exits without
# a complete result is relaunched up to 2 more times. Markers: RC24 k=<k> UP / DONE / FAIL, RC24 FINISHED.
# usage: ladder_rc24_client.sh <ks csv> <ws urls csv> <workers> <gpu ids csv> [out root] [tasks csv]
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

for K in "${K_ARR[@]}"; do
  echo "RC24 k=$K UP $(date +%H:%M:%S)"
  declare -A tries=()
  for round in 0 1 2; do
    i=0; pending=0
    for T in "${T_ARR[@]}"; do for R in "${R_ARR[@]}"; do
      OUT="$ROOT/k${K}/${T}_t${R}.json"
      complete "$OUT" && continue
      pending=$((pending+1))
      key="${T}_t${R}"; [ "${tries[$key]:-0}" -gt "$round" ] && continue
      while [ "$(running)" -ge "$W" ]; do sleep 20; done
      tries[$key]=$((round+1))
      bash "$HERE/run_shard_rc24.sh" "$T" "$K" "$R" "${G_ARR[$((i % ${#G_ARR[@]}))]}" "${U_ARR[$((i % ${#U_ARR[@]}))]}" "$ROOT" >/dev/null
      i=$((i+1)); sleep 2
    done; done
    while [ "$(running)" -gt 0 ]; do sleep 30; done
    left=0; for T in "${T_ARR[@]}"; do for R in "${R_ARR[@]}"; do complete "$ROOT/k${K}/${T}_t${R}.json" || left=$((left+1)); done; done
    echo "RC24 k=$K round=$round launched=$i left=$left $(date +%H:%M:%S)"
    [ "$left" -eq 0 ] && break
  done
  unset tries
  if [ "$left" -ne 0 ]; then echo "RC24 FAIL k=$K left=$left $(date +%H:%M:%S)"; exit 1; fi
  echo "RC24 k=$K DONE $(date +%H:%M:%S)"
done
echo "RC24 FINISHED $(date +%H:%M:%S)"
