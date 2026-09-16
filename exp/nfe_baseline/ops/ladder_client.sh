#!/usr/bin/env bash
# Client-side lane driver (timan107): run every k of one (policy, suite) in order.
#
# usage: ladder_client.sh <policy pi05|groot> <suite> <ks csv> <host> <ports csv> <S> <apool_dir> <filter_dir> <out_dir> <resize>
#
# For each k: wait until every server port reports that k in its handshake
# (probe_metadata.py), launch the S shard clients, wait for all S exit markers,
# then require S result files with NFECLI_EXIT=0. A k whose S result files
# already exist is skipped (resume). Any shortfall stops the lane with
# LANE FAIL so nothing downstream reads a partial k. Markers: LANE k=<k> UP /
# DONE / FAIL, LANE FINISHED.
set -u
POLICY=${1:?policy}; SUITE=${2:?suite}; KS=${3:?ks csv}; HOST=${4:?host}; PORTS=${5:?ports}
S=${6:?shards}; APOOL=${7:?apool}; FILTER=${8:?filter dir}; OUT=${9:?out dir}; RESIZE=${10:?resize}
HERE=$(cd "$(dirname "$0")" && pwd)
PY=${NFE_CLIENT_PY:-/scratch/zixuans8/libero_sim/bin/python}
REPO=${NFE_CLIENT_REPO:-/scratch/zixuans8/openpi_lg}
WAIT_SERVER_S=${NFE_WAIT_SERVER_S:-7200}
TAG=${NFE_TAG:-run}
export HOME=${NFE_HOME:-/home/zixuans8}
IFS=',' read -r -a PORT_ARR <<< "$PORTS"
IFS=',' read -r -a K_ARR <<< "$KS"

results_present() {
  local k=$1 n=0
  for s in $(seq 0 $((S-1))); do [ -f "$OUT/${SUITE}_k${k}_s$s.json" ] && n=$((n+1)); done
  echo $n
}

for K in "${K_ARR[@]}"; do
  if [ "$(results_present "$K")" -eq "$S" ]; then echo "LANE k=$K already done; skip"; continue; fi
  waited=0
  while :; do
    ok=0
    for P in "${PORT_ARR[@]}"; do
      got=$(cd "$REPO" && PYTHONPATH=$REPO/packages/openpi-client/src $PY "$HERE/../probe_metadata.py" "$HOST" "$P" 2>/dev/null | tail -1)
      [ "$got" = "$K" ] && ok=$((ok+1))
    done
    [ "$ok" -eq "${#PORT_ARR[@]}" ] && break
    waited=$((waited+30))
    if [ "$waited" -ge "$WAIT_SERVER_S" ]; then echo "LANE FAIL k=$K servers never reported k ($ok/${#PORT_ARR[@]})"; exit 1; fi
    sleep 30
  done
  echo "LANE k=$K UP $(date +%H:%M:%S) servers=$ok"
  NFE_TAG=$TAG bash "$HERE/launch_clients.sh" "$SUITE" "$K" "$HOST" "$PORTS" "$S" "$APOOL" "$FILTER" "$OUT" "$RESIZE" "$TAG" || { echo "LANE FAIL k=$K launch"; exit 1; }
  while :; do
    done_n=0
    for s in $(seq 0 $((S-1))); do
      grep -q "NFECLI_EXIT=" "/tmp/nfe/nfecli_${TAG}_${SUITE}_k${K}_s$s.log" 2>/dev/null && done_n=$((done_n+1))
    done
    [ "$done_n" -eq "$S" ] && break
    sleep 60
  done
  bad=$(grep -l "NFECLI_EXIT=[1-9]" /tmp/nfe/nfecli_${TAG}_${SUITE}_k${K}_s*.log 2>/dev/null | wc -l)
  have=$(results_present "$K")
  if [ "$bad" -ne 0 ] || [ "$have" -ne "$S" ]; then
    echo "LANE FAIL k=$K bad_exit=$bad results=$have/$S $(date +%H:%M:%S)"; exit 1
  fi
  echo "LANE k=$K DONE $(date +%H:%M:%S) results=$have/$S"
done
echo "LANE FINISHED $POLICY $SUITE $(date +%H:%M:%S)"
