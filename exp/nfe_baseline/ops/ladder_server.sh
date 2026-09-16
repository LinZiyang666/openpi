#!/bin/bash
# Server-side lane driver (h100 / weilandserver): serve every k of one (policy, suite) in order.
#
# usage: ladder_server.sh <policy pi05|groot|pi05_rc|groot_rc> <suite> <ks csv> <base_port> <nprocs_file> [idle_s] [first_conn_s]
#
# The *_rc policies serve RoboCasa365 (one process; <nprocs_file> is ignored)
# through ops/rc/launch_*_rc_server.sh; give them a longer idle_s (400) because
# the client lane rebuilds its worker fleet between the main and pnp lanes.
#
# For each k: launch the servers (count re-read from <nprocs_file> at every k
# so the fleet can be resized at a k boundary), wait until at least one client
# connection has been seen and then none has been open for <idle_s> seconds
# (the client lane launches the next suite/k only after this one finishes, so
# a quiet port means the k is over), stop the servers by PID, next k. A k that
# sees no client within <first_conn_s> is released with NO-CLIENTS so a lane
# resumed on the client side (skipped k) cannot wedge this side. Markers:
# LADDER k=<k> UP / DONE / NO-CLIENTS, LADDER FINISHED.
#
# Host paths come from the environment: NFE_HOME, NFE_REPO, NFE_PI05_PY,
# NFE_PI05_CKPT, NFE_GROOT, NFE_GROOT_PY, NFE_CKPT_DIR, NFE_RC_PI05_CKPT,
# NFE_RC_GROOT_CKPT (defaults = weilandserver).
set -u
POLICY=${1:?policy}; SUITE=${2:?suite}; KS=${3:?ks csv}; BASE=${4:?base port}; NFILE=${5:?nprocs file}
IDLE_S=${6:-150}; FIRST_S=${7:-2400}
HERE=$(cd "$(dirname "$0")" && pwd)
export PATH=/usr/local/bin:/usr/bin:/bin
export NFE_HOME=${NFE_HOME:-/home/weiland}
export HOME=$NFE_HOME
REPO=${NFE_REPO:-/data/openpi_nfe}
IFS=',' read -r -a K_ARR <<< "$KS"

conns() {
  local n=$1 filt="" i
  for i in $(seq 0 $((n-1))); do
    [ -n "$filt" ] && filt="$filt or "
    filt="$filt sport = :$((BASE+i))"
  done
  ss -tnH state established "( $filt )" | wc -l
}

for K in "${K_ARR[@]}"; do
  N=$(cat "$NFILE" 2>/dev/null | tr -dc 0-9); [ -z "$N" ] && N=3
  if [ "$POLICY" = "pi05" ]; then
    bash "$HERE/launch_pi05_servers.sh" "$K" "$BASE" "$N" "$REPO" "${NFE_PI05_PY:-/home/weiland/openpi/.venv/bin/python}" \
      "${NFE_PI05_CKPT:-/home/weiland/.cache/openpi/openpi-assets/checkpoints/pi05_libero_pytorch}"
  elif [ "$POLICY" = "pi05_rc" ]; then
    N=1; NFE_REPO=$REPO bash "$HERE/rc/launch_pi05_rc_server.sh" "$K" "$BASE"
  elif [ "$POLICY" = "groot_rc" ]; then
    N=1; NFE_REPO=$REPO bash "$HERE/rc/launch_groot_rc_server.sh" "$K" "$BASE"
  else
    bash "$HERE/launch_groot_servers.sh" "$SUITE" "$K" "$BASE" "$N" "$REPO" "${NFE_GROOT:-/home/weiland/gr00t_n15}" \
      "${NFE_GROOT_PY:-/home/weiland/gr00t_n15_venv/.venv/bin/python}" "${NFE_CKPT_DIR:-/home/weiland}"
  fi
  up=0; for i in $(seq 0 $((N-1))); do ss -tlnH "sport = :$((BASE+i))" | grep -q . && up=$((up+1)); done
  if [ "$up" -ne "$N" ]; then echo "LADDER FAIL k=$K only $up/$N listening $(date +%H:%M:%S)"; exit 1; fi
  echo "LADDER k=$K UP n=$N $(date +%H:%M:%S)"
  seen=0; idle=0; waited=0
  while :; do
    c=$(conns "$N")
    if [ "$c" -gt 0 ]; then seen=1; idle=0
    elif [ "$seen" -eq 1 ]; then idle=$((idle+15))
    else waited=$((waited+15)); fi
    [ "$seen" -eq 1 ] && [ "$idle" -ge "$IDLE_S" ] && break
    if [ "$seen" -eq 0 ] && [ "$waited" -ge "$FIRST_S" ]; then echo "LADDER k=$K NO-CLIENTS $(date +%H:%M:%S)"; break; fi
    sleep 15
  done
  bash "$HERE/stop_servers.sh" "$BASE" "$N" > /dev/null 2>&1
  echo "LADDER k=$K DONE $(date +%H:%M:%S)"
done
echo "LADDER FINISHED $POLICY $SUITE $(date +%H:%M:%S)"
