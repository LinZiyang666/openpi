#!/bin/bash
# Server-side lane driver (h100 / weilandserver): serve every k of one (policy, suite) in order.
#
# usage: ladder_server.sh <policy pi05|groot|pi05_rc|groot_rc> <suite> <ks csv> <base_port> <nprocs_file> [idle_s] [first_conn_s]
#
# Switching is SIGNAL-DRIVEN: the client lane reports "LANE k=<k> DONE" (LIBERO) / "RCLANE k=<k> DONE"
# (RoboCasa), ops/rc/sig_client.sh forwards it as "DONE k=<k>" to ops/kdone_listener.py on this box,
# which touches /tmp/nfe/kdone_<policy>_<k>; this loop then stops the servers and launches the next k
# at once. The idle timeout (<idle_s>, default 400 s with no open connection after at least one was
# seen) is only a fallback for a lost signal -- the old idle-only protocol wasted 4-9 min per rung
# and was removed (owner, 2026-09-20). A k that sees no client within <first_conn_s> is released
# with NO-CLIENTS so a lane resumed on the client side (skipped k) cannot wedge this side. A k whose
# servers are already up (nfesrv<port> claimed) is adopted instead of relaunched.
#
# Bring-up: on this box `python3 ops/kdone_listener.py <port> <policy>` (tmux) BEFORE the lane; on the
# client box `bash ops/rc/sig_client.sh <lane log> <this host> <port>`.
# Markers: LADDER k=<k> UP / SIGNAL / DONE / NO-CLIENTS, LADDER FINISHED.
#
# The *_rc policies serve RoboCasa365 (one process; <nprocs_file> is ignored) through
# ops/rc/launch_*_rc_server.sh; give them a longer idle_s because the client lane rebuilds its worker
# fleet between the main and pnp lanes. The count is re-read from <nprocs_file> at every k so the
# fleet can be resized at a k boundary.
#
# Host paths come from the environment: NFE_HOME, NFE_REPO, NFE_PI05_PY, NFE_PI05_CKPT, NFE_GROOT,
# NFE_GROOT_PY, NFE_CKPT_DIR, NFE_RC_PI05_CKPT, NFE_RC_GROOT_CKPT (defaults = weilandserver).
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
    if [ -e "/tmp/nfe/kdone_${POLICY}_${K}" ]; then echo "LADDER k=$K SIGNAL $(date +%H:%M:%S)"; break; fi
    if [ "$seen" -eq 0 ] && [ "$waited" -ge "$FIRST_S" ]; then echo "LADDER k=$K NO-CLIENTS $(date +%H:%M:%S)"; break; fi
    sleep 15
  done
  # consume the signal: the next suite of this lane may reuse the same k
  rm -f "/tmp/nfe/kdone_${POLICY}_${K}"
  bash "$HERE/stop_servers.sh" "$BASE" "$N" > /dev/null 2>&1
  echo "LADDER k=$K DONE $(date +%H:%M:%S)"
done
echo "LADDER FINISHED $POLICY $SUITE $(date +%H:%M:%S)"
