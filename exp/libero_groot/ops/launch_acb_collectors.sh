#!/usr/bin/env bash
# Bring up the ActionCache-baseline cohort collectors for one suite (plan
# actioncache_baseline_groot §3.9): N non-concurrent HDF5 collector servers on
# the GR00T island.
#
# No --concurrent, no --cache-config, no --rit-shadow-out: a collector is one
# connection at a time and one HDF5 writer, and the cohort is served by the
# teacher's own 8-step action. The second connection to a collector is refused
# (1013), so the client side (run_acb_collect_clients.sh) is strictly serial
# per port.
#
# Output per server: <root>/<suite>/attempt_<a>/srv<i>/acb_shadow_<suite>/*.h5
#
# usage: launch_acb_collectors.sh <suite> <ckpt> <repo> <gr00t-root> <python> <attempt> [root] [base-port] [n]
set -eu

SUITE=${1:?suite}
CKPT=${2:?checkpoint}
REPO=${3:?repo}
GROOT=${4:?gr00t root}
PY=${5:?python}
ATTEMPT=${6:?attempt index}
ROOT=${7:-/data/libero_cache/acb_shadow_h5}
BASE=${8:-8030}
N=${9:-5}

ADIR=$ROOT/$SUITE/attempt_$ATTEMPT
if [ -d "$ADIR" ]; then
  echo "attempt dir already exists: $ADIR (attempts are never reused; pick the next index)"; exit 1
fi
mkdir -p "$ADIR"

echo "== tearing down any previous collector fleet on $BASE..$((BASE+N-1)) =="
for i in $(seq 0 $((N-1))); do tmux kill-session -t "acbsrv$((BASE+i))" 2>/dev/null || true; done
sleep 5

PYPATH=$GROOT:$GROOT/examples/Libero:$REPO:$REPO/src
echo "== starting $N collectors for $SUITE attempt_$ATTEMPT on $BASE..$((BASE+N-1)) =="
for i in $(seq 0 $((N-1))); do
  P=$((BASE+i))
  mkdir -p "$ADIR/srv$i"
  tmux new -s "acbsrv$P" -d "cd $REPO && PYTHONPATH=$PYPATH HF_HUB_OFFLINE=1 OPENPI_MONITOR_LEVEL=BASIC \
    $PY exp/libero_groot/serve_groot_libero.py \
      --checkpoint $CKPT --port $P --denoising-steps 8 \
      --trace-out $ADIR/srv$i --trace-build-cache 2>&1 | tee $ADIR/srv$i.log"
  # Staggered: N processes loading a 7 GB checkpoint at once thrash the page
  # cache and the first connection then times out against a half-warm server.
  sleep 12
done

echo "== waiting for listeners =="
for _ in $(seq 1 60); do
  up=0
  for i in $(seq 0 $((N-1))); do
    grep -q "SERVER-LISTENING" "$ADIR/srv$i.log" 2>/dev/null && up=$((up+1))
  done
  [ "$up" -eq "$N" ] && { echo "all $N listening"; break; }
  sleep 10
done
for i in $(seq 0 $((N-1))); do
  printf '%s: %s\n' "$((BASE+i))" "$(grep -m1 'serving stack' "$ADIR/srv$i.log" 2>/dev/null || echo 'NOT UP')"
done
# The attempt -> port -> output-dir mapping the verifier binds H5 files to.
{
  printf '{"suite": "%s", "attempt": %s, "checkpoint": "%s", "base_port": %s, "n": %s, "servers": [' "$SUITE" "$ATTEMPT" "$CKPT" "$BASE" "$N"
  for i in $(seq 0 $((N-1))); do
    [ "$i" -gt 0 ] && printf ', '
    printf '{"port": %s, "out_dir": "srv%s", "experiment": "acb_shadow_%s"}' "$((BASE+i))" "$i" "$SUITE"
  done
  printf '], "started_at": "%s"}\n' "$(date -Is)"
} > "$ADIR/servers.json"
echo "wrote $ADIR/servers.json"
