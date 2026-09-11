#!/usr/bin/env bash
# Bring up the RIT calibration (shadow) servers for one suite on one lane.
#
# The shadow serves the teacher's own action and labels every ladder rung in
# the background, so this fleet produces both the score distribution the gate's
# theta is cut from and the per-rung deviations the ladder is fitted on -- in
# one pass over the cohort.
#
# Each process writes its own JSONL and each connection inside it writes its
# own file again: the shadow flushes a whole episode under one append, so two
# writers sharing a file would interleave at line granularity with no column
# that would ever reveal it.
#
# usage: launch_shadow_servers.sh <suite> <ckpt> <repo> <gr00t-root> <python> [base-port] [n]
set -eu

SUITE=${1:?suite}
CKPT=${2:?checkpoint}
REPO=${3:?repo}
GROOT=${4:?gr00t root}
PY=${5:?python}
BASE=${6:-23210}
N=${7:-4}

OUT=$REPO/exp/libero_groot/data/rit/shadow/$SUITE
CFG=$REPO/exp/libero_groot/config/rit/$SUITE/template.yaml
[ -f "$CFG" ] || { echo "missing recipe $CFG"; exit 1; }
[ -d "$OUT/shadow_pool" ] || { echo "missing cohort pool $OUT/shadow_pool"; exit 1; }
mkdir -p "$OUT"

# Stale rows from an earlier attempt would be indistinguishable from this run's
# once they are in the same glob, and the fit reads the glob.
stale=$(ls "$OUT"/shadow_p*.conn_*.jsonl 2>/dev/null | wc -l)
if [ "$stale" -gt 0 ]; then
  ARCH=$OUT/superseded_$(date +%s)
  echo "== archiving $stale earlier shadow file(s) -> $ARCH"
  mkdir -p "$ARCH" && mv "$OUT"/shadow_p*.conn_*.jsonl "$ARCH"/
fi

echo "== tearing down any previous shadow fleet on this lane =="
for i in $(seq 0 $((N-1))); do tmux kill-session -t "lgsrv$((BASE+i))" 2>/dev/null || true; done
sleep 5

PYPATH=$GROOT:$GROOT/examples/Libero:$REPO:$REPO/src
echo "== starting $N shadow servers for $SUITE on $BASE..$((BASE+N-1)) =="
for i in $(seq 0 $((N-1))); do
  P=$((BASE+i))
  tmux new -s "lgsrv$P" -d "cd $REPO && PYTHONPATH=$PYPATH HF_HUB_OFFLINE=1 OPENPI_MONITOR_LEVEL=BASIC \
    $PY exp/libero_groot/serve_groot_libero.py \
      --checkpoint $CKPT --port $P --denoising-steps 8 --concurrent \
      --cache-config $CFG \
      --rit-shadow-out $OUT/shadow_p$P.jsonl 2>&1 | tee /tmp/lgsrv$P.log"
  # Staggered: four processes loading a 7 GB checkpoint at once thrash the page
  # cache and the first connection then times out against a half-warm server.
  sleep 12
done

echo "== waiting for listeners =="
for _ in $(seq 1 60); do
  up=0
  for i in $(seq 0 $((N-1))); do
    grep -q "SERVER-LISTENING" "/tmp/lgsrv$((BASE+i)).log" 2>/dev/null && up=$((up+1))
  done
  [ "$up" -eq "$N" ] && { echo "all $N listening"; break; }
  sleep 10
done
for i in $(seq 0 $((N-1))); do
  P=$((BASE+i))
  printf '%s: %s\n' "$P" "$(grep -m1 'serving stack' /tmp/lgsrv$P.log 2>/dev/null || echo 'NOT UP')"
done
