#!/usr/bin/env bash
# Bring up the evaluation servers for one suite on one lane.
#
# Difference from the shadow fleet: no startup recipe and
# ``--allow-dynamic-bundles``, so the driver ships each arm's yaml over the
# wire and one process serves all 84 arms without a restart. The guards re-run
# per bundle, so an arm whose schedule or library disagrees with this head is
# still refused -- just at swap time instead of at boot.
#
# usage: launch_eval_servers.sh <ckpt> <repo> <gr00t-root> <python> [base-port] [n]
set -eu

CKPT=${1:?checkpoint}
REPO=${2:?repo}
GROOT=${3:?gr00t root}
PY=${4:?python}
BASE=${5:-23210}
N=${6:-4}

echo "== tearing down any fleet on $BASE..$((BASE+N-1)) =="
for i in $(seq 0 $((N-1))); do tmux kill-session -t "lgsrv$((BASE+i))" 2>/dev/null || true; done
sleep 5

PYPATH=$GROOT:$GROOT/examples/Libero:$REPO:$REPO/src
echo "== starting $N eval servers on $BASE..$((BASE+N-1)) =="
for i in $(seq 0 $((N-1))); do
  P=$((BASE+i))
  tmux new -s "lgsrv$P" -d "cd $REPO && PYTHONPATH=$PYPATH HF_HUB_OFFLINE=1 OPENPI_MONITOR_LEVEL=BASIC \
    $PY exp/libero_groot/serve_groot_libero.py \
      --checkpoint $CKPT --port $P --denoising-steps 8 \
      --concurrent --allow-dynamic-bundles 2>&1 | tee /tmp/lgsrv$P.log"
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
