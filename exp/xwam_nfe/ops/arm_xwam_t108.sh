#!/bin/bash
# timan108: when the Cosmos RC24 ladder prints RC24 FINISHED, signal h100 (DONE k=99 -> kdone listener :23232), wait for the
# X-WAM broker frontend :23270, then run the smoke client (task rank 23, 2 evals, GPU 2). Idempotent.
export HOME=/home/zixuans8
mkdir -p /tmp/xwam
tmux has-session -t xwam_arm 2>/dev/null || tmux new -s xwam_arm -d \
  'while ! grep -q "^RC24 FINISHED" /tmp/cosmos/rc24_full.log 2>/dev/null; do sleep 20; done; for t in 1 2 3; do timeout 5 bash -c "echo DONE\ k=99 > /dev/tcp/149.165.153.233/23232" && break; sleep 5; done; echo "XWAM ARM: signalled h100 $(date +%H:%M:%S)" >> /tmp/xwam/arm.log; while ! timeout 3 bash -c "</dev/tcp/149.165.153.233/23270" 2>/dev/null; do sleep 20; done; sleep 240; echo "XWAM ARM: broker up, smoke client $(date +%H:%M:%S)" >> /tmp/xwam/arm.log; bash /tmp/xwam/run_client_xwam.sh 23 2 149.165.153.233 23270 2 smoke >> /tmp/xwam/arm.log 2>&1'
sleep 1; tmux ls | grep xwam_arm
