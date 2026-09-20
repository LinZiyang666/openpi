#!/bin/bash
# h100: when timan108 signals that its Cosmos RC24 ladder finished (kdone_listener touches /tmp/nfe/kdone_pi05_rc_99),
# stop the 10 Cosmos replicas and start the X-WAM broker + one policy server (action steps 10) for the smoke. Idempotent.
export HOME=/home/exouser; export PATH=/home/exouser/.local/bin:/usr/local/bin:/usr/bin:/bin
mkdir -p /tmp/xwam
tmux has-session -t xwam_arm 2>/dev/null || tmux new -s xwam_arm -d \
  "while [ ! -e /tmp/nfe/kdone_pi05_rc_99 ]; do sleep 15; done; echo \"XWAM ARM: RC24 done on t108, freeing h100 \$(date +%H:%M:%S)\" >> /tmp/xwam/arm.log; bash /data/cosmos/openpi_exp/exp/cosmos_nfe/ops/rc24/stop_rc24_servers.sh 10 23250 >> /tmp/xwam/arm.log 2>&1; sleep 10; bash /tmp/xwam/serve_xwam.sh 10 50 0 23270 23271 0 >> /tmp/xwam/arm.log 2>&1"
sleep 1; tmux ls | grep xwam_arm
