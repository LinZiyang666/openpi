#!/bin/bash
# pod_resume.sh RUN OLD_TAG NEW_TAG [REST...]: after a pod restart (pod back, setup.sh and tunnels done), hold chain
# OLD_TAG, SIGTERM RUN's launcher, draft RUNc1 with every unaccepted episode (resume --remaining --verify-closed),
# write its wrappers with RUN's lane env, and start chain NEW_TAG: RUNc1 then REST (the later rounds).
set -u
R=$1; OLD=$2; NEW=$3; shift 3
O=/home/weiland/projects/openpi_ext/scratch/r17_ops; C=/home/weiland/projects/openpi/exp/offline_search/rounds/r17/filter_failed_knob/commands
ENV=$(grep -o 'R17_H100_NODE=.*' $C/launch_$R.sh | head -1); [ -n "$ENV" ] || { echo "no lane env in launch_$R.sh"; exit 2; }
touch $O/mw/hold_ff_$OLD
pid=$(pgrep -f "[f]ilter_failed_knob.launch --run /data/mw_finetune/eval/$R\$")
[ -n "$pid" ] && { ps -o pid,args -p $pid | cut -c1-160; kill -TERM $pid; }
until grep -q "\[ffk-$OLD\] $R ended" $O/mw/chain.log; do sleep 10; done
grep "\[ffk-$OLD\] $R ended" $O/mw/chain.log | tail -1
cd /home/weiland/projects/openpi
env $ENV PYTHONPATH=.:src .venv/bin/python -m exp.offline_search.rounds.r17.filter_failed.resume \
  --source /data/mw_finetune/eval/$R --name ${R}c1 --remaining --verify-closed || { echo "resume failed"; exit 1; }
for step in prepare:filter_failed.stage launch:filter_failed_knob.launch; do
  cat > $C/${step%%:*}_${R}c1.sh <<EOF
#!/usr/bin/env bash
set -euo pipefail
# continuation of $R after the $(date +%m-%d) pod restart (resume.py --remaining --verify-closed)
cd /home/weiland/projects/openpi; export PYTHONPATH=.:src
export $ENV
exec .venv/bin/python -m exp.offline_search.rounds.r17.${step#*:} --run /data/mw_finetune/eval/${R}c1
EOF
done
cd $O && setsid nohup bash ff/chain_ffk.run.sh $NEW "\[ffk-$OLD\] queue done" ${R}c1 "$@" > ff/chain_$NEW.out 2>&1 < /dev/null &
echo "chain $NEW started: ${R}c1 $*"
