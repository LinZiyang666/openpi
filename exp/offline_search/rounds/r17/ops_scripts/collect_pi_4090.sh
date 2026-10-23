#!/bin/bash
# π0.5 MT50 library seeds on the 4090 (local server): smoke, then seed 9, then seed 10 (collect only; finish separately).
F=/data/r17_frozen/openpi_5f026a53
M=/home/weiland/projects/openpi_ext/scratch/r17_ops/mw
unset R17_H100_NODE R17_H100_HOST R17_DRIVER_PORT R17_PUBLIC_PORT
cd $F
ARGS=(--checkpoint /data/ckpt/pi05_metaworld_rlinf --replicas 5 --replica-startup-width 5 --mps --memory-budget-gib 45
      --worker-counts timan107=40,timan1=36 --port 23160 --driver-port 23169)
for item in smoke:9 9 10; do
  [ -f $M/hold_collect_pi_4090 ] && { echo "$(date +%H:%M:%S) [collect-pi-4090] hold, stop before $item" >> $M/chain.log; break; }
  seed=${item#smoke:}; extra=(); run=w3_pi_s$seed
  [ "$item" != "$seed" ] && { extra=(--smoke); run=w3s_pi_s${seed}_smoke; }
  echo "$(date +%H:%M:%S) [collect-pi-4090] start $run" >> $M/chain.log
  bash $M/collect_seed.run.sh $F pi05_mw local $seed /data/r17_library/seed$seed $run --collect-only "${extra[@]}" "${ARGS[@]}"
  rc=$?; echo "$(date +%H:%M:%S) [collect-pi-4090] $run ended rc=$rc" >> $M/chain.log
  [ $rc -eq 0 ] || break
done
echo "$(date +%H:%M:%S) [collect-pi-4090] queue done" >> $M/chain.log
