#!/bin/bash
# For each run verified with checksum_diff_files=0 in pullback.log, delete its H100 servers/ copy (rule: pull back, verify, then delete remote).
L=${1:-/home/weiland/.claude/jobs/35f2869b/tmp/pullback.log}; C=/home/weiland/projects/openpi_ext/scratch/r17_ops/mw/chain.log; done_runs=""
until grep -q " done$" $L && [ "$(grep -c 'verified ' $L)" = "$(echo $done_runs | wc -w)" ]; do
  for r in $(grep "verified .* checksum_diff_files=0 " $L | awk '{print $3}'); do
    case " $done_runs " in *" $r "*) continue;; esac
    timeout 300 tether exec --timeout 290s h100 -- bash -lc "rm -rf /data/mw_finetune/eval/$r/servers && df -h /data | tail -1" > /tmp/pf_$r.out 2>&1
    echo "$(date +%T) [pullback] H100 freed $r ($(tail -1 /tmp/pf_$r.out | awk '{print $4}') free); local copy /nvme/weiland/openpi/h100_pullback_1024/$r" >> $C
    done_runs="$done_runs $r"
  done
  sleep 30
done
