#!/usr/bin/env bash
# One-line-per-item health probe for the RIT LOTO line on weilandserver (cron / L3).
# Prints PROBE lines only; never modifies anything.
export HOME=/home/weiland
R=/data/libero_cache/rit_loto
ts=$(date '+%m-%d %H:%M')
echo "PROBE $ts tmux: $(tmux ls 2>/dev/null | grep -oE '^(loto[a-z]*_[a-z0-9_]+|srv[0-9]+)' | tr '\n' ' ')"
echo "PROBE $ts gpu: $(nvidia-smi --query-gpu=memory.used,utilization.gpu --format=csv,noheader | tr '\n' ' ')"
for s in libero_spatial libero_10; do
  d=$R/$s
  nf=$( [ -f $d/noise_floor.json ] && python3 -c "import json;print(json.load(open('$d/noise_floor.json'))['d_ref1_ref2']['n'])" 2>/dev/null || echo -)
  pg=$( [ -f $d/parity_gate.json ] && python3 -c "import json;print(json.load(open('$d/parity_gate.json'))['status'])" 2>/dev/null || echo -)
  rows=$( [ -f $d/loto_table.jsonl ] && wc -l < $d/loto_table.jsonl || echo -)
  rec=$( [ -f $d/loto_table.jsonl.record.json ] && echo done || echo -)
  fits=$( [ -f $d/fits.json ] && echo done || echo -)
  band=$( [ -f $d/bootstrap_band.json ] && echo done || echo -)
  echo "PROBE $ts $s: noise_floor_n=$nf parity=$pg table_rows=$rows table_record=$rec fits=$fits band=$band"
done
for n in lotonf_libero_spatial lotonf_libero_10 lotopar_libero_spatial lotopar_libero_10 lototab_libero_spatial lototab_libero_10 lotofit_libero_spatial lotofit_libero_10 srv0; do
  [ -f /tmp/$n.log ] && echo "PROBE $ts $n.log tail: $(grep -v -i 'warning\|pynvml' /tmp/$n.log | tail -1 | cut -c1-160)"
done
V=$R/libero_10/verify_logs
if [ -d $V ]; then
  for tag in smoke verify; do
    [ -d $V/$tag ] && echo "PROBE $ts verify_logs/$tag: h5=$(find $V/$tag -name '*.h5' | wc -l) tmp=$(find $V/$tag -name '*.h5.tmp' | wc -l) sidecar_rows=$(cat $V/$tag/conn_*/*/decisions.jsonl 2>/dev/null | wc -l)"
  done
fi
echo "PROBE $ts frozen_run: $( [ -f $R/libero_10/frozen_run.json ] && echo present || echo -) verify_report: $( [ -f $R/libero_10/verify/verify.json ] && echo present || echo -)"
