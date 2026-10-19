#!/bin/bash
# Emit one line whenever the A100 queue/run state of MODEL changes; exit on COMPLETE/FAILED.
M=$1; prev=""
while true; do
  s=$(tether exec --timeout 40s a100 -- python3 -c "
import json,glob,os
q='/home/exouser/xgpu_1017/cl18/queue/$M.json'
st=json.load(open(q)) if os.path.exists(q) else None
runs=[]
for r in sorted(glob.glob('/home/exouser/xgpu_1017/cl18/runs/${M}_*/RUN.json')):
    if 'startup_failed' in r: continue
    x=json.load(open(r)); n=0
    j=os.path.join(os.path.dirname(r),'driver/journal.jsonl')
    if os.path.exists(j): n=sum(1 for l in open(j) if '\"accepted\": true' in l)
    runs.append('%s:%s:%d'%(os.path.basename(os.path.dirname(r)).replace('${M}_',''),x['status'],n//10*10))
print((st or {}).get('status'),' '.join(runs))
" 2>/dev/null || echo "poll_error")
  if [ "$s" != "$prev" ]; then echo "$M $s"; prev="$s"; fi
  case "$s" in COMPLETE*|FAILED*) exit 0;; esac
  sleep 60
done
