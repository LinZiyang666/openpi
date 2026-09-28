"""Aggregate Q5 shadow fields from coordinator decision JSONL; never change an action."""
import argparse,json
from collections import defaultdict
from pathlib import Path
import numpy as np

def summarize(paths):
    groups=defaultdict(list)
    for path in paths:
        for line in Path(path).read_text().splitlines():
            row=json.loads(line);gpu=row.get('gpu_retrieval')
            if row.get('ev')=='dec' and gpu and gpu['mode']=='shadow':
                groups[row['tag']].append((row,gpu))
    report={}
    for tag,rows in groups.items():
        result={'decisions':len(rows),'failed_decisions':sum(not r.get('ok',False) for r,g in rows)}
        for key in ('top1_agree','top16_set_agree','top16_order_agree','chunk_agree','confidence_agree'):
            result[key]={'count':sum(bool(g[key]) for r,g in rows),'fraction':sum(bool(g[key]) for r,g in rows)/len(rows)}
        for key in ('event_ms','gpu_path_event_ms','wall_ms','cpu_query_ms','cpu_path_ms','chunk_max_abs','confidence_abs'):
            a=[g[key] for r,g in rows if g.get(key) is not None]
            if a:result[key]={'n':len(a),'p50':float(np.median(a)),'p90':float(np.percentile(a,90)),'max':float(np.max(a))}
        result['chunk_mismatch_step0']=sum(not g['chunk_agree'] and r['step']==0 for r,g in rows)
        result['chunk_mismatch_later']=sum(not g['chunk_agree'] and r['step']>0 for r,g in rows)
        report[tag]=result
    return report
if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--log-dir',required=True);p.add_argument('--out');a=p.parse_args()
    text=json.dumps(summarize(Path(a.log_dir).rglob('decisions_*.jsonl')),indent=2)
    if a.out:Path(a.out).write_text(text+'\n')
    print(text)
