"""Read-only serving/reference and measured-rate inventory from terminal journals."""
from collections import Counter
import time
import json
from pathlib import Path
import yaml
from .audit import EVAL,HERE,loads,io_ready
from .io_budget import reader

STEPS={('groot','metaworld'):(4,2),('dp','metaworld'):(100,10),('fastwam','metaworld'):(10,3),
       ('dp','libero'):(100,10),('fastwam','libero'):(10,1)}


def journal(root):
    path=root/'driver/journal.jsonl';data={};begin=end=None;durations=0.
    if not path.is_file():return dict(accepted=0)
    count=0;start=time.monotonic()
    with reader(),path.open('rb') as stream:
        for line in stream:
            count+=len(line)
            if count>=1024*1024:
                io_ready();time.sleep(max(0,count/2**22-(time.monotonic()-start)))
                count=0;start=time.monotonic()
            try:r=loads(line)
            except ValueError:continue
            if r.get('accepted') is not True or r.get('status') not in ('done','failed'):continue
            if r.get('error') is not None:continue
            data[r['task_uid']]=bool(r.get('success'))
            if isinstance(r.get('ts'),(int,float)):
                begin=min(begin,r['ts']) if begin is not None else r['ts']
                end=max(end,r['ts']) if end is not None else r['ts']
            durations+=r.get('duration_s',0) or 0
    span=end-begin if begin is not None and end>begin else None
    return dict(accepted=len(data),successes=sum(data.values()),first_ts=begin,last_ts=end,
        span_seconds=span,episodes_per_minute=60*len(data)/span if span else None,duration_sum=durations)


def inspect_run(path):
    c=json.loads(path.read_text());root=path.parent
    if (c.get('model'),c.get('domain')) not in STEPS:return None
    full=root/'assets/full.yaml'
    if not full.is_file():return None
    doc=yaml.safe_load(full.read_text());binding=doc.get('adapter_binding',{})
    k=binding.get('num_steps');mode=next((m for m,n in zip(('full','reduced'),STEPS[c['model'],c['domain']]) if k==n),None)
    counts=journal(root)
    return dict(run=root.name,model=c['model'],domain=c['domain'],mode=mode,steps=k,
        gpu='H100' if c['server_node']=='h100' else 'RTX 4090',replicas=c['replicas'],
        batch=(c.get('dp_batching') or c.get('fastwam_batching') or {}).get('max_batch',1),mps=c.get('mps',False),
        worker_nodes=c.get('execution_worker_nodes',c.get('worker_nodes')),runtime=c.get('runtime'),
        server_python=c.get('server_python'),commit=binding.get('commit_horizon'),
        pure=not c.get('cache_study') and not c.get('library') and not doc.get('recipe',{}).get('enabled',False),
        smoke=c.get('smoke',False),**counts)


def main():
    paths=sorted(EVAL.glob('*/campaign.json'))
    rows=[]
    for path in paths:
        io_ready();row=inspect_run(path)
        if row:rows.append(row)
        time.sleep(3)
    refs=[]
    for (model,domain),steps in STEPS.items():
        for mode,k in zip(('full','reduced'),steps):
            for gpu in ('H100','RTX 4090'):
                expected=2500 if domain=='metaworld' else 1000
                candidates=[r for r in rows if (r['model'],r['domain'],r['steps'],r['gpu'])==(model,domain,k,gpu)
                            and r['pure'] and not r['smoke'] and r['accepted']>=expected]
                refs.append(dict(model=model,domain=domain,mode=mode,gpu=gpu,episodes_needed=expected,
                                 references=candidates,missing=not candidates))
    (HERE/'INVENTORY.json').write_text(json.dumps(dict(runs=rows,references=refs),indent=2)+'\n')
    print('runs',len(rows),'reference cells',len(refs),'missing',sum(r['missing'] for r in refs),flush=True)


if __name__=='__main__':main()
