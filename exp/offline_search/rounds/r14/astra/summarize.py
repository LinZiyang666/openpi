from common import *
fence()
import numpy as np

def csvread(p):
    with Path(p).open() as f:return list(csv.DictReader(f))

parts=[];dev=[]
for cell in CELLS:
    for prefix,out in [('partition_',parts),('dev_partition_',dev)]:
        p=HERE/f'{prefix}{cell}.csv'
        if not p.exists():continue
        rows=csvread(p)
        for variant in dict.fromkeys(r['variant'] for r in rows):
            r=[x for x in rows if x['variant']==variant]
            keys=['mse_raw','mse_corrected','grip_error','top1_same_task','donor_task_mass']
            keys+=['mse_blend05','gate_mean'] if out is parts else ['frozen_mse','guard_disagreement','proposed_guard_rate','recorded_guard_rate']
            val={k:float(np.mean([float(x[k]) for x in r if x.get(k) not in (None,'')])) if any(x.get(k) not in (None,'') for x in r) else None for k in keys}
            out.append(dict(cell=cell,variant=variant,episodes=len(r),**val))
dump(HERE/'partition_summary.json',parts);dump(HERE/'dev_partition_summary.json',dev)
checks=read(HERE/'allocation_checks.json');robust=read(HERE/'allocation_robustness.json')
rows=[]
for cell in CELLS:
    for dose in (.32,.40):
        r=[x for x in checks if x['cell']==cell and x['target']==dose]
        b=[x for x in robust if x['cell']==cell and x['target']==dose]
        rows.append(dict(cell=cell,target=dose,checks=len(r),
            allocation_budget_pass=sum(abs(x['allocation_ir']-dose)<=.02 for x in r),
            reference_budget_pass=sum(abs(x['reference_ir']-dose)<=.02 for x in r),
            paired_pass=sum(abs(x['delta_ir'])<=.01 for x in r),
            max_abs_delta=max(abs(x['delta_ir']) for x in r),
            max_abs_budget_error=max(abs(x['allocation_ir']-dose) for x in r),
            min_engagement=min(x['placement_change'] for x in r),
            library_capture_delta=next(x['capture_delta'] for x in r if x['source']=='library_identity_holdout'),
            bootstrap_abs_delta_p90=float(np.quantile([abs(x['delta_ir']) for x in b],.9))))
dump(HERE/'allocation_summary.json',rows)
print(table(['cell','dose','budget pass','match pass','max pair IR difference','min changed placements'],[
    [r['cell'],r['target'],f"{r['allocation_budget_pass']}/{r['checks']}",f"{r['paired_pass']}/{r['checks']}",f"{r['max_abs_delta']:.4f}",f"{r['min_engagement']:.3f}"] for r in rows]))
print(table(['cell','scope Δloss','metric Δloss','head Δloss','all Δloss'],[
  [cell]+[f"{100*(next(r['mse_corrected'] for r in parts if r['cell']==cell and r['variant']==v)/next(r['mse_corrected'] for r in parts if r['cell']==cell and r['variant']=='base')-1):+.1f}%" for v in ('scope','metric','head','all_pool')] for cell in CELLS if any(r['cell']==cell for r in parts)]))
provenance('summary')
