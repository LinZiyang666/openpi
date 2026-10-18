"""Bounded, portable summary of a finished concurrent run (CPU only)."""
import argparse
import json
from pathlib import Path
import subprocess
from artifacts import write_json
from concurrent_compare import compare


def summarize(root,budget):
    root=Path(root)
    run=json.loads((root/'RUN.json').read_text())
    report={k:run.get(k) for k in ('status','error','pin','input_identity','gpu','clients',
        'rounds','deliveries','memory','live_reference','mps_cleanup','harness_files')}
    report.update(path=str(root),campaign={k:run['campaign'].get(k) for k in
        ('model','replicas','memory_budget_gib','stage_batching','dp_batching','fastwam_batching',
         'xgpu_probe','xgpu_numerics','xgpu_batch_telemetry')})
    if run['status']=='COMPLETE':
        result=compare(root,root,budget)
        report['self_comparison']={k:v for k,v in result.items() if k!='rows'}
        groups={}
        for row in result['rows']:
            g=groups.setdefault(row['key'][0],dict(requests=0,max_abs=0.,relative_l2=0.,outside_budget=0))
            g['requests']+=1
            g['max_abs']=max(g['max_abs'],row['cross']['max_abs'])
            g['relative_l2']=max(g['relative_l2'],row['cross']['relative_l2'])
            g['outside_budget']+=not row['within_budget']
        report['groups']=groups
        write_json(root/'SELF_COMPARISON.json',result)
    else:
        report['log_tails']={p.name:p.read_text()[-3000:] for p in root.glob('*.log')}
    report['run_bytes']=int(subprocess.check_output(['du','-sb',str(root)]).split()[0])
    report['study_bytes']=int(subprocess.check_output(['du','-sb',str(root.parent.parent)]).split()[0])
    write_json(root/'SUMMARY.json',report)
    print(json.dumps({k:report.get(k) for k in ('path','status','error','groups','run_bytes','study_bytes')}))
    return report


if __name__=='__main__':
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('run')
    p.add_argument('--absolute-budget',type=float,default=.001)
    a=p.parse_args();summarize(a.run,a.absolute_budget)
