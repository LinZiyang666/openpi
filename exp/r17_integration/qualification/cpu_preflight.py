"""Run every prepared slot with GPU hidden and explicit --preflight only."""
import argparse
import json
import os
from pathlib import Path
import subprocess
import time
from .assets import write_json


def main():
    p=argparse.ArgumentParser();p.add_argument('--output',required=True);a=p.parse_args()
    assert os.environ.get('CUDA_VISIBLE_DEVICES')==''
    q=Path('/home/weiland/projects/openpi/exp/r17_integration/qualification')
    out=Path(a.output);out.mkdir(parents=True,exist_ok=False)
    catalog=json.loads((q/'SLOTS.json').read_text())['slots']
    results=[]
    for job in catalog:
        name=job['slot'];run=out/name
        check=subprocess.run(['bash',str(q/(name+'.sh')),'--preflight','--run',str(run)],capture_output=True,text=True,timeout=250)
        (out/(name+'.log')).write_text(check.stdout+check.stderr)
        receipt=json.loads((run/'receipt.json').read_text())
        actual=receipt.get('remote_receipt',receipt)
        row=dict(slot=name,host=job['host'],status=receipt['status'],exit_code=check.returncode,
            gpu_executed=receipt['gpu_executed'],receipt=str(run/'receipt.json'),future_checkpoint=bool(job.get('future_checkpoint')),
            disabled_batch=job['phase']=='batch_disabled')
        row['reason']=actual.get('reason',receipt.get('reason'))
        row['dependency_blocked']=job['model']=='dp' and job['phase']=='native' and row['status']=='PARTIAL' and (row['reason'] or '').startswith('DP probe pinned descriptor/paired-input API pending Q_to_P4;')
        assert not row['gpu_executed'],'CPU preflight must not run any GPU workload'
        results.append(row)
        write_json(q/'receipts/CPU_SLOTS.json',dict(status='RUNNING',results=results,cpu_only=True))
        print(name+' '+row['status'],flush=True)
    handled=all(r['status']=='PASS' and r['exit_code']==0 or r['status']=='PARTIAL' and r['exit_code']==2 and
        (r['disabled_batch'] and 'B>1 is disabled' in (r['reason'] or '') or
         r['future_checkpoint'] and (r['reason'] or '').startswith('future immutable checkpoint not yet published:') or r['dependency_blocked'])
        for r in results)
    status='FAIL' if not handled else ('PARTIAL' if any(r['dependency_blocked'] for r in results) else 'PASS')
    write_json(q/'receipts/CPU_SLOTS.json',dict(status=status,results=results,
        cpu_only=True,actual_gpu_executed=False,unpublished_future_slots=sum((r['reason'] or '').startswith('future immutable checkpoint not yet published:') for r in results),
        retired_batch_slots=sum(r['disabled_batch'] for r in results),dependency_blocked_slots=sum(r['dependency_blocked'] for r in results)))
    return 0 if status=='PASS' else 2 if status=='PARTIAL' else 1


if __name__=='__main__': raise SystemExit(main())
