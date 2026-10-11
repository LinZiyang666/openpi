"""G2 slots use the formal owned-process supervisor, with no fabricated recipe."""
from __future__ import annotations
import json
from pathlib import Path
import time
from exp.r17_integration.ops.formal_process import write_json


def run(job, directory, *, preflight=False):
    from exp.r17_integration.ops.formal_eval import main, FORMAL_ROOT
    directory=Path(directory);directory.mkdir(parents=True,exist_ok=False)
    name='Q_cosmos_'+job['phase']+'_'+time.strftime('%Y%m%d_%H%M%S')+'_'+str(__import__('os').getpid())
    args=['--model','cosmos_libero','--server-node',job['host'],'--worker-node',job['worker_node'],
          '--checkpoint',job['checkpoint'],'--run-name',name,'--replicas','1','--workers','2',
          '--memory-budget-gib','10','--seconds','600','--episode-seconds','240','--retries','0',
          '--prepare-only' if preflight else '--run']
    if job['phase']=='native':
        args+=['--native-parity-only']
    else:
        args+=['--smoke','--smoke-episodes','2']
    try:
        code=main(args)
        report=dict(status='PASS' if code==0 else 'FAIL',slot=job['slot'],formal_run=str(FORMAL_ROOT/name),
                    cpu_only=preflight, gpu_executed=not preflight,
                    qualification='native parity' if job['phase']=='native' else 'formal proxy/conductor smoke',
                    unsupported=['recipes','trace/snapshots','warm resume/reset','CP2','B>1','split'],exit_code=code)
    except Exception as exc:
        report=dict(status='FAIL',slot=job['slot'],formal_run=str(FORMAL_ROOT/name),error=repr(exc),cpu_only=preflight)
    write_json(directory/'receipt.json',report)
    print(json.dumps(report))
    return 0 if report['status']=='PASS' else 1
