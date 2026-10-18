"""Serial A100 handoff: finish receipt -> summarize -> owned cleanup -> one model.

Explicitly named previous and next models. Stops on staging or model failure so
dependencies can be repaired without discarding and retransferring weights.
"""
import argparse
import json
from pathlib import Path
import subprocess
import time
from a100_stage import WORK,copy,cleanup,stage,audit


if __name__=='__main__':
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--after-run',required=True);p.add_argument('--previous-model',required=True)
    p.add_argument('--next-model',choices=['pi05','groot','fastwam','cosmos'],required=True)
    p.add_argument('--output-name',required=True);p.add_argument('--port',type=int,default=29800)
    p.add_argument('--cold-starts',type=int,default=2,choices=(1,2))
    p.add_argument('--replicas',type=int);p.add_argument('--memory-gib',type=float);p.add_argument('--free-gib',type=float)
    a=p.parse_args()
    previous=Path(a.after_run).resolve();receipt=previous/'RUN.json'
    if not previous.is_relative_to(WORK/'results') or Path(a.output_name).name!=a.output_name:
        raise ValueError('only owned run directories accepted')
    deadline=time.monotonic()+3*3600
    while time.monotonic()<deadline:
        r=json.loads(receipt.read_text()) if receipt.exists() else {}
        if r.get('status') in ('FAILED','COMPLETE') and r.get('mps_cleanup',{}).get('daemon_closed'):break
        time.sleep(10)
    else:raise TimeoutError('previous test did not close; no cleanup or new transfer')
    subprocess.run([str(WORK/'env/bin/python'),str(WORK/'tools/a100_summarize.py'),str(previous)],check=True)
    cleanup(a.previous_model,receipt)
    copy('xr_tools','tools')
    stage(a.next_model)
    # Staging can take minutes. Refresh small coordinator-owned evidence tools
    # at this explicit pre-launch boundary; run_concurrent then seals its code.
    copy('xr_tools','tools')
    copy('xc_bundles','concurrent/bundles')
    argv=[str(WORK/'env/bin/python'),str(WORK/'tools/run_concurrent.py'),'--host','a100',
        '--model',a.next_model,'--capacity-arm','--batch-telemetry','--minutes','45',
        '--output',str(WORK/'results'/a.output_name),'--port',str(a.port),'--run']
    for flag,value in (('--replicas',a.replicas),('--memory-gib',a.memory_gib),('--free-gib',a.free_gib)):
        if value is not None:argv += [flag,str(value)]
    for repeat in range(a.cold_starts):
        current=list(argv)
        if repeat:
            current[current.index('--output')+1]+='_cold2'
        audit('start_concurrent',cold_start=repeat,argv=current)
        result=subprocess.run(current)
        audit('concurrent_finished',model=a.next_model,cold_start=repeat,returncode=result.returncode)
        if result.returncode:raise SystemExit(result.returncode)
        finished=Path(current[current.index('--output')+1])
        subprocess.run([str(WORK/'env/bin/python'),str(WORK/'tools/a100_summarize.py'),str(finished)],check=True)
    raise SystemExit(0)
