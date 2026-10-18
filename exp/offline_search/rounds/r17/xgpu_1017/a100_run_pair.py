"""Two isolated cold starts on the dedicated A100, each with a CPU summary.

No transfer or cleanup. Defaults to a dry plan; only --run may use the GPU.
"""
import argparse
import json
import os
from pathlib import Path
import subprocess
from a100_stage import WORK,audit


if __name__=='__main__':
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--model',required=True,choices=('dp','pi05','groot','fastwam','cosmos'))
    p.add_argument('--deployment',default='canonical',choices=('canonical','fw4090_reduced','longseg_recal1018'))
    p.add_argument('--output-name',required=True);p.add_argument('--port',type=int,default=29800)
    p.add_argument('--replicas',type=int);p.add_argument('--memory-gib',type=float);p.add_argument('--free-gib',type=float)
    p.add_argument('--minutes',type=int,default=45);p.add_argument('--cold-starts',type=int,choices=(1,2),default=2)
    p.add_argument('--numerics',action='append',default=[]);p.add_argument('--run',action='store_true')
    a=p.parse_args()
    if Path(a.output_name).name!=a.output_name:raise ValueError('one owned output basename required')
    argv=[str(WORK/'env/bin/python'),str(WORK/'tools/run_concurrent.py'),'--host','a100',
        '--model',a.model,'--deployment',a.deployment,'--capacity-arm','--batch-telemetry',
        '--minutes',str(a.minutes),'--port',str(a.port)]
    for flag,value in (('--replicas',a.replicas),('--memory-gib',a.memory_gib),('--free-gib',a.free_gib)):
        if value is not None:argv += [flag,str(value)]
    for option in a.numerics:argv += ['--numerics',option]
    print(json.dumps(dict(argv=argv,cold_starts=a.cold_starts,run=a.run)),flush=True)
    if a.run:
        names=subprocess.check_output(['nvidia-smi','--query-gpu=name','--format=csv,noheader'],text=True)
        if 'A100' not in names or 'H100' in names:raise ValueError('dedicated A100 only')
        for repeat in range(a.cold_starts):
            root=WORK/'results'/(a.output_name+('_cold2' if repeat else ''))
            current=argv+['--output',str(root),'--run']
            audit('start_concurrent',cold_start=repeat,argv=current)
            result=subprocess.run(current)
            audit('concurrent_finished',model=a.model,cold_start=repeat,returncode=result.returncode)
            if (root/'RUN.json').exists():
                subprocess.run([str(WORK/'env/bin/python'),str(WORK/'tools/a100_summarize.py'),str(root),
                    '--absolute-budget','0.000345707' if a.model=='dp' else '0.001'],
                    env=dict(os.environ,CUDA_VISIBLE_DEVICES=''),check=True)
            if result.returncode:raise SystemExit(result.returncode)
