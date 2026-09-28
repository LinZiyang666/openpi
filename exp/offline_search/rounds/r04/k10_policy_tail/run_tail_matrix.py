import json
from pathlib import Path
import subprocess
import sys
B=Path(__file__).resolve().parent
P=['taskset','-c','26-29,70-73','env','OMP_NUM_THREADS=1','OPENBLAS_NUM_THREADS=1','MKL_NUM_THREADS=1','CUDA_VISIBLE_DEVICES=','PYTHONDONTWRITEBYTECODE=1','PYTHONPATH=.:src',sys.executable]
source=sys.argv[1];reports=[]
for r in json.loads((B/'arms_k10.json').read_text()):
    for judge in ('guard_only','threshold:inf','periodic:5'):
        kw={**r['kwargs'],'ncal':64}
        if judge=='threshold:inf':kw['guards']=False
        suite=r['suite'];tag=r['name']+'_'+judge.replace(':','_');out=f'/tmp/k10_{source}_matrix/{tag}'
        cmd=P+[str(B/'launch_test.py'),source,'normal','--blind','--policy-tail','--cell',f'pi05_{suite}_cache','--yaml',f"exp/trace_dual/config/tr_pi05_{'sp' if suite=='spatial' else 'l10'}_cache.yaml",'--root','/home/weiland/trace_runs/offline_search_store','--method',r['method'],'--kwargs',json.dumps(kw),'--judge',judge,'--judge-step0','miss','--out',out]
        with (B/'results'/source/f'{tag}.log').open('w') as f:subprocess.run(cmd,stdout=f,stderr=subprocess.STDOUT,check=True)
        report=json.loads((Path(out)/'selftest_report.json').read_text());assert report['PASS']
        if judge=='threshold:inf':assert report['policy_tail']==24 and report['miss']==24
        reports.append(dict(tag=tag,command=cmd,**report))
        print(tag,report,flush=True)
        (B/'results'/source/'tail_matrix.json').write_text(json.dumps(reports,indent=2))
