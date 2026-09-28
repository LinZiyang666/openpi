import json
import os
from pathlib import Path
import subprocess
import sys
BASE=Path(__file__).resolve().parent
prefix=['taskset','-c','26-29,70-73','env','OMP_NUM_THREADS=1','OPENBLAS_NUM_THREADS=1','MKL_NUM_THREADS=1','CUDA_VISIBLE_DEVICES=','PYTHONDONTWRITEBYTECODE=1','PYTHONPATH=.:src',sys.executable]
reports=[]
source=sys.argv[1]
for mode in ('pure','mixed','r4','blind','rand1','rand2'):
    out=Path('/tmp/k10_parity')/source/mode
    args=['--cell','pi05_spatial_cache','--root','/home/weiland/trace_runs/offline_search_store','--yaml','exp/trace_dual/config/tr_pi05_sp_cache.yaml','--method','exp.offline_search.closed_loop.probe:'+('ProbeB0' if mode=='pure' else 'ProbeBlind' if mode=='blind' else 'ProbeForce'),'--episodes','2','--out',str(out)]
    if mode!='pure':args+=['--judge','guard_only']
    if mode=='blind':args+=['--blind']
    if mode.startswith('rand'):
        arm=json.loads((BASE/'arms_rand.json').read_text())[int(mode[-1])-1]
        args[args.index('--cell')+1]='pi05_l10_cache';args[args.index('--yaml')+1]='exp/trace_dual/config/tr_pi05_l10_cache.yaml'
        args[args.index('--method')+1]=arm['method']
        args+=['--kwargs',json.dumps(arm['kwargs']),'--fit-artifact',arm['_reuse_fit'],'--rand-seed','20260927','--rand-replicate',mode[-1]]
    for variant in ('before',source):
        out.mkdir(parents=True,exist_ok=True)
        # Blind driver appends. Archive outputs before removing only its log files.
        for f in out.glob('decisions_*.jsonl'):f.unlink()
        command=prefix+[str(BASE/'launch_test.py'),variant,'parity_r4' if mode=='r4' else 'parity']
        with (BASE/'results'/source/f'parity_{mode}_{variant}.log').open('w') as log:
            subprocess.run(command,env={**os.environ,'K10_TEST_ARGS':json.dumps(args)},stdout=log,stderr=subprocess.STDOUT,check=True)
        data=next(out.glob('decisions_*.jsonl')).read_bytes()
        (BASE/'results'/source/f'parity_{mode}_{variant}.jsonl').write_bytes(data)
        if variant=='before':old=data
        else:
            assert old==data,(mode,len(old),len(data))
            reports.append(dict(mode=mode,bytes=len(data),rows=len(data.splitlines()),byte_identical=True))
    print(reports[-1],flush=True)
(BASE/'results'/source/'parity.json').write_text(json.dumps(reports,indent=2))
