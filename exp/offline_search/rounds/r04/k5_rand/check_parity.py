import json
import os
from pathlib import Path
import subprocess
import sys
BASE=Path(__file__).resolve().parent
prefix=['taskset','-c','26-29,70-73','env','OMP_NUM_THREADS=1','OPENBLAS_NUM_THREADS=1','MKL_NUM_THREADS=1','CUDA_VISIBLE_DEVICES=','PYTHONDONTWRITEBYTECODE=1','PYTHONPATH=.:src',sys.executable]
reports=[]
for mode in ('pure','mixed','r4'):
    out=Path('/tmp/k5_parity')/mode
    args=['--cell','pi05_spatial_cache','--root','/home/weiland/trace_runs/offline_search_store',
          '--yaml','exp/trace_dual/config/tr_pi05_sp_cache.yaml','--method','exp.offline_search.closed_loop.probe:'+('ProbeB0' if mode=='pure' else 'ProbeForce'), '--episodes','2','--out',str(out)]
    if mode!='pure': args+=['--judge','guard_only']
    for variant in ('before',sys.argv[1]):
        command=prefix+[str(BASE/'launch_test.py'),variant,'parity_r4' if mode=='r4' else 'parity']
        with (BASE/'results'/f'parity_{mode}_{variant}.log').open('w') as log:
            subprocess.run(command,env={**os.environ,'K5_TEST_ARGS':json.dumps(args)},stdout=log,stderr=subprocess.STDOUT,check=True)
        data=(out/'decisions_selftest.jsonl').read_bytes()
        (BASE/'results'/f'parity_{mode}_{variant}.jsonl').write_bytes(data)
        if variant=='before': old=data
        else:
            assert old==data, (mode,len(old),len(data))
            reports.append(dict(mode=mode,bytes=len(data),rows=len(data.splitlines()),byte_identical=True))
(BASE/'results'/f'parity_{sys.argv[1]}.json').write_text(json.dumps(reports,indent=2))
print(json.dumps(reports))
