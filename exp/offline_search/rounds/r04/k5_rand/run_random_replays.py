import concurrent.futures
import json
from pathlib import Path
import subprocess
import sys
BASE=Path(__file__).resolve().parent
source=sys.argv[1]
prefix=['taskset','-c','26-29,70-73','env','OMP_NUM_THREADS=1','OPENBLAS_NUM_THREADS=1','MKL_NUM_THREADS=1','CUDA_VISIBLE_DEVICES=','PYTHONDONTWRITEBYTECODE=1','PYTHONPATH=.:src',sys.executable]
commands=[]
for arm in json.loads((BASE/'arms_rand.json').read_text()):
    name=arm['name'];rep=name[-1]
    args=['--cell','pi05_l10_cache','--yaml','exp/trace_dual/config/tr_pi05_l10_cache.yaml','--root','/home/weiland/trace_runs/offline_search_store','--method',arm['method'],'--kwargs',json.dumps(arm['kwargs']),'--fit-artifact',arm['_reuse_fit'],'--judge','guard_only','--rand-seed','20260927','--rand-replicate',rep,'--episodes','4','--out',f'/tmp/k5_{source}_replays/{name}']
    commands.append((name,prefix+[str(BASE/'launch_test.py'),source,'normal',*args]))
def run(item):
    name,cmd=item
    with (BASE/'results'/f'{source}_{name}.log').open('w') as log: subprocess.run(cmd,stdout=log,stderr=subprocess.STDOUT,check=True)
    report=json.loads(Path(f'/tmp/k5_{source}_replays/{name}/selftest_report.json').read_text())
    (BASE/'results'/f'{source}_{name}.json').write_text(json.dumps(report,indent=2))
    return name,report['decisions'],report['PASS']
(BASE/'results'/f'replay_commands_{source}.json').write_text(json.dumps(commands,indent=2))
with concurrent.futures.ThreadPoolExecutor(max_workers=2) as pool:
    for result in pool.map(run,commands): print(result,flush=True)
