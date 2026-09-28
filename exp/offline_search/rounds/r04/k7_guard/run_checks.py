"""Run owned checks with explicit CPU affinity; no servers or simulation."""
import argparse
import concurrent.futures
import json
from pathlib import Path
import subprocess
from exp.offline_search.rounds.r04.k7_guard.prepare import PREFIX,HERE,SPEC

p=argparse.ArgumentParser();p.add_argument('mode',choices=['parity','rates','smoke','plugin','edges']);p.add_argument('--tag',default='final');a=p.parse_args()
out=HERE/'results'/a.tag;out.mkdir(parents=True,exist_ok=True)
commands=[]
if a.mode in ('parity','rates'):
    for key in ('pi05_l10','pi05_spatial'):
        for scale in (50,500):
            commands.append((f'{a.mode}_{key}_{scale}',PREFIX+['-m','exp.offline_search.rounds.r04.k7_guard.evidence',a.mode,'--key',key,'--scale',str(scale)]))
elif a.mode=='smoke':
    for suite in ('l10','spatial'):
        for scale in (50,500):
            for arm in ('inf','cache'):
                tag=f'smoke_pi05_{suite}_{arm}_{scale}'
                kw=dict(base_kwargs=dict(lib='current' if scale==50 else 'big',kref=5 if scale==50 else 8,budget=0),stuck_guard='vision_confirmed',progress_guard='noprog_span',events='none')
                commands.append((tag,PREFIX+['-m','exp.offline_search.harness.smoke','--method',SPEC,'--kwargs',json.dumps(kw),'--cell',f'pi05_{suite}_{arm}','--episodes','2','--root','/home/weiland/trace_runs/offline_search_store','--no-ref','--out',str(out/tag)]))
elif a.mode=='plugin':
    for row in json.loads((HERE/'arms_k7.json').read_text()):
        tag='plugin_'+row['name']; suite=row['suite']
        commands.append((tag,PREFIX+['-m','exp.offline_search.closed_loop.selftest','--blind',
            '--cell',f'pi05_{suite}_cache','--yaml',f"exp/trace_dual/config/tr_pi05_{'sp' if suite=='spatial' else suite}_cache.yaml",
            '--method',SPEC,'--kwargs',json.dumps(row['kwargs']),'--fit-artifact',f"/tmp/k7_guard_fits/{row['name']}.pkl",
            '--judge','guard_only','--episodes','2','--root','/home/weiland/trace_runs/offline_search_store','--out',str(out/tag)]))
else:
    commands=[('edges',PREFIX+['-m','exp.offline_search.rounds.r04.k7_guard.edge_checks'])]
def run(item):
    tag,cmd=item
    with (out/(tag+'.log')).open('w') as f:
        r=subprocess.run(cmd,stdout=f,stderr=subprocess.STDOUT)
    result=dict(tag=tag,returncode=r.returncode,command=cmd)
    print(json.dumps(result),flush=True);return result
with concurrent.futures.ThreadPoolExecutor(max_workers=2) as pool:
    results=list(pool.map(run,commands))
(out/(a.mode+'_commands.json')).write_text(json.dumps(results,indent=2))
assert all(r['returncode']==0 for r in results),results
