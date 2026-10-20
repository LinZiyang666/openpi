"""Reuse the batch-one latency profiler on each lane's frozen model island.

initialize is CPU-only. prepare stages only the small profiler/input overlay
by the existing public rsync path. launch is for the coordinator on an idle
GPU; it never installs packages or recalibrates a knob/stall artifact.
"""
import argparse
import os
from pathlib import Path
import shlex
import subprocess
import time
import json,re

from .audit import HERE
from .io_budget import dump,copy,filesystem_budget
from exp.offline_search.rounds.r17.filter_failed_knob.common import read,REPO

INDEX=HERE/'PROFILES.json'

def first_row_metadata(parent):
    """Read only the arrays header and first recorded row, at most 16 MiB."""
    decoder=json.JSONDecoder();text=''
    with filesystem_budget(),(parent/'READY.json').open() as stream:
        for _ in range(256):
            block=stream.read(65536)
            if not block:break
            text+=block
            try:
                arrays=re.search(r'"arrays"\s*:\s*',text)
                row=re.search(r'"rows"\s*:\s*\[\s*',text)
                if arrays and row:
                    a,_=decoder.raw_decode(text,arrays.end());r,_=decoder.raw_decode(text,row.end())
                    if 'task_name' in r:return dict(arrays=a,rows=[r])
            except json.JSONDecodeError:pass
    raise ValueError('First observation metadata is outside the bounded READY prefix: '+str(parent))

def initialize():
    jobs=read(HERE/'CAMPAIGNS.json')['jobs'];units=[]
    for model,benchmark,gpu in [('dp','mt50','RTX 4090'),('groot','mt50','H100'),
        ('fastwam','mt50','H100'),('dp','libero','RTX 4090'),
        ('fastwam','libero','H100'),('fastwam','libero','RTX 4090'),('cosmos','libero','H100')]:
        if model=='cosmos':
            # Read-only provenance from the unaffected official-A lane. Its
            # descriptor is copied into the independent profiler bundle.
            j=dict(run='/data/mw_finetune/eval/fa_co_r_u_l')
        else:j=next(j for j in jobs if j['model']==model and j['benchmark']==benchmark and j['gpu']==gpu and j['stage']=='long')
        token={'dp':'dp','groot':'gr','fastwam':'fw','cosmos':'co'}[model]
        n=f'fa_ls_timing_{token}_{"m" if benchmark=="mt50" else "l"}_{"h" if gpu=="H100" else "4"}'
        root=HERE/'profile_bundles'/n;root.mkdir(parents=True,exist_ok=True)
        for package in ('exp','exp/r17_integration','exp/r17_integration/profile'):
            p=root/package;p.mkdir(exist_ok=True)
            (p/'__init__.py').write_text('from pkgutil import extend_path\n__path__=extend_path(__path__,__name__)\n')
        for file in ('latency.py','certify.py'):
            target=root/'exp/r17_integration/profile'/file
            if not target.exists():copy(REPO/'exp/r17_integration/profile'/file,target)
        c=read(Path(j['run'])/'campaign.json');suites=('mt50',) if benchmark=='mt50' else ('libero_10','libero_spatial')
        if not (root/'load.json').exists():copy(Path(j['run'])/'assets/load.json',root/'load.json')
        for suite in suites:
            target=root/(suite+'.npz')
            if target.exists():continue
            if suite=='mt50':copy(REPO/'exp/r17_integration/profile/inputs_mt50/assembly_seed8_idx0.npz',target)
            else:
                # One recorded development-pool observation, solely for cost
                # measurement. Read array headers + row zero, never full RGB.
                import numpy as np
                # Serving snapshots omit RGB, since retrieval needs fitted
                # codes only. Use the canonical recorded B-pool RGB row.
                parent=Path('/data/r17_library/libero')/model/suite/'bpool'
                m=first_row_metadata(parent)
                obs={}
                for key,field in [('observation/image','image_0'),('observation/wrist_image','image_1'),('observation/state','state')]:
                    entry=m['arrays'][field]
                    data=np.load(parent/entry['file'],mmap_mode='r',allow_pickle=False)
                    obs[key]=np.asarray(data[0])[None].copy()
                tasks=read(REPO/'exp/r17_integration/qualification/specs/libero_benchmark_spec.json')['tasks']
                prompt=next(lang for task,lang,_ in tasks if task==m['rows'][0]['task_name'])
                obs['prompt']=np.asarray([prompt])
                if model=='cosmos':
                    obs['observation/cosmos_proprio']=obs.pop('observation/state')
                np.savez(target,**obs)
        destination='/data/mw_finetune/qualification/'+n if gpu=='H100' else str(root)
        unit=dict(name=n,run=j['run'],bundle=str(root),destination=destination,gpu=gpu,
            model=model,benchmark=benchmark,suites=list(suites),episodes=0,shared_full_reduced=True)
        dump(root/'UNIT.json',unit);units.append(unit)
        for verb in ('prepare','launch'):
            path=HERE/'commands'/f'{verb}_{n}.sh'
            path.write_text('#!/usr/bin/env bash\nset -euo pipefail\ncd '+shlex.quote(str(REPO))+'\n'+
                shlex.quote(str(REPO/'.venv/bin/python'))+' -m exp.offline_search.rounds.r17.longseg_1017.profiles '+verb+' --name '+n+'\n');path.chmod(0o755)
    dump(INDEX,dict(units=units,artifact_parameters_changed=False))

def operate(name,verb):
    from exp.r17_integration.ops import formal_eval as f
    u=next(u for u in read(INDEX)['units'] if u['name']==name)
    c=read(Path(u['run'])/'campaign.json');remote=u['gpu']=='H100'
    if verb=='prepare':
        if remote:
            from exp.r17_integration.ops.sync_back import snapshot_run
            archive=Path(u['bundle']+'.tar')
            receipt=snapshot_run(u['bundle'],archive)
            f.remote('h100',['mkdir','-p',u['destination']],replay_safe=True)
            f.transfer(['rsync','push',str(archive),'h100:'+u['destination']+'/bundle.tar'])
            code='from exp.r17_integration.ops.sync_back import install_snapshot;install_snapshot('+repr(u['destination']+'/bundle.tar')+','+repr(u['destination'])+',expected_sha256='+repr(receipt['archive_sha256'])+')'
            f.remote('h100',['env','PYTHONPATH='+f.pythonpath(c['runtime']),c['server_python'],'-c',code],seconds=180,replay_safe=True)
        print('PREPARED small profile overlay',name);return
    env=f.server_env(c,c['runtime']);env['PYTHONPATH']=u['destination']+':'+env['PYTHONPATH']
    profile_model=u['model']+('_mw' if u['benchmark']=='mt50' else '_libero')
    for suite in (u['suites'] if u['benchmark']=='mt50' else ['libero_10']):  # one LIBERO timing (owner 10-20)
        for cached in ([False,True] if u['model']=='fastwam' and suite!='mt50' else [False]):
            directory='results_cached_text' if cached else 'results_mt50' if suite=='mt50' else 'results'
            out=u['destination']+'/'+directory
            args=[c['server_python'],'-m','exp.r17_integration.profile.certify','--model',profile_model,
                '--suite',suite,'--python',c['server_python'],'--out-dir',out,
                '--descriptor',u['destination']+'/load.json','--observations',u['destination']+'/'+suite+'.npz']
            if cached:args+=['--precompute-text']
            argv=['env',*[k+'='+str(v) for k,v in env.items()],*args]
            if remote:
                print(f.remote('h100',argv,seconds=3600))
                target=HERE/'profiles/h100'/directory;target.mkdir(parents=True,exist_ok=True)
                previous=target/(profile_model+'_'+suite+'.json')
                if previous.exists():previous.rename(previous.with_name(previous.name+'.previous_'+str(time.time_ns())))
                f.transfer(['rsync','pull','h100:'+out+'/'+profile_model+'_'+suite+'.json',str(target/(profile_model+'_'+suite+'.json'))])
            else:
                # -m puts cwd first on sys.path; from the runtime its plain exp/ package
                # shadows the bundle's extend_path one, so exp.r17_integration.profile is missing.
                subprocess.run(argv,cwd=u['destination'],check=True)
                target=HERE/'profiles/4090'/directory;target.mkdir(parents=True,exist_ok=True)
                previous=target/(profile_model+'_'+suite+'.json')
                if previous.exists():previous.rename(previous.with_name(previous.name+'.previous_'+str(time.time_ns())))
                copy(Path(out)/(profile_model+'_'+suite+'.json'),target/(profile_model+'_'+suite+'.json'))

def main():
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('verb',choices=('initialize','prepare','launch'));p.add_argument('--name');a=p.parse_args()
    initialize() if a.verb=='initialize' else operate(a.name,a.verb)
if __name__=='__main__':main()
