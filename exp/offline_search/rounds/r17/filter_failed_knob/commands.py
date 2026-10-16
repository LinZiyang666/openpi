"""Write coordinator commands and the resource inventory; execute nothing."""
from __future__ import annotations

from pathlib import Path
import shlex

from .common import HERE, REPO, read, dump

MODULE='exp.offline_search.rounds.r17.filter_failed_knob'
PREFIX='env CUDA_VISIBLE_DEVICES= JAX_PLATFORMS=cpu PYTHONDONTWRITEBYTECODE=1 OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1 PYTHONPATH=.:src .venv/bin/python'
LAUNCH_PREFIX='env -u CUDA_VISIBLE_DEVICES -u JAX_PLATFORMS OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1 PYTHONPATH=.:src .venv/bin/python'


def script(path,lines):
    path.parent.mkdir(parents=True,exist_ok=True)
    path.write_text('#!/usr/bin/env bash\nset -euo pipefail\ncd '+shlex.quote(str(REPO))+'\n'+'\n'.join(lines)+'\n')
    path.chmod(0o755)


def main():
    jobs=read(HERE/'CAMPAIGNS.json')['jobs'];out=HERE/'commands'
    for job in jobs:
        job['launch_claim_exists']=(Path(job['run'])/'LAUNCH.json').exists()
        for action in ('prepare','launch'):
            prefix=PREFIX if action=='prepare' else LAUNCH_PREFIX
            script(out/f'{action}_{job["name"]}.sh',[f'{prefix} -m {MODULE}.{"stage" if action=="prepare" else "launch"} --run {shlex.quote(job["run"])}'])
    vla=read(HERE/'vla/runs/off/plan.json')
    for path in vla['groups']:
        group=read(path);name=Path(group['root']).name
        script(out/f'prepare_{name}.sh',[f'bash {HERE}/vla.sh verify'])
        script(out/f'launch_{name}.sh',[f'bash {HERE}/vla.sh run --group {shlex.quote(path)} --workers 64 --port 23159'])
        jobs.append(dict(name=name,model=group['model'],benchmark='libero',mode='reduced',stage='off',
                         gpu='RTX 4090',workers={'timan107':64},replicas=3 if group['model']=='pi05' else 4,
                         episodes=sum(len(e['dispatch_pairs']) for e in group['entries']),group=path,pipeline='R16 k1ref'))
    # This is a schedule, never an automatically executing chain. Node overlap
    # means the frozen workers' EGL reservations cannot be shared.
    reduced=sorted([j for j in jobs if j['mode']=='reduced'],key=lambda j:(j['stage']!='off',j['gpu'],j['name']))
    full=sorted([j for j in jobs if j['mode']=='full'],key=lambda j:(j['gpu'],j['name']))
    order=reduced+full
    dump(HERE/'LAUNCH_ORDER.json',dict(jobs=order,rule='reduced first; one campaign at a time is a valid conflict-free schedule',
                                     conflicts=[dict(a=a['name'],b=b['name'],nodes=sorted(a['workers'].keys()&b['workers'].keys()))
                                       for i,a in enumerate(order) for b in order[i+1:] if a['workers'].keys()&b['workers'].keys()]))
    for gpu in ('RTX 4090','H100'):
        for mode in ('reduced','full'):
            (out/f'{"4090" if gpu=="RTX 4090" else "h100"}_{mode}.txt').write_text(''.join(
                f'{"# ALREADY LAUNCHED; do not prepare/relaunch: " if j.get("launch_claim_exists") else ""}{j["name"]}\t{j["episodes"]}\t{j["workers"]}\n' for j in order if j['gpu']==gpu and j['mode']==mode))
    dump(HERE/'ALL_CAMPAIGNS.json',dict(jobs=jobs,initial_episodes=sum(j['episodes'] for j in jobs)))
    print(f'{len(jobs)} initial definitions / {sum(j["episodes"] for j in jobs):,} episodes')


if __name__=='__main__':main()
