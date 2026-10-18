"""Off-first LIBERO reruns, using formal_a_queue's unchanged cell rule."""
from __future__ import annotations
import argparse
from pathlib import Path
import shlex

from .audit import HERE,EVAL
from .build import clone,commands,COMMANDS
from .io_budget import dump
from exp.offline_search.rounds.r17.filter_failed_knob.common import read,REPO
from exp.offline_search.rounds.r17.filter_failed_knob import formal_a_queue as q

INDEX=HERE/'FORMAL_QUEUE.json'
TOKENS={'dp':'dp','fastwam':'fw'}
def name(model,mode,arm,stage):return f'fa_ls_{TOKENS[model]}_{mode[0]}_{arm[0] if arm=="filtered" else "u"}_{stage[0]}_l'
def put(u):
    d=read(INDEX) if INDEX.exists() else dict(units=[])
    d['units']=[r for r in d['units'] if r['name']!=u['name']]+[u];dump(INDEX,d)
def selected(u):return next(r for r in read(INDEX)['units'] if r['name']==u)

def initialize():
    templates=read(q.DRAFTS)['jobs']
    previous=read(q.INDEX)['units']
    for model in TOKENS:
        for mode in ('reduced','full'):
            for arm in ('original','filtered'):
                old=next(j for j in templates if (j['model'],j['mode'],j['arm'])==(model,mode,arm))
                prior=next(u for u in previous if (u['model'],u['mode'],u['arm'],u['stage'])==(model,mode,arm,'off' if mode=='full' else 'levels'))
                count=sum(prior['workers'].values())
                workers={'timan1':count//2,'timan107':count-count//2}
                common=dict(model=model,mode=mode,arm=arm,source=old['run'],workers=workers,gpu=old['gpu'],
                    replicas=old['replicas'],batch=old['batch'],pipeline='R17')
                n=name(model,mode,arm,'off')
                existing={u['name']:u for u in read(INDEX)['units']} if INDEX.exists() else {}
                if n in existing and name(model,mode,arm,'levels') in existing:continue
                job=clone(old['run'],n,workers=workers,mode=mode,stage='off',kind='off')
                job.update(arm=arm);dump(Path(job['run'])/'FFK_DEFINITION.json',job)
                put(dict(common,name=n,stage='off',status='BUILT',episodes=job['episodes'],jobs=[job]))
                n=name(model,mode,arm,'levels')
                put(dict(common,name=n,stage='levels',status='WAITING_OFF',episodes=None,maximum_episodes=30000,jobs=[]))
                for verb in ('prepare','launch'):
                    path=COMMANDS/f'{verb}_{n}.sh'
                    path.write_text('#!/usr/bin/env bash\nset -euo pipefail\ncd '+shlex.quote(str(REPO))+'\n'+
                        shlex.quote(str(REPO/'.venv/bin/python'))+' -m exp.offline_search.rounds.r17.longseg_1017.workflow '+
                        verb+' --name '+n+'\n');path.chmod(0o755)

def _build_levels(n):
    from .analyze import reference
    from exp.offline_search.rounds.r17.filter_failed_knob import formal_a_analysis as a
    u=selected(n)
    if u['status']!='WAITING_OFF':return u
    off=selected(name(u['model'],u['mode'],u['arm'],'off'))
    data=a.load_r17(off['jobs'][0]);pure=reference(u['model'],'libero',u['mode'],u['gpu'])
    rows=[dict(suite=s,size=size,**q.classify(arms.get('koff',{}),pure.get(s,{}))) for (s,size),arms in data.items()]
    dump(HERE/'off_decisions'/f'{n}.json',dict(rows=rows,dev_outcomes_used=False))
    if len(rows)!=12 or any(r['status']=='pending' for r in rows):raise ValueError('Wait for all 12 complete new OFF/pure cells')
    keep={(r['suite'],r['size']) for r in rows if r['status']=='tune'}
    seal=read(Path(u['source'])/'assets/CACHE_STUDY.json')
    groups=[r for r in seal['groups'] if (r['suite'],r['library_size']) in keep and not r['group'].endswith(('_koff','_pure'))]
    if groups:
        j=clone(u['source'],n,workers=u['workers'],groups=groups,mode=u['mode'],stage='levels')
        j.update(arm=u['arm']);dump(Path(j['run'])/'FFK_DEFINITION.json',j)
        u.update(jobs=[j],status='BUILT',episodes=j['episodes'])
    else:u.update(jobs=[],status='SKIP_OFF_SUFFICIENT',episodes=0)
    u['selected_cells']=sorted(keep);put(u);return u

def build_levels(n):
    from .io_budget import filesystem_budget
    with filesystem_budget():return _build_levels(n)

def dispatch(n,verb):
    u=build_levels(n)
    if not u['jobs']:print('SKIP_OFF_SUFFICIENT',n);return
    assert len(u['jobs'])==1;root=Path(u['jobs'][0]['run'])
    if verb=='prepare':
        from exp.offline_search.rounds.r17.filter_failed.stage import stage
        stage(root)
    else:
        from exp.offline_search.rounds.r17.filter_failed_knob.launch import main
        main(root)

def main():
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('action',choices=('initialize','build-levels','prepare','launch'))
    p.add_argument('--name');a=p.parse_args()
    if a.action=='initialize':initialize()
    elif a.action=='build-levels':build_levels(a.name)
    else:dispatch(a.name,a.action)
if __name__=='__main__':main()
