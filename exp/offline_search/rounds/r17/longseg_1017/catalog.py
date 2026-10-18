"""Build the audited logical lanes and new OFF-first formal-A definitions."""
from __future__ import annotations
import argparse
import copy
import json
from pathlib import Path

from .audit import HERE,EVAL
from .build import clone
from . import io_budget as budget
from exp.offline_search.rounds.r17.filter_failed_knob.common import read,MT

TOK={'dp':'dp','groot':'gr','fastwam':'fw'}
def workers(src):
    c=read(Path(src)/'campaign.json');n=c.get('execution_workers',c['workers'])
    return dict(timan1=n//2,timan107=n-n//2)

def local_dp_template(src):
    """New deployment metadata; same archive, model, sampler and R/B/MPS.

    The archive's content identity does not include the deployment host. The
    new release records local explicitly; no runtime check is skipped.
    """
    import hashlib
    src=Path(src);root=EVAL/('fa_ls_t_'+hashlib.sha256(str(src).encode()).hexdigest()[:8])
    if root.exists():return root
    root.mkdir(parents=True)
    c=copy.deepcopy(read(src/'campaign.json'));release=read(src/'RUNTIME_RELEASE.json')
    c.update(server_node='local',server_python='/data/mw_finetune/dp/env/bin/python',
        longseg_deployment_origin=dict(source=str(src),server_node='h100',target='RTX 4090',
            archive_content_unchanged=True,R_B_MPS_unchanged=True))
    if c['public_port']<23100 or c['public_port']>23199:raise ValueError('Public endpoint must use the existing direct-port range')
    from exp.r17_integration.ops.formal_eval import open_tunnels
    open_tunnels(c,root)
    c.update(assets=str(root/'assets'),local_assets=str(root/'assets'),runtime_from=str(root))
    budget.tree(src/'assets',root/'assets')
    for name in ('runtime.tar','workers.tar','WORKER_DEPLOYMENT.json'):
        ancestor=src
        while not (ancestor/name).exists():
            old=read(ancestor/'campaign.json');ancestor=Path(old.get('filter_failed_source') or old['runtime_from'])
        budget.copy(ancestor/name,root/name)
    for name in ('STUDY.json',):
        original=src/name if (src/name).exists() else Path(read(src/'campaign.json')['cache_study'])
        if not original.is_absolute():original=Path.cwd()/original
        budget.copy(original,root/name)
    c['cache_study']=str(root/'STUDY.json')
    release.update(node='local',longseg_deployment_origin=str(src/'RUNTIME_RELEASE.json'))
    budget.dump(root/'RUNTIME_RELEASE.json',release);budget.dump(root/'campaign.json',c)
    return root

def initialize():
    if (HERE/'CAMPAIGNS.json').exists():raise FileExistsError('Catalog already built')
    audit=read(HERE/'AUDIT.json');affected={r['run'] for r in audit['rows'] if r['affected']}
    jobs=[];mapping={};done=set()
    # Rerun the complete MT50 grids once per logical lane, including the
    # never-launched full-step filtered definitions. Continuations collapse
    # back to a fresh whole plan; no old outcome is inherited.
    for model in TOK:
        for mode,source in zip(('reduced','full'),MT[model]):
            fleet=workers(EVAL/source)
            for arm in ('original','filtered'):
                src=EVAL/source if arm=='original' else EVAL/f'ffk1015_{model}_mt50_{mode}_descent'
                origin=src
                if model=='dp':src=local_dp_template(src)
                n=f'fa_ls_{TOK[model]}_{mode[0]}_{"u" if arm=="original" else "f"}_m'
                j=clone(src,n,workers=fleet,mode=mode,stage='descent')
                j.update(arm=arm,original_source=str(origin));budget.dump(Path(j['run'])/'FFK_DEFINITION.json',j)
                jobs.append(j);mapping[origin.name]=n;done.add(origin.name)
                for old in affected:
                    c=read(EVAL/old/'campaign.json')
                    if c.get('cache_resume_selection',{}).get('source')==str(origin):mapping[old]=n;done.add(old)
            # Library-independent full-commit fallback and, on DP's new GPU,
            # a same-runtime/same-batch native-length pure reference.
            origin=EVAL/source;src=local_dp_template(origin) if model=='dp' else origin
            for kind,token in ([('long','p'),('pure','b')] if model=='dp' else [('long','p')]):
                n=f'fa_ls_{TOK[model]}_{mode[0]}_{token}_m'
                j=clone(src,n,workers=fleet,mode=mode,stage=kind,kind=kind)
                j.update(arm='pure_reference');budget.dump(Path(j['run'])/'FFK_DEFINITION.json',j);jobs.append(j)
    # Formal-A is rebuilt from the frozen ladders, never from withdrawn dev
    # levels or off classifications. Both modes start with new OFF results.
    from . import workflow
    workflow.initialize()
    units=read(workflow.INDEX)['units'];jobs += [j for u in units for j in u['jobs']]
    for model in ('dp','fastwam'):
        for mode in ('reduced','full'):
            off=next(u for u in units if u['model']==model and u['mode']==mode and u['arm']=='original' and u['stage']=='off')
            src=off['source'];n=f'fa_ls_{TOK[model]}_{mode[0]}_p_l'
            j=clone(src,n,workers=off['workers'],mode=mode,stage='long',kind='long')
            j.update(arm='pure_reference');budget.dump(Path(j['run'])/'FFK_DEFINITION.json',j);jobs.append(j)
            if not (model=='fastwam' and mode=='reduced'):
                n=f'fa_ls_{TOK[model]}_{mode[0]}_b_l'
                j=clone(src,n,workers=off['workers'],mode=mode,stage='pure',kind='pure')
                j.update(arm='pure_reference');budget.dump(Path(j['run'])/'FFK_DEFINITION.json',j);jobs.append(j)
    # Cache-variant studies keep their complete original plan, including their
    # unaffected Cache groups. Native resume families are one fresh full lane.
    families={};withdrawn_dev=[]
    for old in sorted(affected-done):
        if old.startswith('fa_') and read(EVAL/old/'campaign.json')['domain']=='libero':
            mapping[old]='FORMAL_QUEUE.json';continue
        c=read(EVAL/old/'campaign.json');key=c.get('cache_plan_name',old)
        if read(EVAL/old/'assets/CACHE_STUDY.json').get('eval_pool','A')=='B':
            withdrawn_dev.append(old);mapping[old]='TRASH_ONLY_NO_REPLACEMENT_OWNER_20261017';continue
        families.setdefault(key,[]).append(old)
    from . import segments
    extra,renames,cohorts=segments.build([n for members in families.values() for n in members],workers)
    jobs+=extra;mapping.update(renames)
    budget.dump(HERE/'CAMPAIGNS.json',dict(jobs=jobs,formal_queue=str(workflow.INDEX),withdrawn_dev_diagnostics=withdrawn_dev,source_cohorts=cohorts))
    budget.dump(HERE/'NAMES.json',dict(old_to_new=mapping,decoupled=True))
    refs={}
    for j in jobs:
        if j['stage'] in ('long','pure'):
            key=f'{j["model"]}:{j["benchmark"]}:{j["mode"]}:{j["gpu"]}'+(':long' if j['stage']=='long' else '')
            refs[key]=dict(run=j['run'],new=True,episodes=j['episodes'])
    for r in read(HERE/'REFERENCES.json')['cells']:
        if r['key'] in refs:continue
        eligible=next((v for v in r['references'] if v['descriptor_equal']),None)
        if eligible:refs[r['key']]=dict(run=str(EVAL/eligible['run']),new=False)
    budget.dump(HERE/'REFERENCE_SELECTION.json',dict(references=refs))
    from .ports import finalize
    finalize()
    from .balance import main as balance
    balance()

if __name__=='__main__':
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('action',choices=('initialize',))
    p.parse_args();initialize()
