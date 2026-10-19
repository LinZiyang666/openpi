"""Formal LIBERO test-A grids. Definitions, off/pure references; no dev outcomes."""
from __future__ import annotations

import argparse
import copy
import io
import json
from pathlib import Path
import pickle
import shlex
from contextlib import redirect_stdout

import numpy as np

from .common import HERE, EVAL, REPO, LEGACY, LIBERO, read, dump, sha, summarize
from .workflow import TEST_SOURCE, LABELS
from . import vla_full as vf
from .vla import candidates
from .commands import script, PREFIX, LAUNCH_PREFIX

ROOT=HERE/'formal_a_1016'
MODELS=('dp','fastwam','cosmos','pi05','groot')
ARMS=('original','filtered')
R17_LEVELS=('k50','k40','k30','k20','k10','koff')
VLA_LEVELS=('ir45','ir40','ir32','ir25','off')
OFF={
 'dp':{'original':EVAL/'fsreddp2','filtered':EVAL/'ffk1015_dp_libero_reduced_off'},
 'fastwam':{'original':EVAL/'fsredfw','filtered':EVAL/'ffk1015_fastwam_libero_reduced_off'},
 'cosmos':{'original':EVAL/'ffk1015_cosmos_libero_reduced_off_u','filtered':EVAL/'ffk1015_cosmos_libero_reduced_off'},
}
PURE={'dp':{'full':'hs7dpctl_full','reduced':'hs7dpctl_ddim10'},
      'fastwam':{'full':'hs6fw_res_152931','reduced':'hs7fwk1_4090c'},
      'cosmos':{'full':'formal_cosmos_libero_official2000_20261010_130639_a8ca2043',
                'reduced':'hs2_cosmos_h100_k2incl_c16_t107_20261010_115120'}}


def path(model,mode,arm):
    return ROOT/'runs'/f'ffk1015_{model}_libero_{mode}_formal_A_{arm}'


def library_info(cell,mode,blob):
    """Authoritative episode labels, independent of withdrawn result folders."""
    model,short,size=cell.removesuffix('_k1').split('_')
    parent=LEGACY/f'{model}_{short}'/blob['kwargs']['library']
    ids=sorted(e for values in blob['kwargs']['episode_subset'].values() for e in values)
    assert set(np.unique(blob['method'].inner.base.lib_ep))==set(ids)
    info=dict(model=model,mode=mode,cell=cell.removesuffix('_k1'),size=int(size),
        suite='libero_10' if short=='l10' else 'libero_spatial',parent=str(parent),labels=str(parent/'episodes.json'),
        **summarize(read(parent/'episodes.json'),ids,id_key='episode_id',task_key='task_id'))
    target=ROOT/'LIBRARIES.json';rows=read(target)['cells'] if target.exists() else []
    if not any((r['cell'],r['mode'])==(info['cell'],mode) for r in rows):rows.append(info);dump(target,dict(cells=rows))
    return info


def r17(model,mode,arm):
    from .build import build
    from exp.offline_search.rounds.r17.filter_failed.resume import topology
    source=EVAL/TEST_SOURCE[model][mode=='full']
    levels=R17_LEVELS if mode=='full' else R17_LEVELS[:-1]
    name=path(model,mode,arm).name
    with redirect_stdout(io.StringIO()):
        job=build(source,name,unfiltered=arm=='original',ladder_source=EVAL/LIBERO[model][mode=='full'],test_levels=levels)
    run=Path(job['run'])
    c=read(run/'campaign.json');counts=c.get('execution_worker_nodes',c['worker_nodes'])
    release=read(source/'RUNTIME_RELEASE.json')
    if c['runtime']!=release['root']:
        c['formal_A_runtime_origin']=dict(source_pointer=c['runtime'],published_release=str(source/'RUNTIME_RELEASE.json'),
            reason='Use the sealed official-A archive/ready identity, not a later mutable runtime pointer')
        c['runtime']=release['root'];dump(run/'campaign.json',c)
        job['runtime']=c['runtime'];dump(run/'FFK_DEFINITION.json',job)
    if 'timan1' in counts:
        # Owner permits any environment machine. Keep the original total W.
        counts={k:v for k,v in counts.items() if k!='timan1'}
        counts['timan108']=counts.get('timan108',0)+c.get('execution_worker_nodes',c['worker_nodes'])['timan1']
        with redirect_stdout(io.StringIO()):topology(run,','.join(f'{k}={v}' for k,v in counts.items()))
        c=read(run/'campaign.json')
    result=dict(job,model=model,mode=mode,arm=arm,pipeline='R17',stage='grid',levels=levels,
        off_reuse=str(OFF[model][arm]) if mode=='reduced' else None,pure_reference=str(EVAL/PURE[model][mode]),
        workers=c.get('execution_worker_nodes',c['worker_nodes']),batch=(c.get('dp_batching') or c.get('fastwam_batching') or {}).get('max_batch',1),mps=c.get('mps'))
    dump(run/'FFK_DEFINITION.json',result)
    return result


def full_vla(model,arm):
    p=vf.p;root=path(model,'full',arm);root.mkdir(parents=True,exist_ok=True)
    assert not any(root.rglob('LAUNCH.json')) and not any(root.rglob('DONE.json'))
    jobs=[];proof=[]
    for cell in (c for c in vf.CELLS if c.startswith(model+'_')):
        run=root/f'{root.name}_{cell}';manifest=run/'eval500.json';manifest.parent.mkdir(parents=True,exist_ok=True)
        manifest.write_bytes((vf.TRACE/'r11_knob_1/eval500.json').read_bytes());specs=[]
        for level in VLA_LEVELS:
            rec=vf.original(cell,level,True);source=Path(rec['path']);blob=pickle.loads(source.read_bytes())
            info=library_info(cell,'full',blob)
            if arm=='filtered':
                out=candidates(blob,info['kept_episode_ids']);vf.unchanged(blob,out)
                target=root/'fits'/f'{cell}_{level}.pkl';target.parent.mkdir(exist_ok=True)
                target.write_bytes(pickle.dumps(out,protocol=4));rec.update(path=str(target),sha256=sha(target))
            name=f'{root.name}_{cell}_{level}'
            specs.append(p.arm_spec(cell,name,rec,manifest))
            proof.append(dict(cell=cell,level=level,source=str(source),source_sha256=sha(source),path=rec['path'],sha256=rec['sha256'],
                original_bytes_identical=sha(source)==rec['sha256'] if arm=='original' else None,no_refit=True))
        p.emit(run,cell,specs,VLA_LEVELS,pool='A',final_root=run)
        jobs.append(dict(name=run.name,run=str(run),source=vf.topology(cell,'task_descent',True)['source_root'],
            model=model,mode='full',arm=arm,pipeline='R15',episodes=2500,gpu='H100' if model=='pi05' else 'RTX 4090',
            replicas=vf.topology(cell,'task_descent',True)['replicas'],batch=1,
            workers=vf.topology(cell,'task_descent',True)['workers'],topology=vf.topology(cell,'task_descent',True),levels=VLA_LEVELS))
    dump(root/'ARTIFACTS.json',dict(rows=proof))
    return jobs


def reduced_vla(model,arm):
    from exp.offline_search.rounds.r16 import conductor as c,retest as rt,analyze as a
    root=path(model,'reduced',arm);root.mkdir(parents=True,exist_ok=False)
    original=c.read_group(vf.TRACE/f'r16_conductor_test_{model}_k1_v1/group.json')
    design=read(REPO/'exp/offline_search/rounds/r16/design.json');group=copy.deepcopy(original);entries=[];proof=[]
    offroot=(REPO/'exp/offline_search/rounds/r16/k1ref/runs/off' if arm=='original' else HERE/'vla/runs/off')
    offgroup=read(offroot/(f'r16_k1ref_{model}_off' if arm=='original' else f'ffk1015_{model}_libero_reduced_off')/'group.json')
    coverage={e['row']['r16_cell']:{t for t,i in e['dispatch_pairs']} for e in offgroup['entries']}
    for source in original['entries']:
        cell=source['row']['r16_cell'];basecell=cell.removesuffix('_k1')
        for level in VLA_LEVELS:
            tasks=[t for t in range(10) if level!='off' or t not in coverage.get(cell,set())]
            if not tasks:continue
            name=f'{root.name}_{cell}_{level}'
            entry=rt.build_entry(source,root,name,{str(t):level for t in range(10)},tasks,design,a.Reader())
            fit=rt.fit_path(entry['row']);before=pickle.loads(fit.read_bytes())
            info=library_info(cell,'reduced',before)
            original_blob=pickle.loads(rt.fit_path(source['row']).read_bytes())
            assert pickle.dumps(before['method'].inner,protocol=4)==pickle.dumps(original_blob['method'].inner,protocol=4)
            if arm=='filtered':
                blob=candidates(before,info['kept_episode_ids']);vf.unchanged(before,blob)
                fit.write_bytes(pickle.dumps(blob,protocol=4))
            entry.update(fit_sha256=sha(fit),formal_grid_level=level)
            # Re-seal the ordinary entry after applying the existing row filter.
            seal=read(fit.parent/'ROOT.json');seal['fit_sha256']={str(fit):sha(fit)}
            seal['definitions_sha256']['fit.pkl']=sha(fit);dump(fit.parent/'ROOT.json',seal)
            entry['source_seal_sha256']=sha(fit.parent/'ROOT.json')
            entries.append(entry);proof.append(dict(cell=basecell,level=level,source=str(rt.fit_path(source['row'])),
                fitted_arrays_unchanged=True,rows_filtered=arm=='filtered',no_refit=True))
    group.update(root=str(root),entries=entries,phase='test',references=None,k1ref=False,uniform_tables=True,
                 formal_test_A=True,original_episode_uids=False,yaml_id_is_original_arm=True)
    group['definitions_sha256']={str(p):sha(p) for p in root.rglob('*') if p.is_file()}
    group.pop('group_sha256',None);group['group_sha256']=a.sha(c.canonical(group));dump(root/'group.json',group)
    dump(root/'ROOT.json',dict(group_sha256=group['group_sha256'],definitions_sha256=group['definitions_sha256']))
    dump(root/'ARTIFACTS.json',dict(rows=proof))
    return [dict(name=root.name,run=str(root),source=str(Path(original['root'])),model=model,mode='reduced',arm=arm,
        pipeline='R16',gpu='RTX 4090',replicas=3 if model=='pi05' else 4,batch=1,workers={'timan107':64},
        episodes=sum(len(e['pairs']) for e in entries),levels=VLA_LEVELS,off_reuse=str(offroot/(f'r16_k1ref_{model}_off' if arm=='original' else f'ffk1015_{model}_libero_reduced_off')/'group.json'))]


def commands(jobs):
    for job in jobs:
        run=Path(job['run']);prepare=[];launch=[]
        if job['pipeline']=='R17':
            prepare=[f'{PREFIX} -m exp.offline_search.rounds.r17.filter_failed_knob.stage --run {run}']
            launch=[f'{LAUNCH_PREFIX} -m exp.offline_search.rounds.r17.filter_failed_knob.launch --run {run}']
        elif job['pipeline']=='R15':
            from . import vla_full as v
            env=' '.join(shlex.quote(k+'='+value) for k,value in job['topology']['env'].items())
            names=' '.join(r['arm'] for r in read(run/'arms.json'))
            prepare=[f'env CUDA_VISIBLE_DEVICES= JAX_PLATFORMS=cpu {env} PYTHONPATH=.:src .venv/bin/python -m exp.offline_search.rounds.r17.filter_failed.legacy sync --concurrent {run} {names}']
            launch=[f'env -u CUDA_VISIBLE_DEVICES -u JAX_PLATFORMS {env} PYTHONPATH=.:src .venv/bin/python -m exp.offline_search.rounds.r17.filter_failed.legacy chain {run} {names}']
            if job.get('worker_host_failover'):
                module='exp.offline_search.rounds.r17.filter_failed_knob.vla_workers'
                prepare=[f'{PREFIX} -m {module} prepare --job {job["name"]}']
                launch=[f'{LAUNCH_PREFIX} -m {module} launch --job {job["name"]}']
        else:
            prepare=[f'{PREFIX} -m exp.offline_search.rounds.r17.filter_failed_knob.formal_a check --job {job["name"]}']
            # Generic R16 conductor from this checkout; its resident servers run the frozen R16
            # server checkout (conductor_runner.SERVER_TREE), workers come from the central scheduler.
            launch=[f'{LAUNCH_PREFIX} -m exp.offline_search.rounds.r16.conductor_runner run --group {run}/group.json'
                    ' --agent-island /scratch/zixuans8/openpi_r16_conductor_v1 --port 23159 --ports '+
                    ('23150,23151,23152' if job['model']=='pi05' else '23150,23151,23152,23153')]
        for action,lines in [('prepare',prepare),('launch',launch)]:script(ROOT/'commands'/f'{action}_{job["name"]}.sh',lines)
    for model in MODELS:
        for mode in ('reduced','full'):
            for arm in ARMS:
                group=[j for j in jobs if (j['model'],j['mode'],j['arm'])==(model,mode,arm)]
                for action in ('prepare','launch'):
                    script(ROOT/'commands'/f'{action}_profile_{model}_{mode}_{arm}.sh',
                        [f'bash {ROOT}/commands/{action}_{j["name"]}.sh' for j in group])


def build_all():
    ROOT.mkdir(exist_ok=False);jobs=[]
    for model in MODELS:
        for mode in ('reduced','full'):
            for arm in ARMS:
                if model in TEST_SOURCE:rows=[r17(model,mode,arm)]
                elif mode=='full':rows=full_vla(model,arm)
                else:rows=reduced_vla(model,arm)
                jobs.extend(rows);dump(ROOT/'CAMPAIGNS.json',dict(jobs=jobs))
                print('DEFINED',model,mode,arm,sum(r['episodes'] for r in rows),flush=True)
    commands(jobs)


def main():
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('action',choices=['build','check','analyze','inventory'])
    p.add_argument('--job');args=p.parse_args()
    if args.action=='build':build_all()
    else:
        from . import formal_a_analysis as analysis
        getattr(analysis,args.action)(args.job)


if __name__=='__main__':main()
