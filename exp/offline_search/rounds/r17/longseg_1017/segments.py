"""Replay historical resume cohorts without inheriting any old outcomes."""
from collections import defaultdict
import copy
from pathlib import Path

from .audit import EVAL
from .build import clone
from .io_budget import dump
from exp.offline_search.rounds.r17.filter_failed_knob.common import read
from exp.offline_search.rounds.r17.filter_failed_knob.build import cpu_plan

def signature(task):
    ex=task.extra
    return (task.experiment,ex.get('library_size',500),ex['cache_group'],task.task_id,
            task.orig_init_state_idx,ex.get('seed'),ex['num_steps'],ex['bundle_identity'],
            ex['recipe_identity'],ex['commit_horizon'],ex['yaml_identity'])

def lineage(root):
    chain=[];seen=set()
    while True:
        root=Path(root).resolve()
        if root in seen:raise ValueError('Cyclic source continuation')
        seen.add(root);c=read(root/'campaign.json');chain.append((root,c))
        parent=c.get('cache_resume_selection',{}).get('source')
        if not parent:return list(reversed(chain))
        root=parent

def build(names,workers):
    roots=defaultdict(list)
    for n in names:
        chain=lineage(EVAL/n);roots[chain[0][0]].append(chain)
    jobs=[];mapping={};families=[]
    for number,(base,chains) in enumerate(sorted(roots.items(),key=lambda x:str(x[0])),1):
        if base.name=='csfw_k3':
            # Owner 2026-10-17: R3 fits only the three smaller libraries;
            # replay whole cells, rather than the old restart cohorts.
            union=set();new_names=[]
            for suffix,source,small,replicas in (('s','csfw_k3w5',True,3),('b','csfw_k3w6',False,2)):
                src=EVAL/source;c=read(src/'campaign.json')
                assert c['replicas']==replicas
                groups=[r for r in read(src/'assets/CACHE_STUDY.json')['groups']
                        if (r['library_size']<=1000)==small]
                n=f'fa_ls_v{number:02d}_{suffix}'
                j=clone(src,n,workers=workers(src),groups=groups,stage='cache_variants')
                new_c=read(Path(j['run'])/'campaign.json')
                new_c['longseg_1017']['owner_size_replica_split']=dict(
                    sizes=[250,500,1000] if small else [1500,2000,2500],replicas=replicas,
                    reason='Owner 2026-10-17: RUN_CONFIGS R3 exceeds memory cap at size 1500 and above')
                dump(Path(j['run'])/'campaign.json',new_c)
                _,_,tasks,_=cpu_plan(new_c)
                chosen={signature(t) for t in tasks}
                original=copy.deepcopy(c);original.pop('cache_resume_selection',None)
                original['local_assets']=str(src/'assets')
                _,_,old_tasks,_=cpu_plan(original)
                assert chosen=={signature(t) for t in old_tasks if (t.extra['library_size']<=1000)==small}
                assert not union&chosen
                union|=chosen
                j.update(historical_lane=str(base),historical_segment=str(src),
                    no_old_outcomes_inherited=True,owner_size_replica_split=True,
                    library_sizes=[250,500,1000] if small else [1500,2000,2500],
                    source_selection_canonical_equal=True)
                assert len(tasks)==60000
                dump(Path(j['run'])/'FFK_DEFINITION.json',j);jobs.append(j);new_names.append(n)
            assert len(union)==120000
            for chain in chains:
                for src,_ in chain:mapping[src.name]=new_names
            families.append(dict(source=str(base),jobs=new_names,episodes=120000,
                unique_canonical_outcomes=120000,overlap=0,owner_size_replica_split=True))
            continue
        chain=max(chains,key=len)
        if any([p for p,c in v]!=[p for p,c in chain[:len(v)]] for v in chains):
            raise ValueError('Branching historical continuations need an explicit selection: '+str(base))
        union=set();expected=None;new_names=[]
        for i,(src,c) in enumerate(chain):
            full=copy.deepcopy(c);full.pop('cache_resume_selection',None);full['local_assets']=str(src/'assets')
            _,_,tasks,_=cpu_plan(full);by_uid={t.task_uid:t for t in tasks}
            initial=set(c.get('cache_resume_selection',{}).get('retained_task_uids',[]))
            if i+1<len(chain):
                kept=set(chain[i+1][1]['cache_resume_selection']['retained_task_uids'])-initial
            else:kept=set(c.get('cache_resume_selection',{}).get('missing_task_uids',by_uid))
            assert kept<=set(by_uid),(src,'native full-plan UIDs differ')
            if expected is None:expected={signature(t) for t in tasks}
            chosen={signature(by_uid[u]) for u in kept}
            assert not union&chosen,'A source cohort overlaps its ancestor'
            union|=chosen
            if not kept:continue
            selected=defaultdict(lambda:defaultdict(list))
            for u in kept:
                t=by_uid[u];ex=t.extra;suite='mt50' if c['domain']=='metaworld' else t.experiment
                selected[suite,ex.get('library_size',500),ex['cache_group']][ex['task_name']].append(t.orig_init_state_idx)
            groups=[]
            for old in read(src/'assets/CACHE_STUDY.json')['groups']:
                task_indices=selected[old['suite'],old.get('library_size',500),old['group']]
                buckets=defaultdict(list)
                for task,indices in task_indices.items():buckets[tuple(sorted(indices))].append(task)
                for indices,names_ in buckets.items():
                    groups.append(dict(old,tasks=sorted(names_),indices=list(indices),
                        num_trials_per_task=next(by_uid[u].extra['num_trials_per_task'] for u in kept)))
            n=f'fa_ls_v{number:02d}_{i}'
            j=clone(src,n,workers=workers(src),groups=groups,stage='cache_variants')
            new=read(Path(j['run'])/'campaign.json');_,_,planned,_=cpu_plan(new)
            assert {signature(t) for t in planned}==chosen
            assert len(planned)==len(kept)
            j.update(source_selection_canonical_equal=True,source_cohort_episodes=len(kept),
                historical_lane=str(base),historical_segment=str(src),
                no_old_outcomes_inherited=True,uid_namespace='native task/init cohort namespace; RNG seed/campaign unchanged')
            dump(Path(j['run'])/'FFK_DEFINITION.json',j);jobs.append(j);mapping[src.name]=n;new_names.append(n)
        assert union==expected,(base,'fresh cohorts do not cover the whole logical plan')
        families.append(dict(source=str(base),jobs=new_names,episodes=len(union),unique_canonical_outcomes=len(union),overlap=0))
    return jobs,mapping,families
