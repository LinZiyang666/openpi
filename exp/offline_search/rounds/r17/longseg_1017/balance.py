"""Place the movable DP cache study on the qualified 4090 R4/B8 shape."""
from dataclasses import asdict
import hashlib
from pathlib import Path
import time
import yaml

from .audit import HERE
from .catalog import local_dp_template
from .build import clone,pin
from .ports import route
from .io_budget import dump,copy
from exp.offline_search.rounds.r17.filter_failed_knob.common import read
from exp.offline_search.rounds.r17.filter_failed_knob.build import cpu_plan


def main():
    catalog=read(HERE/'CAMPAIGNS.json');jobs=catalog['jobs']
    assert not any((Path(v['run'])/n).exists() for v in jobs for n in ('PREPARED.json','LAUNCH.json','RECEIPT.json'))
    j=next(j for j in jobs if j['name']=='fa_ls_v01_0');root=Path(j['run'])
    if j['gpu']=='H100':
        assert not any((root/n).exists() for n in ('PREPARED.json','LAUNCH.json','RECEIPT.json'))
        original=Path(j['source']);private=local_dp_template(original)
        c=read(root/'campaign.json');template=read(private/'campaign.json')
        assert c['memory_budget_gib']<=40 and c['replicas']==4 and c['dp_batching']['max_batch']==8
        c.update(server_node='local',server_python=template['server_python'],runtime_from=str(private),
                 filter_failed_source=str(private),longseg_deployment_origin=template['longseg_deployment_origin'])
        route(c,root)
        release=root/'RUNTIME_RELEASE.json';release.rename(release.with_name(release.name+'.before_gpu_rebind_'+str(time.time_ns())))
        copy(private/'RUNTIME_RELEASE.json',release)
        _,model,tasks,_=cpu_plan(c)
        assert hashlib.sha256('\n'.join(t.task_uid for t in tasks).encode()).hexdigest()==j['task_uid_sha256']
        p=root/'EPISODE_PLAN.json';p.rename(p.with_name(p.name+'.before_gpu_rebind_'+str(time.time_ns())))
        dump(p,dict(schema='r17.episode_plan.v1',model=model,tasks=[asdict(t) for t in tasks],
            expected_episodes=len(tasks),endpoint=c['endpoint']),compact=True)
        pin(root,c);dump(root/'campaign.json',c)
        j.update(gpu='RTX 4090',source=str(private),original_source=str(original),python=c['server_python'],
            source_full_native_plan_equal=True,gpu_rebound_same_R_B_MPS=True,
            bootstrap_sha256=c['ffk_bootstrap_sha256'],evidence_sha256=c['ffk_evidence_sha256'])
        dump(root/'FFK_DEFINITION.json',j)
    control_name='fa_ls_dp_r_b_v'
    if not any(v['name']==control_name for v in jobs):
        control=clone(j['source'],control_name,workers=j['workers'],mode='reduced',stage='pure',kind='pure')
        control.update(arm='pure_reference',reference_scope='cache_variant_only')
        dump(Path(control['run'])/'FFK_DEFINITION.json',control);jobs.append(control)
    for v in jobs:
        if v['stage']!='cache_variants':continue
        src=Path(v['run']);g=read(src/'assets/CACHE_STUDY.json')['groups']
        steps={yaml.safe_load((src/'assets'/r['file']).read_text())['adapter_binding']['num_steps'] for r in g}
        full={'dp':100,'fastwam':10,'groot':4}[v['model']]
        reduced={('dp','mt50'):10,('dp','libero'):10,('fastwam','mt50'):3,('fastwam','libero'):1,('groot','mt50'):2}[v['model'],v['benchmark']]
        if steps=={full}:v['mode']='full'
        elif steps=={reduced}:v['mode']='reduced'
        else:raise ValueError('Source study mixes unexpected step modes: '+v['name'])
        dump(src/'FFK_DEFINITION.json',v)
    catalog['balance']=dict(movable_cache_study=j['name'],gpu='RTX 4090',R=4,B=8,budget_gib=24,
        same_task_uids=True,new_same_GPU_pure=control_name)
    dump(HERE/'CAMPAIGNS.json',catalog)
    inventory=read(HERE/'VARIANT_REFERENCES.json')
    for r in inventory['rows']:
        if r['name']==j['name']:
            r.update(key='dp:mt50:reduced:RTX 4090',reference=str(Path(j['run']).parent/control_name),
                compatible=True,new_matching_control=True,missing_reference_episodes=0)
    dump(HERE/'VARIANT_REFERENCES.json',inventory)
    print('balanced deployment',j['name'],j['gpu'],'native reference',control_name,flush=True)
if __name__=='__main__':main()
