"""Pin final server/CPU evidence bodies on our never-prepared fresh reruns."""
from pathlib import Path
from .audit import HERE
from .io_budget import dump
from .build import pin
from exp.offline_search.rounds.r17.filter_failed_knob.common import read

def normalize_controls(catalog):
    import hashlib,time,yaml
    from dataclasses import asdict
    from exp.offline_search.rounds.r17.filter_failed_knob.build import cpu_plan,publish_study
    from exp.offline_search.rounds.r17.filter_failed.build import register_metadata
    from exp.offline_search.rounds.r17.filter_failed_knob.common import sha
    from exp.r17_integration.common.config import config_from_dict
    from exp.r17_integration.common.recipe import bundle_identity,recipe_identity
    for j in catalog['jobs']:
        if j['stage'] not in ('pure','long'):continue
        root=Path(j['run']);c=read(root/'campaign.json');seal=read(root/'assets/CACHE_STUDY.json')
        assert not any((root/n).exists() for n in ('PREPARED.json','LAUNCH.json','RECEIPT.json'))
        changed=False
        for row in seal['groups']:
            path=root/'assets'/row['file'];doc=yaml.safe_load(path.read_text())
            assert not doc['recipe']['enabled']
            if doc['recipe']['corrector'] is False:continue
            doc['recipe']['corrector']=False;register_metadata(j['source'],doc)
            cfg=config_from_dict(doc);identity=bundle_identity(cfg)
            path.rename(path.with_name(path.name+'.before_explicit_disabled_corrector_'+str(time.time_ns())))
            path.write_text(yaml.safe_dump(doc,sort_keys=False))
            row.update(bundle_identity=identity,recipe_identity=recipe_identity(cfg),sha256=sha(path),
                bundle_id='ls_'+row['suite']+'_'+str(row['library_size'])+'_'+row['group']+'_'+identity[:12])
            changed=True
        if not changed:continue
        publish_study(root,Path(j['source']),c,seal)
        _,model,tasks,_=cpu_plan(c)
        assert len(tasks)==j['episodes']
        path=root/'EPISODE_PLAN.json';path.rename(path.with_name(path.name+'.before_explicit_disabled_corrector_'+str(time.time_ns())))
        dump(path,dict(schema='r17.episode_plan.v1',model=model,tasks=[asdict(t) for t in tasks],
            expected_episodes=len(tasks),endpoint=c['endpoint']),compact=True)
        j['task_uid_sha256']=hashlib.sha256('\n'.join(t.task_uid for t in tasks).encode()).hexdigest()
        j['disabled_cache_corrector_explicitly_false']=True
        dump(root/'FFK_DEFINITION.json',j)

def main():
    catalog=read(HERE/'CAMPAIGNS.json')
    assert not any((Path(j['run'])/n).exists() for j in catalog['jobs'] for n in ('PREPARED.json','LAUNCH.json','RECEIPT.json'))
    normalize_controls(catalog)
    for j in catalog['jobs']:
        root=Path(j['run'])
        assert root.name.startswith('fa_ls_')
        assert not any((root/n).exists() for n in ('PREPARED.json','LAUNCH.json','RECEIPT.json'))
        c=read(root/'campaign.json')
        admission=Path(c['filter_failed_source'])/'servers/replica_0/CACHE_CAPACITY_ADMISSION.json'
        c['ffk_dp_capacity_bridge']=(c['model']=='dp' and c['domain']=='libero' and
            c['replicas'] in (3,4) and admission.exists())
        if c['ffk_dp_capacity_bridge']:c['longseg_1017']['native_admission_bridge_source']=str(admission)
        if j['stage'] in ('long','pure'):c['longseg_1017_control_kind']=j['stage']
        pin(root,c);dump(root/'campaign.json',c)
        j['bootstrap_sha256']=c['ffk_bootstrap_sha256'];j['evidence_sha256']=c['ffk_evidence_sha256']
        dump(root/'FFK_DEFINITION.json',j)
    dump(HERE/'CAMPAIGNS.json',catalog)
    lookup={j['name']:j for j in catalog['jobs']};q=read(HERE/'FORMAL_QUEUE.json')
    for u in q['units']:u['jobs']=[lookup.get(j['name'],j) for j in u['jobs']]
    dump(HERE/'FORMAL_QUEUE.json',q)
if __name__=='__main__':main()
