"""Disjoint coordinator transport ports for fresh dual-GPU queue units."""
import copy
from pathlib import Path


def route(c,root,workers=None):
    from exp.r17_integration.ops.formal_eval import open_tunnels
    old={k:c.get(k) for k in ('public_port','driver_port','endpoint_host','endpoint_port','endpoint')}
    c['driver_port']=23139 if c['server_node']=='local' else 23199
    if c['server_node']=='local':c['public_port']=23130
    c['backend_ports']=list(range(c['public_port']+1,c['public_port']+1+c['replicas']))
    c['ports']=[c['public_port'],*c['backend_ports'],c['driver_port']]
    assert len(c['ports'])==len(set(c['ports']))
    projected=copy.deepcopy(c)
    if workers is not None:projected['execution_worker_nodes']=workers
    open_tunnels(projected,Path(root))
    for k in ('endpoint_host','endpoint_port','endpoint','driver_host','driver_public_port'):c[k]=projected[k]
    c['longseg_1017']['transport_rebind']=dict(source=old,destination={k:c.get(k) for k in old},
        model_serving_R_B_MPS_unchanged=True,task_identity_unchanged=True,
        reason='Forward remote workers through native direct-port range; isolate H100 driver from 4090 server/driver ports')
    return c


def finalize():
    """Apply the final transport map only to our never-prepared fresh jobs."""
    from dataclasses import asdict
    import hashlib
    import time
    from .audit import HERE
    from .io_budget import dump
    from exp.offline_search.rounds.r17.filter_failed_knob.common import read
    from exp.offline_search.rounds.r17.filter_failed_knob.build import cpu_plan
    catalog=read(HERE/'CAMPAIGNS.json');changed=[]
    assert not any((Path(j['run'])/n).exists() for j in catalog['jobs'] for n in ('PREPARED.json','LAUNCH.json','RECEIPT.json'))
    for j in catalog['jobs']:
        root=Path(j['run']);c=read(root/'campaign.json')
        if c['longseg_1017'].get('transport_rebind'):continue
        assert root.name.startswith('fa_ls_')
        assert not any((root/n).exists() for n in ('PREPARED.json','LAUNCH.json','RECEIPT.json'))
        route(c,root)
        _,model,tasks,_=cpu_plan(c)
        pin=hashlib.sha256('\n'.join(t.task_uid for t in tasks).encode()).hexdigest()
        assert pin==j['task_uid_sha256']
        old=root/'EPISODE_PLAN.json'
        old.rename(old.with_name(old.name+'.before_transport_'+str(time.time_ns())))
        dump(old,dict(schema='r17.episode_plan.v1',model=model,tasks=[asdict(t) for t in tasks],
            expected_episodes=len(tasks),endpoint=c['endpoint']),compact=True)
        dump(root/'campaign.json',c)
        j['transport_plan_rebound']=True
        dump(root/'FFK_DEFINITION.json',j);changed.append(j['name'])
        print('transport',j['name'],c['endpoint'],c['driver_port'],flush=True)
    dump(HERE/'CAMPAIGNS.json',catalog)
    queue=read(HERE/'FORMAL_QUEUE.json');lookup={j['name']:j for j in catalog['jobs']}
    for u in queue['units']:u['jobs']=[lookup.get(j['name'],j) for j in u['jobs']]
    dump(HERE/'FORMAL_QUEUE.json',queue)
    dump(HERE/'TRANSPORT.json',dict(changed=changed,local_server_port=23130,local_driver=23139,
        h100_driver=23199,task_uids_equal=True,scientific_fields_unchanged=True))

if __name__=='__main__':finalize()
