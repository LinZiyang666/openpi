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


def execution_host(c,root):
    """filter_failed.stage: move a built, never-prepared H100 job to the R17_H100_NODE machine.

    The route() rebind again, routing fields only: the driver entry of ports ->
    R17_DRIVER_PORT, endpoint (and backend) host -> R17_H100_HOST, plus
    execution_host. EPISODE_PLAN tasks change only server_host/server_port; stage
    then re-derives them with the recorded planner. The previous campaign.json
    and EPISODE_PLAN.json stay beside them as *.before_alias_<ns>.
    Without the alias variables the job is returned untouched.
    """
    import time
    from exp.r17_integration.ops.formal_eval import h100_alias,check_execution_host,open_tunnels
    from exp.offline_search.rounds.r17.filter_failed_knob.common import read
    from .io_budget import dump
    alias=h100_alias()
    if c['server_node']!='h100':
        return c
    if alias is None:
        # A job built for an alias machine (e.g. h100-temp) moves back to the H100 itself: same rebind, H100 lane driver.
        if c.get('execution_host','h100')=='h100':
            return c
        from exp.r17_integration.ops.formal_eval import H100_IP
        alias=dict(node='h100',host=H100_IP,driver_port=23199,public_port=None)
    elif alias['driver_port']==23199:raise ValueError('R17_DRIVER_PORT must differ from the H100 lane driver 23199')
    if (c.get('execution_host'),c['driver_port'],c['endpoint_host'])==(alias['node'],alias['driver_port'],alias['host']) and (
            not alias['public_port'] or c['public_port']==alias['public_port']):
        return c
    root=Path(root)
    if any((root/n).exists() for n in ('PREPARED.json','LAUNCH.json','RECEIPT.json')):
        raise ValueError('prepared for '+c.get('execution_host','h100')+'; reset it before moving it to '+alias['node'])
    keys=('ports','driver_port','driver_host','driver_public_port','endpoint_host','endpoint_port','endpoint',
          'backend_host','execution_host','public_port','backend_ports')
    new=copy.deepcopy(c)
    if alias['public_port']:  # jupyter pod: the ports its tunnels forward (formal_eval.POD_PROFILES)
        new['public_port']=alias['public_port']
        new['backend_ports']=list(range(alias['public_port']+1,alias['public_port']+1+c['replicas']))
        new['ports']=[new['public_port'],*new['backend_ports'],c['driver_port']]
    new['ports']=[p for p in new['ports'] if p!=c['driver_port']]+[alias['driver_port']]
    new['driver_port']=alias['driver_port']
    assert len(new['ports'])==len(set(new['ports']))
    projected=copy.deepcopy(new)
    open_tunnels(projected,root)
    for k in ('endpoint_host','endpoint_port','endpoint','driver_host','driver_public_port'):new[k]=projected[k]
    if 'backend_host' in new:new['backend_host']=alias['host']
    new['execution_host']=alias['node']
    assert {k for k in set(c)|set(new) if c.get(k)!=new.get(k)}<=set(keys)
    new['execution_host_rebind']=dict(source={k:c[k] for k in keys if k in c},destination={k:new[k] for k in keys if k in new},
        model_serving_R_B_MPS_unchanged=True,task_identity_unchanged=True,
        reason='Serve this H100 job on '+alias['node']+'; own driver port beside the H100 lane')
    plan=read(root/'EPISODE_PLAN.json')
    if any((t['server_host'],t['server_port'])!=(c['endpoint_host'],c['endpoint_port']) for t in plan['tasks']):
        raise ValueError('EPISODE_PLAN routing differs from campaign endpoint')
    routing=dict(server_host=new['endpoint_host'],server_port=new['endpoint_port'])
    plan.update(endpoint=new['endpoint'],tasks=[dict(t,**routing) for t in plan['tasks']])
    stamp=str(time.time_ns())
    for name in ('campaign.json','EPISODE_PLAN.json'):(root/name).rename(root/(name+'.before_alias_'+stamp))
    dump(root/'EPISODE_PLAN.json',plan,compact=True);dump(root/'campaign.json',new)
    print('execution_host',c['run_name'],alias['node'],new['endpoint'],new['driver_port'],flush=True)
    return new


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
