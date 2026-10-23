"""Coordinator-only resident server lifecycle. Never called by CPU verification."""
from __future__ import annotations

import argparse
import fcntl
import json
import os
from pathlib import Path
import shlex
import sys
import threading
import time
import uuid

from .prepare import HERE, REPO, write_json
from .conductor import read_group, run_driver

LOCAL_ISLAND = '/data/mw_finetune/qualification/r16_local_island'  # same worker.tar, local libero_sim config
SERVER_TREE = Path('/home/weiland/projects/openpi_r16_k1ref_server')  # frozen R16 server checkout (k1ref/snapshot.py)
# Where the residents run: the 4090 behind ziyanglin.com by default; a wrapper (longseg_off_1022/runner.py) sets the H100.
ENDPOINT_HOST = 'ziyanglin.com'  # address the driver and the worker agents dial for the residents
SERVER_NODE = 'local'  # the residents' machine for the central scheduler
LOOPBACK = REPO/'exp/offline_search/rounds/r17/filter_failed_knob/r16_local/sitecustomize.py'
# A group's worker agents: the island's timan107 (shared class) and the 4090 itself (local class, its servers).
AGENTS = {'timan107': dict(python='/scratch/zixuans8/openpi/.venv/bin/python', home='/home/zixuans8',
                           path='/scratch/zixuans8/dsp_bin:/usr/local/bin:/usr/bin:/bin', driver_host='ziyanglin.com'),
          'timan1': dict(python='/scratch/zixuans8/openpi/.venv/bin/python', home='/home/zixuans8',
                         path='/scratch/zixuans8/dsp_bin:/usr/local/bin:/usr/bin:/bin', driver_host='ziyanglin.com'),
          # a100: the 4090 host's libero_sim, island and LIBERO assets at the same paths; agent on R17's env311.
          'a100': dict(python='/data/mw_finetune/qualification/h100_workers/env311/bin/python', home='/home/weiland',
                       path='/home/weiland/zl_bin:/usr/local/bin:/usr/bin:/bin', driver_host='ziyanglin.com'),
          'local': dict(python=sys.executable, home='/home/weiland',
                        path='/home/weiland/miniconda3/bin:/usr/local/bin:/usr/bin:/bin', driver_host='127.0.0.1')}


def server_count(group, groot_servers_per_suite=None, split=None):
    m=group['model'];smoke='smoke' in Path(group['root']).name
    if m=='groot' and split:return sum(split)
    if groot_servers_per_suite is not None and (type(groot_servers_per_suite) is not int or groot_servers_per_suite<1):
        raise ValueError('GR00T servers per suite must be a positive integer')
    if m=='pi05':return 1 if smoke else 3
    return 2*(groot_servers_per_suite if groot_servers_per_suite is not None else (1 if smoke else 2))


def server_threads(count, cpus=32):
    """CPU threads per resident: the local servers share 32 pinned CPUs (local_server.CPUS)."""
    return max(1, min(4, cpus//max(1, count)))


def server_specs(group, ports, generation, *, groot_servers_per_suite=None, split=None):
    m=group['model'];count=server_count(group,groot_servers_per_suite,split)
    if len(ports)!=count:raise ValueError(f'{m} requires {count} servers')
    if len(set(ports))!=count or any(not 23100<=p<=23199 for p in ports):raise ValueError('Invalid/duplicate server ports')
    specs=[]
    for i,port in enumerate(ports):
        # GR00T residents serve one suite; --groot-suite-servers sets (libero_10, libero_spatial) counts.
        suite='libero_10' if m=='pi05' or i<(split[0] if split else count//2) else 'libero_spatial'
        entry=next(e for e in group['entries'] if e['row']['suite']==suite)
        row=entry['row'];name=row['arm'];root=Path(group['root'])
        plugin=['--os-method',row['method'],'--os-kwargs',json.dumps(row['kwargs']),
                '--os-cell',row['cell'],*row['plugin_args']]
        specs.append(dict(role='server',server_host='local',run_root=str(root),
            model=m,suite=suite,port=port,yaml=row['yaml'],out=str(root/'runs'/f'resident_{generation}_{i}'),
            tag=f'r16_{m}_k{group["k"]}_{generation}_{i}',plugin=plugin,
            env=dict(STAGE1_ONLY='0',OMP=str(server_threads(count)),GROOT_DENOISING_STEPS=str(group['k'])),
            need_mb=10000 if m=='pi05' else 6500,registry=str(root/'group.json')))
    return specs


def resident_command(spec):
    argv,env=BASE_COMMAND(spec)
    index=next(i for i,value in enumerate(argv) if str(value).endswith(('serve_pi05.py','serve_groot.py')))
    argv[index:index+1]=['-m','exp.offline_search.rounds.r16.resident','--registry',spec['registry'],
                        '--server-tag',str(spec['port'])]
    if spec['model']=='groot':
        argv[index+6:index+6]=['--resident-suite',spec['suite']]
    return argv,env


BASE_COMMAND=None


def stop_servers(specs, *, release_timeout_s=240):
    """Stop every owned receipt, then wait for slow releases; report together."""
    from exp.offline_search.closed_loop.ops.h100 import local_server
    pending=[];failures=[]
    for spec in specs:
        try:print(local_server.stop(spec),flush=True)
        except Exception as error:
            pending.append((spec,error))
    deadline=time.monotonic()+release_timeout_s
    for spec,error in pending:
        if 'LOCAL_PORT_RELEASE_TIMEOUT' in str(error):
            try:
                receipt=local_server.matching(spec)
                local_server.wait_port_release(spec,[receipt] if receipt else [],deadline)
                # Once both owned processes and the port are gone, finish its receipt.
                print(local_server.stop(spec,timeout_s=0),flush=True)
                continue
            except Exception as retry_error:error=retry_error
        failures.append(f'port={spec["port"]}: {error}')
    if failures:raise RuntimeError('R16_STOP_INCOMPLETE after all recorded servers: '+'; '.join(failures))


def extra_ports(count, excluded):
    from exp.offline_search.closed_loop.ops.h100 import local_server
    ports=local_server.choose_ports(count=count)
    if not set(ports)&excluded:return ports
    # A resumed driver's own port is currently free, but is still reserved.
    ports=[]
    for port in range(23100,23197):
        if port in excluded:continue
        try:chosen=local_server.choose_ports(count=1,requested=[port])
        except RuntimeError as error:
            if 'LOCAL_PORT_BUSY' in str(error):continue
            raise
        ports.extend(chosen)
        if len(ports)==count:return ports
    raise RuntimeError('LOCAL_PORT_BUSY: no unreserved server ports')


def prepare_servers(group,args):
    """Reconcile only owned server receipts; episode journals/fences are untouched."""
    from exp.offline_search.closed_loop.ops.h100 import local_server
    root=Path(group['root']);plan=root/'runner_state.json'
    per_suite=getattr(args,'groot_servers_per_suite',None);split=getattr(args,'groot_suite_servers',None)
    count=server_count(group,per_suite,split)
    if split and args.ports is None:raise ValueError('--groot-suite-servers needs --ports')
    state=json.loads(plan.read_bytes()) if plan.exists() else None
    if state and state['group_sha256']!=group['group_sha256']:raise ValueError('Changed runner group')
    if state and state.get('retired'):
        stop_servers(state['retired']);state['retired']=[];write_json(plan,state)
    old=state['servers'] if state else [];alive={s['port']:local_server.status(s)['running'] for s in old}
    if args.ports is not None:
        # Validate the complete requested topology before retiring anything.
        desired=server_specs(group,args.ports,1,groot_servers_per_suite=per_suite,split=split)
        if args.port in args.ports:raise ValueError('Driver/server port overlap')
        kept=[s for target in desired for s in old if alive[s['port']] and
              (s['port'],s['suite'])==(target['port'],target['suite'])]
    else:
        suites=['libero_10'] if group['model']=='pi05' else ['libero_10','libero_spatial']
        kept=[s for suite in suites for s in [x for x in old if x['suite']==suite and alive[x['port']]][:count//len(suites)]]
    retired=[s for s in old if s not in kept]
    changed=bool(retired) or len(kept)!=count
    gen=(state['generation']+int(changed)) if state else 1
    state=dict(generation=gen,servers=kept,retired=retired,group_sha256=group['group_sha256'])
    write_json(plan,state) # never orphan surplus receipts if interrupted during shutdown
    stop_servers(retired)
    state['retired']=[];write_json(plan,state)
    missing=count-len(kept)
    if args.ports is not None:
        ports=args.ports
        fresh=[p for p in ports if p not in {s['port'] for s in kept}]
        if fresh:local_server.choose_ports(count=len(fresh),requested=fresh)
    elif missing:
        fresh=extra_ports(missing,{args.port,*[s['port'] for s in kept]})
        if group['model']=='pi05':ports=[s['port'] for s in kept]+fresh
        else:
            n=count//2;cut=n-sum(s['suite']=='libero_10' for s in kept)
            ports=[s['port'] for s in kept if s['suite']=='libero_10']+fresh[:cut]+\
                  [s['port'] for s in kept if s['suite']=='libero_spatial']+fresh[cut:]
    else:ports=[s['port'] for s in kept]
    if args.port in ports:raise ValueError('Driver/server port overlap')
    candidates=server_specs(group,ports,gen,groot_servers_per_suite=per_suite,split=split)
    specs=[next((s for s in kept if s['port']==new['port']),new) for new in candidates]
    state['servers']=specs;write_json(plan,state)
    return specs


def _run(args):
    # Own receipts; this code never stops any other coordinator line. Worker counts per node
    # come from the central scheduler (driver BatchClient, one NodeExecutor per agent).
    from exp.offline_search.closed_loop.ops.h100 import local_server
    from exp.offline_search.closed_loop.ops.h100 import control
    from exp.offline_search.rounds.r17.scheduler import legacy
    from exp.offline_search.rounds.r17.scheduler.defaults import require
    global BASE_COMMAND
    group=read_group(args.group);root=Path(group['root'])
    if not 23100<=args.port<=23199:raise ValueError('Use the local public forwarded port range')
    if (root/'DONE.json').exists():
        if json.loads((root/'DONE.json').read_bytes())['group_sha256']!=group['group_sha256']:raise ValueError('Changed DONE group')
        # A controller may die after persisting DONE but before releasing its models.
        # stop() authenticates only this group's exact owned receipts; no new launch.
        if (root/'runner_state.json').exists():
            state=json.loads((root/'runner_state.json').read_bytes())
            if state['group_sha256']!=group['group_sha256']:raise ValueError('Changed completed runner group')
            stop_servers(state['servers']+state.get('retired',[]))
        print('GROUP_ALREADY_COMPLETE; owned models released; no launches',flush=True);return
    server_count(group,getattr(args,'groot_servers_per_suite',None),getattr(args,'groot_suite_servers',None))
    if args.smoke_episodes is not None and 'smoke' not in root.name:raise ValueError('Use prepare-smoke first')
    remote_repo=args.agent_island
    if remote_repo=='/scratch/zixuans8/openpi_trace_h100' or not remote_repo.startswith('/scratch/zixuans8/openpi_r16_conductor_'):
        raise ValueError('Use an isolated R16 conductor worker island')
    islands=dict(timan107=remote_repo,timan1=remote_repo,a100=LOCAL_ISLAND,local=LOCAL_ISLAND)
    # The 4090 has no VRAM left for renderers once the residents fill it (pi05 R3: 23 of 24 GB).
    if getattr(args,'no_local_workers',False):del islands['local']
    campaign=legacy.campaign(root.name,SERVER_NODE,tuple(islands))
    legacy.install(root,islands,[(HERE/'conductor.py','exp/offline_search/rounds/r16/conductor.py'),
                                 (LOOPBACK,'loopback/sitecustomize.py')])
    require(campaign,root)
    agents={}
    for node,island in islands.items():
        folder=island+'/r16_agents/'+group['group_sha256'][:20]
        agents[node]=dict(island=island,lease=folder+'/lease.json',log=folder+'/agent.log',
                          config=folder+'/scheduler/campaign.json',lock=folder+'/scheduler/agent.lock')
        # The node's campaign; fail before starting models if this group's previous agent still runs there.
        script=('from pathlib import Path;p=Path('+repr(agents[node]['config'])+');p.parent.mkdir(parents=True,exist_ok=True);'
                'p.write_text('+repr(json.dumps(legacy.node_campaign(campaign,node)))+')')
        control.remote(node,['python3','-c',script],timeout=30)
        control.remote(node,['flock','-n',agents[node]['lock'],'true'],timeout=30)
    write_json(root/'scheduler/campaign.json',campaign)
    specs=prepare_servers(group,args)
    BASE_COMMAND,tree=local_server.command,local_server.TREE
    local_server.command,local_server.TREE=resident_command,SERVER_TREE
    try:
        for spec in specs:
            print(local_server.start(spec),flush=True)
            deadline=time.monotonic()+1800
            while not local_server.status(spec)['listening']:
                status=local_server.status(spec)
                if status['dead']:raise RuntimeError(status['tail'])
                if time.monotonic()>deadline:raise TimeoutError('Resident startup timed out')
                time.sleep(1)
            # Models start one at a time; startup peaks do not overlap.
    finally:local_server.command,local_server.TREE=BASE_COMMAND,tree
    endpoints=','.join(f'{ENDPOINT_HOST}:{s["port"]}'+('='+s['suite'] if group['model']=='groot' else '') for s in specs)
    token=uuid.uuid4().hex
    def renew(active):
        for node,a in agents.items():
            script=('import json,os,time;from pathlib import Path;'
                    'p=Path('+repr(a['lease'])+');p.parent.mkdir(parents=True,exist_ok=True);'
                    'q=p.with_suffix(".new");q.write_text(json.dumps(dict(token='+repr(token)+
                    ',active='+repr(active)+',ts=time.time())));os.replace(q,p)')
            control.remote(node,['python3','-c',script],timeout=30,attempts=1)
    renew(True)
    for node,a in agents.items():
        o,island=AGENTS[node],a['island']
        agent=[o['python'],'-B','-m','exp.offline_search.rounds.r16.conductor','agent','--node',node,'--model',group['model'],
               '--phase',group['phase'],'--servers',endpoints,'--driver-host',o['driver_host'],'--port',str(args.port),
               '--workers',str(campaign['scheduler_nodes'][node]['max']),'--lease-file',a['lease'],'--lease-token',token]
        # bash -c receives one properly quoted argument; no JSON-as-shell quoting.
        env=dict(HOME=o['home'],LIBERO_CONFIG_PATH=island+'/os_cl/libero',
                 OMP_NUM_THREADS='1',OPENBLAS_NUM_THREADS='1',MKL_NUM_THREADS='1',
                 PYTHONPATH=island+'/packages/openpi-client/src:'+island+'/src:'+island,
                 PYTHONDONTWRITEBYTECODE='1',OPENPI_AUTOSCALE_CONFIG=a['config'],PATH=o['path'])
        # Background only the agent: `cd && nohup ... &` backgrounds the whole list in a subshell that keeps
        # the caller's stdout pipe open, so a local (subprocess) start waited for the agent and timed out.
        command='cd '+shlex.quote(island)+' && { setsid nohup env '+shlex.join([f'{k}={v}' for k,v in env.items()]+agent)+\
                ' >'+shlex.quote(a['log'])+' 2>&1 </dev/null & }'
        control.remote(node,['bash','-c',command],timeout=30)
    lease_stop=threading.Event()
    def heartbeat():
        while not lease_stop.wait(15):
            try:renew(True)
            except Exception as error:print('AGENT_LEASE_RENEW_FAILED '+str(error),flush=True)
    thread=threading.Thread(target=heartbeat,daemon=True);thread.start()
    write_json(root/'agent_lease.json',dict(agents=agents,token=token,expires_after_s=90,
                                          transport='existing control.remote (tether exec / local)'))
    try:
        args.servers=endpoints
        os.environ['OPENPI_AUTOSCALE_CONFIG']=str(root/'scheduler/campaign.json')  # the driver's BatchClient
        # The agents retry their connections until driver bind/setup completes.
        run_driver(args)
    finally:
        lease_stop.set();thread.join(timeout=35)
        try:
            renew(False)
            deadline=time.monotonic()+120  # an agent's shutdown collects its kernel evidence first
            for node,a in agents.items():
                while True:
                    try:
                        control.remote(node,['flock','-n',a['lock'],'true'],timeout=10,attempts=1)
                        break
                    except Exception:
                        if time.monotonic()>deadline:raise
                        time.sleep(1)
        except Exception as error:print('AGENT_LEASE_RELEASE_FAILED; own workers expire within 90 seconds: '+str(error),flush=True)
    if (root/'DONE.json').exists():
        stop_servers(specs)
        print('GROUP_COMPLETE; resident fleet released',flush=True)
    else:
        print('INTERRUPTED; completed episodes kept; same command resumes the resident fleet',flush=True)


def run(args):
    group=read_group(args.group)
    # One runner per group; worker nodes are shared with every batch through the central scheduler.
    with (Path(group['root'])/'runner.lock').open('a') as lock:
        fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
        _run(args)


def main():
    ap=argparse.ArgumentParser(description=__doc__);ap.add_argument('action',choices=['run'])
    ap.add_argument('--group',type=Path,required=True);ap.add_argument('--agent-island',required=True)
    ap.add_argument('--port',type=int,default=23179)
    ap.add_argument('--ports',type=lambda s:list(map(int,s.split(','))))
    ap.add_argument('--no-local-workers',action='store_true',help='workers on timan107 only')
    ap.add_argument('--groot-servers-per-suite',type=int,
                    help='Identical processes per checkpoint (default: 2; isolated smoke: 1)')
    ap.add_argument('--groot-suite-servers',type=lambda s:tuple(map(int,s.split(','))),
                    help='GR00T residents for libero_10,libero_spatial (e.g. 3,1; 4,0 once spatial is done)')
    ap.add_argument('--smoke-episodes',type=int)
    run(ap.parse_args())


if __name__=='__main__':main()
