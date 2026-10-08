"""Coordinator-only resident server lifecycle. Never called by CPU verification."""
from __future__ import annotations

import argparse
import fcntl
import json
import os
from pathlib import Path
import shlex
import threading
import time
import uuid

from .prepare import HERE, REPO, write_json
from .conductor import read_group, run_driver


def server_count(group, groot_servers_per_suite=None):
    m=group['model'];smoke='smoke' in Path(group['root']).name
    if groot_servers_per_suite is not None and (type(groot_servers_per_suite) is not int or groot_servers_per_suite<1):
        raise ValueError('GR00T servers per suite must be a positive integer')
    if m=='pi05':return 1 if smoke else 3
    return 2*(groot_servers_per_suite if groot_servers_per_suite is not None else (1 if smoke else 2))


def server_specs(group, ports, generation, *, groot_servers_per_suite=None):
    m=group['model'];count=server_count(group,groot_servers_per_suite)
    if len(ports)!=count:raise ValueError(f'{m} requires {count} servers')
    if len(set(ports))!=count or any(not 23100<=p<=23199 for p in ports):raise ValueError('Invalid/duplicate server ports')
    specs=[]
    for i,port in enumerate(ports):
        suite='libero_10' if m=='pi05' or i<count//2 else 'libero_spatial'
        entry=next(e for e in group['entries'] if e['row']['suite']==suite)
        row=entry['row'];name=row['arm'];root=Path(group['root'])
        plugin=['--os-method',row['method'],'--os-kwargs',json.dumps(row['kwargs']),
                '--os-cell',row['cell'],*row['plugin_args']]
        specs.append(dict(role='server',server_host='local',run_root=str(root),
            model=m,suite=suite,port=port,yaml=row['yaml'],out=str(root/'runs'/f'resident_{generation}_{i}'),
            tag=f'r16_{m}_k{group["k"]}_{generation}_{i}',plugin=plugin,
            env=dict(STAGE1_ONLY='0',OMP='1',GROOT_DENOISING_STEPS=str(group['k'])),
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
    per_suite=getattr(args,'groot_servers_per_suite',None);count=server_count(group,per_suite)
    state=json.loads(plan.read_bytes()) if plan.exists() else None
    if state and state['group_sha256']!=group['group_sha256']:raise ValueError('Changed runner group')
    if state and state.get('retired'):
        stop_servers(state['retired']);state['retired']=[];write_json(plan,state)
    old=state['servers'] if state else [];alive={s['port']:local_server.status(s)['running'] for s in old}
    if args.ports is not None:
        # Validate the complete requested topology before retiring anything.
        desired=server_specs(group,args.ports,1,groot_servers_per_suite=per_suite)
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
    candidates=server_specs(group,ports,gen,groot_servers_per_suite=per_suite)
    specs=[next((s for s in kept if s['port']==new['port']),new) for new in candidates]
    state['servers']=specs;write_json(plan,state)
    return specs


def _run(args):
    # Own receipt + fleet lock; this code never stops any other coordinator line.
    from exp.offline_search.closed_loop.ops.h100 import local_server
    from exp.offline_search.closed_loop.ops.h100 import control
    global BASE_COMMAND
    group=read_group(args.group);root=Path(group['root'])
    if not 1<=args.workers<=64:raise ValueError('Worker cap is 64')
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
    server_count(group,getattr(args,'groot_servers_per_suite',None))
    if args.smoke_episodes is not None and 'smoke' not in root.name:raise ValueError('Use prepare-smoke first')
    remote_repo=args.agent_island
    if remote_repo=='/scratch/zixuans8/openpi_trace_h100' or not remote_repo.startswith('/scratch/zixuans8/openpi_r16_conductor_'):
        raise ValueError('Use an isolated R16 conductor worker island')
    # Fail before starting models if the old line owns the worker fleet.
    control.remote('timan107',['flock','-n','/scratch/zixuans8/openpi_trace_h100/os_cl/locks/fleet.lock','true'])
    specs=prepare_servers(group,args)
    BASE_COMMAND=local_server.command;local_server.command=resident_command
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
    finally:local_server.command=BASE_COMMAND
    endpoints=','.join(f'ziyanglin.com:{s["port"]}'+('='+s['suite'] if group['model']=='groot' else '') for s in specs)
    py='/scratch/zixuans8/openpi/.venv/bin/python'
    lease_path=remote_repo+'/r16_agents/'+group['group_sha256'][:20]+'/lease.json'
    token=uuid.uuid4().hex
    agent=[py,'-B','-m','exp.offline_search.rounds.r16.conductor','agent','--model',group['model'],
           '--phase',group['phase'],'--servers',endpoints,'--driver-host','ziyanglin.com',
           '--port',str(args.port),'--workers',str(args.workers),'--lease-file',lease_path,'--lease-token',token]
    # bash -c receives one properly quoted argument; no JSON-as-shell quoting.
    env=dict(HOME='/home/zixuans8',LIBERO_CONFIG_PATH=remote_repo+'/os_cl/libero',
             OMP_NUM_THREADS='1',OPENBLAS_NUM_THREADS='1',MKL_NUM_THREADS='1',
             PYTHONPATH=remote_repo+'/packages/openpi-client/src:'+remote_repo+'/src:'+remote_repo,
             PYTHONDONTWRITEBYTECODE='1',
             PATH='/scratch/zixuans8/dsp_bin:/usr/local/bin:/usr/bin:/bin')
    def renew(active):
        script=('import json,os,time;from pathlib import Path;'
                'p=Path('+repr(lease_path)+');p.parent.mkdir(parents=True,exist_ok=True);'
                'q=p.with_suffix(".new");q.write_text(json.dumps(dict(token='+repr(token)+
                ',active='+repr(active)+',ts=time.time())));os.replace(q,p)')
        control.remote('timan107',['python3','-c',script],timeout=30,attempts=1)
    renew(True)
    remote_log=str(Path(lease_path).parent/'agent.log')
    remote='cd '+shlex.quote(remote_repo)+' && nohup env '+shlex.join([f'{k}={v}' for k,v in env.items()]+agent)+\
           ' >'+shlex.quote(remote_log)+' 2>&1 </dev/null &'
    control.remote('timan107',['bash','-c',remote],timeout=30)
    lease_stop=threading.Event()
    def heartbeat():
        while not lease_stop.wait(15):
            try:renew(True)
            except Exception as error:print('AGENT_LEASE_RENEW_FAILED '+str(error),flush=True)
    thread=threading.Thread(target=heartbeat,daemon=True);thread.start()
    write_json(root/'agent_lease.json',dict(host='timan107',lease_path=lease_path,log=remote_log,token=token,
                                          expires_after_s=90,transport='existing control.remote / tether'))
    try:
        args.servers=endpoints
        # The agent retries its connection until driver bind/setup completes.
        run_driver(args)
    finally:
        lease_stop.set();thread.join(timeout=35)
        try:
            renew(False)
            deadline=time.monotonic()+30
            while True:
                try:
                    control.remote('timan107',['flock','-n','/scratch/zixuans8/openpi_trace_h100/os_cl/locks/fleet.lock','true'],timeout=10,attempts=1)
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
    from exp.offline_search.closed_loop.ops.h100 import control
    previous={key:os.environ.get(key) for key in ('SERVER_HOST','WORKER_HOST')}
    os.environ.update(SERVER_HOST='local',WORKER_HOST='timan107')
    try:
        # The old chain holds this orchestration lease BETWEEN arms too.
        # A free remote episode lock alone must never steal its inter-arm gap.
        with control.fleet_lock(), (Path(group['root'])/'runner.lock').open('a') as lock:
            fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
            _run(args)
    finally:
        for key,value in previous.items():
            if value is None:os.environ.pop(key,None)
            else:os.environ[key]=value


def main():
    ap=argparse.ArgumentParser(description=__doc__);ap.add_argument('action',choices=['run'])
    ap.add_argument('--group',type=Path,required=True);ap.add_argument('--agent-island',required=True)
    ap.add_argument('--workers',type=int,default=64);ap.add_argument('--port',type=int,default=23179)
    ap.add_argument('--ports',type=lambda s:list(map(int,s.split(','))))
    ap.add_argument('--groot-servers-per-suite',type=int,
                    help='Identical processes per checkpoint (default: 2; isolated smoke: 1)')
    ap.add_argument('--smoke-episodes',type=int)
    run(ap.parse_args())


if __name__=='__main__':main()
