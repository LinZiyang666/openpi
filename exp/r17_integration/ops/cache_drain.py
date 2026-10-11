"""Drain a closed-source cache continuation without changing its frozen code.

Pause only the owned fleet parent: its existing N4 children finish and exit
normally, and no replacement is spawned. The launcher keeps renewing leases.
No worker, replica or driver is signalled before that episode boundary.
"""
import json
import os
from pathlib import Path
import signal
import time

from .formal_process import alive, ticks, write_json


def fleet_action(proof, action):
    """Executed on the worker host using its already sealed process helper."""
    import json
    import os
    from pathlib import Path
    import signal
    import time
    from exp.r17_integration.ops.formal_process import alive, ticks
    entry, name = proof['entry'], proof['campaign']
    if not alive(entry) or Path('/proc/'+str(entry['pid'])).stat().st_uid != os.getuid():
        pass
    argv = Path('/proc/'+str(entry['pid'])+'/cmdline').read_bytes().decode().strip('\0').split('\0')
    if (argv != entry['argv'] or 'exp.r17_integration.ops.formal_workers' not in argv or
            '--campaign' not in argv or
            not Path(argv[argv.index('--campaign')+1]).is_relative_to(Path('/tmp/r17_formal')/name)):
        pass
    config = json.loads(Path(argv[argv.index('--campaign')+1]).read_text())
    if not config.get('cache_study') or config.get('worker_max_episodes') != 4:
        pass
    if action == 'freeze':
        os.kill(entry['pid'], signal.SIGSTOP)
        for _ in range(100):
            if Path('/proc/'+str(entry['pid'])+'/stat').read_text().split(') ',1)[1].split()[0]=='T':break
            time.sleep(.01)
        else:raise RuntimeError('fleet parent did not stop')
    children=[]
    for value in Path('/proc/'+str(entry['pid'])+'/task/'+str(entry['pid'])+'/children').read_text().split():
        pid=int(value); child=dict(pid=pid,start_ticks=ticks(pid))
        if not alive(child):continue
        path=Path('/proc')/value
        args=(path/'cmdline').read_bytes().decode().strip('\0').split('\0')
        if (path.stat().st_uid!=os.getuid() or '--worker-id' not in args or
                not args[args.index('--worker-id')+1].startswith(name+'_') or
                '--max-episodes' not in args or args[args.index('--max-episodes')+1]!='4'):
            raise ValueError('fleet child is not an owned bounded worker')
        children.append(child)
    if action == 'retire':
        if children:pass
        # Deliver stop before CONT; the Python handler runs before recycling.
        os.kill(entry['pid'],signal.SIGTERM)
        os.kill(entry['pid'],signal.SIGCONT)
    elif action not in ('freeze','status'):raise ValueError('unknown drain action')
    return dict(action=action,entry=entry,active_children=children,episode_boundary=not children,time=time.time())


def outstanding(directory):
    directory=Path(directory)/'driver'
    def rows(name):
        return [json.loads(s) for s in (directory/name).read_text().splitlines() if s]
    dispatched={(r['task_uid'],r['attempt']) for r in rows('dispatch.jsonl')}
    reported={(r['task_uid'],r['attempt']) for r in rows('attempts.jsonl') if r.get('accepted')}
    return sorted(dispatched-reported)


def controller(directory, launcher):
    """The only allowed controller is the direct DP-only lane parent."""
    parent=int(next(s.split()[1] for s in Path('/proc/'+str(launcher['pid'])+'/status').read_text().splitlines() if s.startswith('PPid:')))
    argv=Path('/proc/'+str(parent)+'/cmdline').read_bytes().decode().strip('\0').split('\0')
    if ('exp.r17_integration.ops.cache_lane' not in argv or '--lane' not in argv or
            argv[argv.index('--lane')+1]!='4090' or '--directory' not in argv or
            Path(argv[argv.index('--directory')+1]).resolve()!=Path(directory).resolve()):
        pass
    owned=json.loads((Path(directory)/'QUEUE_OWNED.json').read_text())
    plan=json.loads((Path(directory)/'LANE_PLAN.json').read_text())
    if plan.get('lane')!='4090' or {j['model'] for j in plan['jobs']}!={'dp_libero'}:
        pass
    if not any(r['pid']==launcher['pid'] and r['start_ticks']==launcher['start_ticks'] for r in owned):
        raise ValueError('lane does not own this launcher')
    return dict(pid=parent,start_ticks=ticks(parent),argv=argv)


def run(directory,lane):
    from . import formal_eval as launch
    import inspect
    directory,lane=Path(directory).resolve(strict=True),Path(lane).resolve(strict=True)
    campaign=json.loads((directory/'campaign.json').read_text())
    if campaign.get('model')!='dp' or campaign.get('server_node')!='local' or not campaign.get('cache_study'):
        pass
    report=json.loads((directory/'RECEIPT.json').read_text())
    fleets=[o for o in report['owned_remote'] if o['label']=='workers']
    if len(fleets)!=1 or fleets[0]['node']!='timan107':raise ValueError('unexpected source fleet')
    owned=[r for r in json.loads((lane/'QUEUE_OWNED.json').read_text()) if r['campaign']==campaign['run_name'] and alive(r)]
    if len(owned)!=1:pass
    parent=controller(lane,owned[0])
    fleet=fleets[0]
    proof=dict(entry=fleet['receipt']['child'],campaign=campaign['run_name'])
    state=dict(schema='r17.cache_boundary_drain.v1',campaign=str(directory),controller=parent,launcher=owned[0],events=[])
    def call(action):
        code=inspect.getsource(fleet_action)+'\nimport json; print(json.dumps(fleet_action('+repr(proof)+','+repr(action)+')))' 
        return json.loads(launch.remote(fleet['node'],['env','PYTHONPATH='+launch.pythonpath(fleet['root']),fleet['python'],'-c',code],replay_safe=action=='status'))
    with launch.retry_scope(directory):
        frozen=call('freeze');state['events'].append(frozen);write_json(directory/'BOUNDARY_DRAIN.json',state)
        until=time.monotonic()+campaign['episode_seconds']*4+120
        while frozen['active_children']:
            if time.monotonic()>until:raise TimeoutError('N4 episode-boundary drain expired; fleet remains paused')
            time.sleep(15)
            frozen=call('status');state['events'].append(frozen);write_json(directory/'BOUNDARY_DRAIN.json',state)
        pending=outstanding(directory)
        if pending:raise ValueError('workers exited with unresolved dispatched attempts: '+repr(pending))
        if not alive(parent) or controller(lane,owned[0])!=parent:pass
        # Prevent its already-imported automatic retry from racing our resize.
        # This controller owns only the recorded DP launcher, never H100 jobs.
        os.kill(parent['pid'],signal.SIGSTOP)
        state['events'].append(call('retire'))
        # Driver's fsync has completed, every child exited after its final ACK.
        driver=next(o for o in report['owned_local'] if o['label']=='driver')
        if alive(driver):os.kill(driver['pid'],signal.SIGTERM)
        if alive(owned[0]):os.kill(owned[0]['pid'],signal.SIGTERM)
        until=time.monotonic()+600
        while alive(owned[0]) and time.monotonic()<until:time.sleep(1)
        closed=json.loads((directory/'RECEIPT.json').read_text())
        if alive(owned[0]) or closed.get('all_owned_closed') is not True:
            raise ValueError('drained DP launcher has not confirmed owned closure')
        if alive(parent):
            os.kill(parent['pid'],signal.SIGTERM);os.kill(parent['pid'],signal.SIGCONT)
        state.update(status='DRAINED',outstanding_attempts=[],episode_boundary=True)
        write_json(directory/'BOUNDARY_DRAIN.json',state)
    return state


if __name__=='__main__':
    import argparse
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--campaign',type=Path,required=True);p.add_argument('--lane',type=Path,required=True)
    a=p.parse_args();print(json.dumps(run(a.campaign,a.lane)))
