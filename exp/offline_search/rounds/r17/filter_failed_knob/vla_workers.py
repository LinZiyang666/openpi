"""Worker-host placement for legacy formal-A jobs; serving and plans stay native.

Only the explicit prepare/launch entry points stage or launch anything. Probes
are read-only, bounded, and parallel so an offline agent cannot stall the queue.
"""
from __future__ import annotations

import argparse
from concurrent.futures import ThreadPoolExecutor
import copy
import inspect
import json
import os
from pathlib import Path
import shlex
import subprocess
import sys
import types
from unittest.mock import patch

from .common import read, dump, sha
from . import formal_a as f

from exp.offline_search.rounds.r17.autoscale_1017.switch import enabled, source as preparation_source

MODULE = 'exp.offline_search.rounds.r17.filter_failed_knob.vla_workers'
R16_ISLAND = '/scratch/zixuans8/openpi_r16_conductor_ffk1017'


def lookup(name):
    from .formal_a_queue import analysis_jobs
    return next(j for j in analysis_jobs(False) if j['name'] == name)


def probe(host, pipeline):
    """Read-only file, VRAM/RAM and pending R17 reservation probe."""
    from exp.offline_search.closed_loop.ops.h100.fleet import island
    root = str(island(host)) if pipeline == 'R15' else (
        '/scratch/zixuans8/openpi_r16_conductor_v1' if host == 'timan107' else R16_ISLAND)
    required = ['/scratch/zixuans8/openpi/.venv/bin/python', '/scratch/zixuans8/libero_sim/bin/python']
    if pipeline == 'R15':
        required += [root+'/os_cl/'+n for n in ('node.py', 'run_arm.sh', 'run_gtp_subset.py')]
    elif host == 'timan107':
        required += [root+'/exp/offline_search/rounds/r16/conductor.py', root+'/worker.sha256']
    else:
        from exp.offline_search.rounds.r17.filter_failed.timan1_gpu2 import GUARD
        required += [GUARD+'/sitecustomize.py', '/usr/bin/bwrap']
    from exp.offline_search.rounds.r17.filter_failed.worker_resilience import capacity
    from exp.offline_search.rounds.r17.autoscale_1017.switch import function_source
    code = ('import json,socket;from pathlib import Path\n'+function_source(capacity)+'\n'
            'print(json.dumps(dict(host=socket.gethostname().split(".")[0],'
            'boot_id=Path("/proc/sys/kernel/random/boot_id").read_text().strip(),'
            'missing=[s for s in '+repr(required)+' if not Path(s).exists()],'
            'island='+repr(root)+',installed=Path('+repr(root+'/worker.sha256')+').is_file(),'
            'admission=capacity('+repr(host)+',"/scratch/zixuans8/mw_r17/formal/runs/ffk_legacy_probe",530,2048))))')
    try:
        result = subprocess.run(['tether', 'exec', '--safe', '--timeout', '90s', host, '--',
                                 '/usr/bin/python3', '-B', '-S', '-c', code],
                                capture_output=True, text=True, timeout=95, check=True)
        row = json.loads(result.stdout.strip().splitlines()[-1])
        row['usable'] = row['host'] == host and not row['missing']
        return row
    except (subprocess.SubprocessError, OSError, ValueError) as exc:
        return dict(usable=False, error=str(exc), host=host)


def choose(job, requested='auto', *, require_capacity=False):
    hosts = ('timan107',) if job['pipeline'] == 'R15' else ('timan107', 'timan1')
    requested = 'timan1' if requested == 'timan1_gpu2' else requested
    if requested != 'auto' and requested not in hosts:
        raise ValueError(f'{job["pipeline"]} worker hosts: {hosts}')
    candidates = hosts if requested == 'auto' else (requested,)
    with ThreadPoolExecutor(max_workers=len(candidates)) as pool:
        rows = dict(zip(candidates, pool.map(lambda h: probe(h, job['pipeline']), candidates)))
    # Desired maxima do not need to fit at prepare/launch. Per-spawn RAM/VRAM
    # admission and the global tuner admit the actual fleet, including N=0.
    host = next((h for h in candidates if rows[h]['usable']), None)
    if host is None:
        raise RuntimeError('No online legacy worker island: '+json.dumps(rows))
    return host, rows


def record(job, host, rows, action):
    from exp.offline_search.rounds.r17.autoscale_1017.defaults import queue_workers
    root = Path(job['run']); path = root/'WORKER_HOST_SELECTION.json'
    previous = read(path) if path.exists() else dict(history=[])
    previous['history'].append(dict(action=action, host=host, probes=rows))
    previous.update(schema='ffk.legacy_worker_placement.v1', host=host,
                    source_workers=job['workers'], execution_workers={host: sum(queue_workers(job).values())},
                    server_topology_unchanged=True, plans_unchanged=True)
    from .vla_shared import worker_lock
    if rows[host].get('island'):
        previous['worker_lock'] = worker_lock(root, rows[host]['island'])
    if enabled():previous.update(host_lock_shared_by_capacity=True, capacity_required_at_launch=False, autoscale='dynamic')
    dump(path, previous)


def r15_command(job, action, host):
    root = Path(job['run'])
    env = dict(os.environ, **job['topology']['env']); env['WORKER_HOST'] = host
    if action == 'prepare':
        env['OSCL_ISOLATE_GENERATED_ASSETS'] = '1'
    env['FFK_WORKER_FLEET_ROOT'] = str(root)
    from exp.offline_search.rounds.r17.autoscale_1017.defaults import queue_workers
    if enabled():env['OPENPI_DESIRED_MAXIMUM'] = str(sum(queue_workers(job).values()))
    names = [r['arm'] for r in read(root/'arms.json')]
    argv = [str(f.REPO/'.venv/bin/python'), '-m',
            'exp.offline_search.rounds.r17.filter_failed.legacy',
            'sync' if action == 'prepare' else 'chain']
    if action == 'prepare': argv += ['--concurrent']
    return [*argv, str(root), *names], env


def r15(job, action, requested='auto'):
    root = Path(job['run']); host, rows = choose(job, requested, require_capacity=action == 'launch')
    # A live/partly accepted driver cannot move without an explicit continuation.
    # Completed native arms are still skipped/collected by the original chain.
    for p in root.glob('runs/*/h100_launch.json'):
        from exp.offline_search.closed_loop.ops.h100 import control
        launch = read(p)
        row = next(r for r in read(root/'arms.json') if r['arm'] == p.parent.name)
        _, manifest, digest = control.selection(root, row)
        done = root/'state'/(f'{row["arm"]}.manifest_{digest}.DONE' if manifest else f'{row["arm"]}.DONE')
        if not done.exists() and launch['worker_host'] != host:
            raise ValueError('Previously launched on another host; build a continuation/replacement first: '+str(root))
    record(job, host, rows, action)
    synced = root/'h100_sync/synced.json'
    if action == 'prepare' or not synced.exists() or read(synced).get('worker_host') != host:
        argv, env = r15_command(job, 'prepare', host)
        subprocess.run(argv, env=env, cwd=f.REPO, check=True)
    if action == 'launch':
        argv, env = r15_command(job, action, host)
        subprocess.run(argv, env=env, cwd=f.REPO, check=True)


def routed_code(code, host, lock):
    """Change only native host/worker-lock literals, including nested closures."""
    replacements = {'timan107': host,
                    '/scratch/zixuans8/openpi_trace_h100/os_cl/locks/fleet.lock': lock}
    return code.replace(co_consts=tuple(
        routed_code(v, host, lock) if isinstance(v, types.CodeType) else
        replacements.get(v, v) if isinstance(v, str) else v for v in code.co_consts))


def r16_agent_bootstrap(host, island, lock=None):
    from exp.offline_search.rounds.r17.filter_failed import timan1_gpu2 as gpu2
    # Original deployment archive remains untouched. The agent's host check and
    # lock follow placement; all pool/horizon/seed checks remain in native code.
    lock = lock or (island+'/worker_fleet.lock' if host == 'timan1' else
            '/scratch/zixuans8/openpi_trace_h100/os_cl/locks/fleet.lock')
    code = ('import types,sys;from exp.offline_search.rounds.r16 import conductor as c\n'+
            inspect.getsource(routed_code)+'\n'+
            'c.run_agent.__code__=routed_code(c.run_agent.__code__,'+repr(host)+','+repr(lock)+')\n')
    if host == 'timan1':
        code += ('from openpi.conductor import agent as a\n'
                 'import subprocess\n'
                 'Spec=a.WorkerSpec\n'
                 'def worker_spec(*args,**kwargs):\n'
                 '    args=list(args);args[2]="2"\n'
                 '    kwargs["env"].update(MUJOCO_EGL_DEVICE_ID="2",CUDA_VISIBLE_DEVICES="",'
                 'PYOPENGL_PLATFORM="egl")\n'
                 '    return Spec(*args,**kwargs)\n'
                 'a.WorkerSpec=worker_spec\n'
                 'source='+repr(preparation_source(gpu2.__file__).decode())+'\n'
                 'native_popen=subprocess.Popen\n'
                 'def spawn(argv,**kwargs):\n'
                 '    index=argv.index("python");args=argv[index+1:]\n'
                 '    python="/scratch/zixuans8/libero_sim/bin/python"\n'
                 '    boot="s="+repr(source)+";g={\\"SOURCE\\":s};exec(s,g);g[\\"namespace\\"]("+repr(python)+","+repr(args)+",\\"legacy\\")"\n'
                 '    return native_popen(argv[:index]+["/usr/bin/python3","-B","-S","-c",boot],**kwargs)\n'
                 'a.subprocess=types.SimpleNamespace(Popen=spawn)\n')
    return code+'c.main()\n'


def stage_r16(host, observation):
    """Called by coordinator prepare/launch only; existing native island is reused."""
    if observation['installed']: return
    if host != 'timan1': raise ValueError('Existing R16 island is missing its worker manifest')
    from exp.offline_search.closed_loop.ops.h100 import control
    from exp.r17_integration.ops.formal_eval import transfer
    archive = f.REPO/'exp/offline_search/rounds/r16/conductor_deployment/worker.tar'
    target = '/tmp/ffk1017_r16_'+sha(archive)+'.tar'; island = observation['island']
    transfer(['rsync', 'push', str(archive), host+':'+target])
    # A dedicated new island; no existing worker release is replaced.
    script = (f'mkdir -p {shlex.quote(island)} && '
              f'test "$(sha256sum {shlex.quote(target)} | cut -d " " -f 1)" = {sha(archive)} && '
              f'tar -xf {shlex.quote(target)} -C {shlex.quote(island)} && '
              f'cd {shlex.quote(island)} && sha256sum -c worker.sha256')
    control.remote(host, ['bash', '-c', script])


def run_r16_selected():
    from exp.offline_search.rounds.r17.autoscale_1017.legacy_release import coordinator
    group_root = Path(sys.argv[sys.argv.index('--group')+1]).parent
    release = coordinator(group_root, sys.argv)
    """Opt-in wrapper; existing queued R16 launch commands retain native behavior."""
    from exp.offline_search.rounds.r16 import conductor as c, conductor_runner as r
    from exp.offline_search.closed_loop.ops.h100 import control, fleet
    from exp.offline_search.rounds.r17.filter_failed.legacy import rsync_transport
    from exp.r17_integration.ops.formal_eval import transfer
    group = Path(sys.argv[sys.argv.index('--group')+1]); job = lookup(group.parent.name)
    host, rows = choose(job, os.environ['VLA_WORKER_HOST'], require_capacity=True); stage_r16(host, rows[host])
    if release and host != release['host']:
        if any((group_root/n).exists() for n in ('launch.json','runner_state.json','DONE.json')):
            raise ValueError('legacy failover after launch requires a continuation')
        from exp.offline_search.rounds.r17.autoscale_1017.legacy_release import stage as stage_autoscale
        release = stage_autoscale(group_root, host, pipeline='R16', maximum=release['maximum'], base=rows[host]['island'])
    from . import vla_shared
    island = rows[host]['island']; lock = vla_shared.worker_lock(job['run'], island)
    # Every native equality check remains an equality check against the chosen
    # host. No serving, library, group seal or scientific-plan field is changed.
    functions = [c.R16Driver._record_pull, c.run_driver, r._run, r.run]
    originals = [fn.__code__ for fn in functions]
    native_remote = control.remote
    bootstrap = r16_agent_bootstrap(host, island, lock)
    def remote(node, argv, *args, **kw):
        argv = list(argv)
        if node == host and argv[:2] == ['bash', '-c']:
            marker = '-m exp.offline_search.rounds.r16.conductor '
            if marker in argv[2]:argv[2] = argv[2].replace(marker, '-c '+shlex.quote(bootstrap)+' ', 1)
        return native_remote(node, argv, *args, **kw)
    if release:
        if '--workers' in sys.argv:
            sys.argv[sys.argv.index('--workers')+1] = str(release['maximum'])
        else:
            sys.argv += ['--workers', str(release['maximum'])]
        os.environ['OPENPI_AUTOSCALE_REMOTE'] = release['remote']
    old_island = sys.argv[sys.argv.index('--agent-island')+1]
    sys.argv[sys.argv.index('--agent-island')+1] = island
    record(job, host, rows, 'launch')
    try:
        for fn in functions:fn.__code__ = routed_code(fn.__code__, host, lock)
        with patch.dict(fleet.FLEETS, {host: (Path(island), (2,), 83)} if host == 'timan1' else {}), \
             patch.object(control, 'remote', remote), \
             patch.object(control, 'fleet_lock', vla_shared.fleet_lock), \
             patch.dict(os.environ, FFK_WORKER_FLEET_ROOT=str(job['run'])), \
             patch.object(control, 'run', rsync_transport(control.run, transfer)):
            r.main()
    finally:
        for fn, code in zip(functions, originals):fn.__code__ = code
        sys.argv[sys.argv.index('--agent-island')+1] = old_island


def recover_pi05():
    """Definitions only: replace the unreachable pure lane, enable held jobs."""
    from . import formal_a_queue as q
    prefix = 'ffk1015_pi05_libero_full_formal_A_'
    old_name = prefix+'pure_l10'; new_name = old_name+'_r1'
    index = read(q.INDEX)
    old = next(u for u in index['units'] if u['name'] == old_name)
    if any(u['name'] == new_name for u in index['units']):
        raise ValueError('Replacement already defined')
    root = f.ROOT/'queue'/new_name/new_name
    if root.exists():raise FileExistsError(root)
    source = Path(old['run']);root.mkdir(parents=True)
    manifest = root/'eval500.json';manifest.write_bytes((source/'eval500.json').read_bytes())
    specs = copy.deepcopy(read(source/'arms_in.json'))
    for spec in specs:spec.update(name=new_name+'_c10', manifest=str(manifest))
    f.vf.p.emit(root, 'pi05_l10_50', specs, ['c10'], pool='A', final_root=root)
    j = dict(old['jobs'][0], name=new_name, run=str(root), worker_host_failover=True,
             replacement_of=old_name, result_policy='replacement_only_do_not_merge')
    new = dict(old, **{k:v for k,v in j.items() if k != 'jobs'}, jobs=[j])
    # Retain the old run and its UNIT.json untouched, including every receipt.
    dump(f.ROOT/'worker_failover_1017'/'REPLACED_PURE_L10.json', dict(
        old_unit=old, replacement=new_name, episodes=500, merge_old=False,
        journal_status='timan108 tether node_offline; no complete accepted journal reachable'))
    index['units'] = [u for u in index['units'] if u['name'] != old_name]
    dump(q.INDEX, index);q.put(new)
    overrides = read(f.ROOT/'PURE_A_OVERRIDES.json')
    overrides['pi05:full:libero_10'] = dict(root=str(root), arm=new_name+'_c10', authorized=True,
        shared_by_both_arms=True, replaces=old_name, result_policy='replacement_only_do_not_merge')
    dump(f.ROOT/'PURE_A_OVERRIDES.json', overrides)
    affected=[]
    for u in read(q.INDEX)['units']:
        if u['model'] != 'pi05' or u['mode'] != 'full':continue
        if u['name'] != new_name:
            for job in u['jobs']:
                run=Path(job['run']);q.fresh(run)
                if any(run.glob('runs/*/h100_launch.json')):
                    raise ValueError('Held job unexpectedly launched: '+str(run))
        u['worker_host_failover'] = True
        u['dependencies'] = [new_name if n == old_name else n for n in u.get('dependencies',[])]
        for job in u['jobs']:job['worker_host_failover'] = True
        q.put(u);affected.extend(u['jobs'])
    # Update only the owner-held native commands. Active queues' command bytes
    # are checked separately and the ordinary publisher reuses them verbatim.
    from .commands import script
    def native(path, lines):
        if '_profile_' not in path.name:script(path.with_name(path.stem+'.native.sh'), lines)
    with patch.object(f, 'script', native):f.commands(affected)
    q.publish()
    # R16 units are already in the active 4090 queue: add opt-in aliases only.
    # Their existing scripts, native groups and running processes stay intact.
    from .commands import PREFIX,LAUNCH_PREFIX
    for u in read(q.INDEX)['units']:
        if u['model'] not in ('pi05','groot') or u['mode'] != 'reduced':continue
        for action,prefix in (('prepare',PREFIX),('launch',LAUNCH_PREFIX)):
            lines=[f'{prefix} -m {MODULE} {action}-unit --unit {u["name"]}']
            for base in (f.ROOT/'commands',f.HERE/'commands'):
                script(base/f'{action}_{u["name"]}_worker_auto.sh',lines)
    print('PURE_REPLACEMENT',new_name,500,'replacement_only_do_not_merge',flush=True)


def prepare_r16_job(job, requested='auto'):
    from exp.offline_search.rounds.r17.autoscale_1017.legacy_release import stage
    from exp.offline_search.rounds.r17.autoscale_1017.defaults import queue_workers, physical_maximum
    host, rows = choose(job, requested)
    stage_r16(host, rows[host])
    counts = queue_workers(job)
    maximum = sum(counts.values())
    if enabled() and job.get('workers') == {'timan107': 64} and not job.get('worker_maxima_explicit'):
        maximum = physical_maximum(host)
    stage(Path(job['run']), host, pipeline='R16', maximum=maximum, base=rows[host]['island'])
    record(job, host, rows, 'prepare')


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('action', choices=('prepare', 'launch', 'probe', 'recover-pi05', 'prepare-unit', 'launch-unit'))
    p.add_argument('--job');p.add_argument('--unit');p.add_argument('--worker-host', default='auto')
    args = p.parse_args()
    if args.action == 'recover-pi05':return recover_pi05()
    if args.action.endswith('-unit'):
        from . import formal_a_queue as q
        action=args.action.removesuffix('-unit')
        os.environ['VLA_WORKER_HOST']=args.worker_host
        q.dispatch(action,args.unit)
        if action=='prepare':
            unit=next(u for u in read(q.INDEX)['units'] if u['name']==args.unit)
            for job in unit['jobs']:
                prepare_r16_job(job,args.worker_host)
        return
    job = lookup(args.job)
    if args.action == 'probe':
        host, rows = choose(job, args.worker_host);print(json.dumps(dict(host=host, probes=rows)));return
    if job['pipeline'] == 'R15':return r15(job, args.action, args.worker_host)
    if args.action == 'prepare':
        prepare_r16_job(job,args.worker_host);return
    raise ValueError('R16: use its native launch command with VLA_WORKER_HOST=auto')


if __name__ == '__main__':main()
