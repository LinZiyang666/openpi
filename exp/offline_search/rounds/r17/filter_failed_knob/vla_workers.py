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
import subprocess
from unittest.mock import patch

from .common import read, dump
from . import formal_a as f

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
    required = ['/scratch/zixuans8/openpi/.venv/bin/python', '/scratch/zixuans8/libero_sim/bin/python'] if host.startswith('timan') else ['/data/mw_finetune/qualification/h100_workers/env311/bin/python', '/home/weiland/projects/openpi_ext/envs/libero_sim/bin/python']  # run_arm.sh PY / SIM
    if pipeline == 'R15':
        required += [root+'/os_cl/'+n for n in ('node.py', 'run_arm.sh', 'run_gtp_subset.py')]
    elif host == 'timan107':
        required += [root+'/exp/offline_search/rounds/r16/conductor.py', root+'/worker.sha256']
    else:
        from exp.offline_search.rounds.r17.filter_failed.timan1_gpu2 import GUARD
        required += [GUARD+'/sitecustomize.py', '/usr/bin/bwrap']
    from exp.offline_search.rounds.r17.filter_failed.worker_resilience import capacity
    code = ('import json,socket;from pathlib import Path\n'+inspect.getsource(capacity)+'\n'
            'print(json.dumps(dict(host=socket.gethostname().split(".")[0],'
            'boot_id=Path("/proc/sys/kernel/random/boot_id").read_text().strip(),'
            'missing=[s for s in '+repr(required)+' if not Path(s).exists()],'
            'island='+repr(root)+',installed=Path('+repr(root+'/worker.sha256')+').is_file(),'
            'admission=capacity('+repr(host)+',"/scratch/zixuans8/mw_r17/formal/runs/ffk_legacy_probe",530,2048))))')
    try:
        from exp.offline_search.closed_loop.ops.h100.fleet import node
        result = subprocess.run(['tether', 'exec', '--safe', '--timeout', '90s', node(host), '--',
                                 '/usr/bin/python3', '-B', '-S', '-c', code],
                                capture_output=True, text=True, timeout=95, check=True)
        row = json.loads(result.stdout.strip().splitlines()[-1])
        # Hostnames differ from fleet keys on the Jetstream machines (a100 'videogen', h100 'vla-cache').
        row['usable'] = row['host'] in (host, {'a100': 'videogen', 'h100w': 'vla-cache'}.get(host)) and not row['missing']
        return row
    except (subprocess.SubprocessError, OSError, ValueError) as exc:
        return dict(usable=False, error=str(exc), host=host)


def choose(job, requested='auto', *, require_capacity=False):
    hosts = (os.environ.get('WORKER_HOST', 'timan107'),) if job['pipeline'] == 'R15' else ('timan107', 'timan1')  # R15: the driver's fleet (fleet.DRIVERS); agents via WORKER_HOSTS
    requested = 'timan1' if requested == 'timan1_gpu2' else requested
    if requested != 'auto' and requested not in hosts:
        raise ValueError(f'{job["pipeline"]} worker hosts: {hosts}')
    candidates = hosts if requested == 'auto' else (requested,)
    with ThreadPoolExecutor(max_workers=len(candidates)) as pool:
        rows = dict(zip(candidates, pool.map(lambda h: probe(h, job['pipeline']), candidates)))
    # Capacity need not fit at prepare/launch: the central scheduler sets the
    # actual worker count, including N=0.
    host = next((h for h in candidates if rows[h]['usable']), None)
    if host is None:
        raise RuntimeError('No online legacy worker island: '+json.dumps(rows))
    return host, rows


def record(job, host, rows, action):
    root = Path(job['run']); path = root/'WORKER_HOST_SELECTION.json'
    previous = read(path) if path.exists() else dict(history=[])
    previous['history'].append(dict(action=action, host=host, probes=rows))
    # Worker counts come from the central scheduler at run time.
    previous.update(schema='ffk.legacy_worker_placement.v1', host=host, source_workers=job['workers'],
                    server_topology_unchanged=True, plans_unchanged=True)
    from .vla_shared import worker_lock
    if rows[host].get('island'):
        previous['worker_lock'] = worker_lock(root, rows[host]['island'])
    dump(path, previous)


def r15_command(job, action, host):
    root = Path(job['run'])
    env = dict(os.environ, **job['topology']['env']); env['WORKER_HOST'] = host
    # Server CPU threads per replica (H100: 10 of its 20 cores, the rest for local workers, as
    # formal_eval.SERVER_CORES; 4090 servers 32 pinned CPUs); the recorded topologies carry OMP=1.
    cores = 10 if env.get('SERVER_HOST', 'h100') == 'h100' else 32
    env['OMP'] = str(max(1, min(4, cores // max(1, int(env.get('LOCAL_SERVER_COUNT') or job.get('replicas') or 1)))))
    if action == 'prepare':
        env['OSCL_ISOLATE_GENERATED_ASSETS'] = '1'
    env['FFK_WORKER_FLEET_ROOT'] = str(root)
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


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('action', choices=('prepare', 'launch', 'probe', 'recover-pi05', 'prepare-unit', 'launch-unit'))
    p.add_argument('--job');p.add_argument('--unit');p.add_argument('--worker-host', default='auto')
    args = p.parse_args()
    if args.action == 'recover-pi05':return recover_pi05()
    if args.action.endswith('-unit'):
        from . import formal_a_queue as q
        return q.dispatch(args.action.removesuffix('-unit'),args.unit)
    job = lookup(args.job)
    if args.action == 'probe':
        host, rows = choose(job, args.worker_host);print(json.dumps(dict(host=host, probes=rows)));return
    if job['pipeline'] == 'R15':return r15(job, args.action, args.worker_host)
    raise ValueError('R16: prepare with formal_a check, launch with rounds.r16.conductor_runner run')


if __name__ == '__main__':main()
