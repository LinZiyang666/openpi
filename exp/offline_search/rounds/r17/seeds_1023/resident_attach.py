"""Keep an R17 resident across prepared campaigns of one lane (owner 10-23).

filter_failed_knob.launch --keep-resident: after a verified complete driver the
campaign leaves its replicas, proxy and private MPS daemon running and registers
them in RESIDENTS/<server host>_<public port>.json. --attach-resident: the next
campaign on that host and public port serves from the registered resident when
model, K (full.yaml), runtime, server Python, ports, replicas, batching, MPS and
the resident bootstrap match and the host has memory for its YAMLs; it only
registers its own YAMLs and starts its own driver and workers, then stops the
resident unless it keeps it too. A mismatching, closed or full resident is
stopped and the campaign starts its own. Without --attach-resident a live
registered resident on the lane is an error.

A resident started to be kept writes replica directories, logs, receipts, leases
and MPS pipes under its own FORMAL_ROOT/rs<port>_<stamp>/ on the server machine,
never in a campaign directory; each campaign records its window there (serving
rows carry `time`). Remote leases (600 s) are renewed between campaigns by
`renew`, a coordinator process. Remote calls need the lane's R17_H100_* env.
"""
from __future__ import annotations

import argparse
from contextlib import contextmanager
import fcntl
import inspect
import json
import os
from pathlib import Path
import shutil
import subprocess
import time

from exp.r17_integration.ops.formal_process import alive, stop_group, ticks, write_json

MODULE = 'exp.offline_search.rounds.r17.seeds_1023.resident_attach'
FORMAL_ROOT = Path('/data/mw_finetune/eval')
RESIDENTS = FORMAL_ROOT/'residents'
RENEW_SECONDS = 120
KEEP_SECONDS = 14*86400  # supervisor deadline of a kept remote resident
MEMORY_FLOOR = .1  # host RAM share that must stay available after the new YAMLs
# Campaign fields a resident is started from; with the asset digests they decide attach.
SIGNATURE = ('model', 'domain', 'checkpoint', 'vendor', 'live_k', 'device', 'replicas', 'memory_budget_gib',
             'server_node', 'public_port', 'backend_ports', 'runtime', 'runtime_identity', 'server_python',
             'server_threads', 'mps', 'dp_batching', 'stage_batching', 'fastwam_batching', 'fastwam_text_table',
             'ffk_dp_capacity_bridge', 'ffk_bootstrap_sha256', 'graph_probe_observations')
ASSETS = ('full.yaml', 'model_spec.json', 'benchmark.json', 'load.json')


def read(path):
    path = Path(path)
    return json.loads(path.read_text()) if path.exists() else None


def host(launch, campaign):
    return 'local' if campaign['server_node'] == 'local' else campaign.get('execution_host') or launch.physical_node('h100')


def registry(launch, campaign):
    return RESIDENTS/f"{host(launch, campaign)}_{campaign['public_port']}.json"


@contextmanager
def locked(path):
    from exp.r17_integration.ops.immutable_release import nvme_dir
    nvme_dir(RESIDENTS, exist_ok=True)
    with Path(path).with_suffix('.lock').open('a') as stream:
        fcntl.flock(stream, fcntl.LOCK_EX)
        yield


def signature(launch, campaign):
    from exp.r17_integration.ops.immutable_release import sha
    assets = Path(campaign['local_assets'])
    value = dict({k: campaign.get(k) for k in SIGNATURE}, host=host(launch, campaign),
                 ffk_arm=(campaign.get('filter_failed_knob') or {}).get('arm'),
                 assets={n: sha(assets/n) if (assets/n).is_file() else None for n in ASSETS})
    return json.loads(json.dumps(value))


def bundles(campaign):
    path = Path(campaign['local_assets'])/'CACHE_STUDY.json'
    return {r['bundle_id'] for r in json.loads(path.read_text())['groups']} if path.exists() else set()


def held_ports(campaign):
    """Local server ports of the lane's registered resident: prepare skips them, launch attaches or stops it."""
    from exp.r17_integration.ops import formal_eval as launch
    if campaign.get('server_node') != 'local' or 'public_port' not in campaign:
        return set()
    reg = read(registry(launch, campaign))
    return set() if reg is None else {reg['signature']['public_port'], *reg['signature']['backend_ports']}


def usage(pids):
    from pathlib import Path
    def kib(path, key):
        for line in Path(path).read_text().splitlines():
            if line.startswith(key+':'):
                return int(line.split()[1])
    total, available = kib('/proc/meminfo', 'MemTotal')//1024, kib('/proc/meminfo', 'MemAvailable')//1024
    try:  # a container (jupyter pod) is bounded by its cgroup, not the node
        limit = Path('/sys/fs/cgroup/memory.max').read_text().strip()
        if limit != 'max':
            used = int(Path('/sys/fs/cgroup/memory.current').read_text())
            total, available = min(total, int(limit)//2**20), min(available, (int(limit)-used)//2**20)
    except (OSError, ValueError):
        pass
    rss = {}
    for pid in pids:
        try:
            rss[str(pid)] = kib(f'/proc/{pid}/status', 'VmRSS')//1024
        except (OSError, TypeError):
            rss[str(pid)] = None
    return dict(total_mib=total, available_mib=available, rss_mib=rss)


def controls(launch, entries):
    """One renew+status session per node (formal_process.controls of the resident's own runtime)."""
    result = {}
    for node, root, python in sorted({(e['node'], e['root'], e['python']) for e in entries}):
        group = [dict(label=e['label'], request=e['request']) for e in entries if (e['node'], e['root'], e['python']) == (node, root, python)]
        code = ('from exp.r17_integration.ops.formal_process import controls; import json; '
                'print(json.dumps(controls('+repr(group)+')))')
        result.update(json.loads(launch.remote(node, ['env', 'PYTHONPATH='+launch.pythonpath(root), python, '-c', code],
                                               seconds=30, replay_safe=True)))
    return result


def serving(status):
    return bool(status) and status.get('status') not in ('CLOSED', 'FAIL') and status.get('alive') is not False \
        and status.get('child_alive') is not False


def live(launch, reg):
    if not all(alive(e) for e in reg['local']):
        return False
    status = controls(launch, reg['remote']) if reg['remote'] else {}
    return all(serving(status.get(e['label'])) for e in reg['remote'])


def room(launch, reg, campaign):
    """Registered YAMLs stay loaded. Projected growth: current replica RSS per registered YAML x new YAMLs."""
    pids = [e['pid'] for e in reg['local']+reg['remote'] if e.get('pid')]
    code = inspect.getsource(usage)+'\nimport json\nprint(json.dumps(usage('+repr(pids)+')))'
    probe = json.loads(launch.remote(reg['host'], [reg['python'], '-c', code], replay_safe=True))
    new = bundles(campaign)-set(reg['bundles'])
    rss = sum(v or 0 for v in probe['rss_mib'].values())
    probe.update(at=time.time(), new_yamls=len(new), registered_yamls=len(reg['bundles']),
                 projected_mib=round(rss*len(new)/max(1, len(reg['bundles']))))
    reg['memory'].append(probe)
    if probe['available_mib']-probe['projected_mib'] < MEMORY_FLOOR*probe['total_mib']:
        return f"memory: {probe['available_mib']} MiB available, {probe['projected_mib']} MiB projected for {len(new)} new YAMLs"
    return None


def archive(reg, status, **extra):
    reg.update(status=status, owner=None, closed_at=time.time(), **extra)
    write_json(RESIDENTS/'history'/f"{reg['key']}.{reg['id']}.json", reg)
    write_json(Path(reg['local_dir'])/'RESIDENT.json', reg)
    path = RESIDENTS/f"{reg['key']}.json"
    if (read(path) or {}).get('id') == reg['id']:
        path.unlink()


def close(launch, reg, status, **extra):
    """Stop the resident (proxy, replicas, then the MPS guard) and archive its registry; caller holds the lock."""
    closure = []
    for e in reversed(reg['remote']):
        try:
            result = json.loads(launch.remote(e['node'], ['env', 'PYTHONPATH='+launch.pythonpath(e['root']), e['python'], '-m',
                'exp.r17_integration.ops.formal_process', 'stop', '--request', json.dumps(e['request'])], seconds=60))
            closure.append(dict(node=e['node'], label=e['label'], closure_confirmed=result.get('alive') is False))
        except Exception as exc:
            closure.append(dict(node=e['node'], label=e['label'], closure_confirmed=False, error=repr(exc)))
    for e in reversed(reg['local']):
        result = stop_group(e, grace=30 if e['label'] == 'mps_guard' else 12)
        closure.append(dict(label=e['label'], **result, closure_confirmed=not alive(e)))
    archive(reg, status, closure=closure, **extra)
    return closure


def claim(launch, campaign, directory, report):
    """Before port admission: the lane's registered resident this campaign serves from, else None."""
    path = registry(launch, campaign)
    if not path.exists():
        return None
    want = campaign.get('resident') or {}
    with locked(path):
        reg = read(path)
        if reg is None:
            return None
        owner = reg.get('owner')
        if owner and alive(owner):
            raise RuntimeError(f"resident {reg['id']} ({path}) is attached to the running {owner['run']}")
        reason = None if live(launch, reg) else 'a resident process closed'
        if reason is None and not want.get('attach'):
            raise RuntimeError(f"{path} registers the live resident {reg['id']}: launch with --attach-resident, or "
                               f"stop it ({MODULE} stop --registry {path})")
        if reason is None:
            ours = signature(launch, campaign)
            diff = sorted(k for k in set(ours) | set(reg['signature']) if ours.get(k) != reg['signature'].get(k))
            reason = 'differs in '+', '.join(diff) if diff else room(launch, reg, campaign)
        if reason:
            report['resident_replaced'] = dict(id=reg['id'], reason=reason,
                closure=close(launch, reg, 'REPLACED', reason=reason, replaced_by=campaign['run_name']))
            return None
        now = time.time()
        reg.update(status='ATTACHED', owner=dict(run=campaign['run_name'], pid=os.getpid(), start_ticks=ticks(os.getpid())))
        reg['batches'].append(dict(run=campaign['run_name'], directory=str(directory), attached_at=now))
        write_json(path, reg)
    campaign['server_run'] = reg['server_run']
    shutil.copyfile(reg['attestations'], Path(directory)/'SERVER_ATTESTATIONS.json')
    report['resident'] = dict(id=reg['id'], role='attached', registry=str(path), server_run=reg['server_run'],
                              local_dir=reg['local_dir'], attached_at=now, memory=reg['memory'][-1],
                              previous=[b['run'] for b in reg['batches'][:-1]])
    return reg


def start(launch, campaign, directory, report):
    """A campaign that will keep its resident starts it under the resident's own directory (same path on the server)."""
    from exp.r17_integration.ops.immutable_release import nvme_dir
    name = f"rs{campaign['public_port']}_{time.strftime('%m%d%H%M%S')}"
    root = nvme_dir(FORMAL_ROOT/name)
    if campaign['server_node'] != 'local':
        launch.remote('h100', ['mkdir', '-p', str(root/'server')])
    campaign['server_run'] = str(root/'servers')
    report['resident'] = dict(id=name, role='started', server_run=campaign['server_run'], local_dir=str(root),
                              started_at=time.time())
    return str(root)


def slim(launch, owned):
    child = (owned.get('receipt') or {}).get('child') or {}
    return dict(node=launch.physical_node(owned['node']), label=owned['label'], root=owned['root'], python=owned['python'],
                transport=owned.get('transport'), pid=child.get('pid'),
                request={k: owned['request'][k] for k in ('lease', 'receipt', 'log')})


def renewer(launch, path, reg):
    with (Path(reg['local_dir'])/'renewer.log').open('ab') as log:
        child = subprocess.Popen([launch.CONTROL_PYTHON, '-u', '-m', MODULE, 'renew', '--registry', str(path), '--id', reg['id']],
                                 cwd=str(launch.ROOT), stdout=log, stderr=subprocess.STDOUT, stdin=subprocess.DEVNULL,
                                 start_new_session=True, close_fds=True)
    return dict(pid=child.pid, start_ticks=ticks(child.pid))


def keep(launch, campaign, directory, report, local, remote_owned, mps_ready, reg):
    """After a verified complete driver: register the resident (new, or the attached one again) for the next campaign."""
    path = registry(launch, campaign)
    acks = read(Path(directory)/'STUDY_BUNDLE_ACKS.json') or dict(rows=[])
    now = time.time()
    with locked(path):
        if reg is None:
            info = report['resident']
            reg = dict(schema='r17.resident.v1', id=info['id'], key=path.stem, host=host(launch, campaign),
                       server_node=campaign['server_node'], python=launch.host_python(campaign),
                       server_run=campaign['server_run'], local_dir=info['local_dir'], started_by=campaign['run_name'],
                       signature=signature(launch, campaign), attestations=str(Path(directory)/'SERVER_ATTESTATIONS.json'),
                       local=[{k: e[k] for k in ('label', 'pid', 'pgid', 'start_ticks', 'log')} for e in local],
                       remote=[slim(launch, o) for o in remote_owned], mps=dict(ready=mps_ready) if mps_ready else None,
                       bundles=[], memory=[], batches=[dict(run=campaign['run_name'], directory=str(directory),
                                                            attached_at=info['started_at'])])
        reg['batches'][-1]['released_at'] = now
        reg['bundles'] = sorted(set(reg['bundles']) | {r['bundle_id'] for r in acks['rows']})
        reg.update(status='KEPT', owner=None)
        write_json(path, reg)
        if reg['remote'] and not (reg.get('renewer') and alive(reg['renewer'])):
            reg['renewer'] = renewer(launch, path, reg)
            write_json(path, reg)
        write_json(Path(reg['local_dir'])/'RESIDENT.json', reg)
    report['resident'].update(kept=True, registry=str(path), released_at=now)


def release(reg, campaign, cleanup):
    """An attached campaign that did not keep the resident has stopped it with its own processes."""
    labels = {e['label'] for e in reg['local']+reg['remote']}
    with locked(RESIDENTS/f"{reg['key']}.json"):
        reg['batches'][-1]['released_at'] = time.time()
        archive(reg, 'STOPPED', stopped_by=campaign['run_name'],
                closure=[r for r in cleanup if r.get('label') in labels])


def renew(path, rid):
    from exp.r17_integration.ops import formal_eval as launch
    while True:
        time.sleep(RENEW_SECONDS)
        reg = read(path)
        if not reg or reg['id'] != rid or reg['status'] not in ('KEPT', 'ATTACHED'):
            return
        try:
            status = controls(launch, reg['remote'])
        except Exception as exc:
            print(time.strftime('%m-%d %H:%M:%S'), 'renewal failed:', repr(exc)[-800:], flush=True)
            continue
        if not all(serving(status.get(e['label'])) for e in reg['remote']):
            print(time.strftime('%m-%d %H:%M:%S'), 'a resident process closed; renewer exits:', json.dumps(status)[-2000:], flush=True)
            return


def stop(run=None, path=None, force=False):
    """Chain exit / rollback: stop the registered resident that `run` served from (or the one at `path`)."""
    from exp.r17_integration.ops import formal_eval as launch
    if run is not None:
        campaign = read(Path(run)/'campaign.json')
        path = registry(launch, campaign)
    with locked(path):
        reg = read(path)
        if reg is None or run is not None and campaign['run_name'] not in [b['run'] for b in reg['batches']]:
            return dict(status='NONE', registry=str(path))
        owner = reg.get('owner')
        if owner and alive(owner) and not force:
            raise RuntimeError(f"resident {reg['id']} is attached to the running {owner['run']}; --force stops it under it")
        return dict(status='STOPPED', id=reg['id'], closure=close(launch, reg, 'STOPPED', stopped_by=f'{MODULE} stop'))


def smoke(root):
    """Verification only: shrink a built, unprepared R17 FFK definition to the smoke plan (2 tasks x 1-2 inits per YAML)."""
    from dataclasses import asdict
    from exp.offline_search.rounds.r17.filter_failed_knob.build import cpu_plan
    root = Path(root)
    if any((root/n).exists() for n in ('PREPARED.json', 'LAUNCH.json', 'RECEIPT.json')):
        raise ValueError('smoke shrinks a definition before its preparation')
    campaign = read(root/'campaign.json')
    campaign['smoke'] = True
    write_json(root/'campaign.json', campaign)
    _, model, tasks, _ = cpu_plan(campaign)
    plan = read(root/'EPISODE_PLAN.json')
    plan.update(model=model, tasks=[asdict(t) for t in tasks], expected_episodes=len(tasks))
    write_json(root/'EPISODE_PLAN.json', plan)
    return dict(run=str(root), episodes=len(tasks))


def main(argv=None):
    p = argparse.ArgumentParser(description=__doc__)
    sub = p.add_subparsers(dest='mode', required=True)
    s = sub.add_parser('stop')
    target = s.add_mutually_exclusive_group(required=True)
    target.add_argument('--run', type=Path, help='a campaign that served from the resident')
    target.add_argument('--registry', type=Path)
    s.add_argument('--force', action='store_true')
    r = sub.add_parser('renew')
    r.add_argument('--registry', type=Path, required=True)
    r.add_argument('--id', required=True)
    sub.add_parser('show')
    m = sub.add_parser('smoke')
    m.add_argument('--run', type=Path, required=True)
    a = p.parse_args(argv)
    if a.mode == 'renew':
        return renew(a.registry, a.id)
    if a.mode == 'show':
        for path in sorted(RESIDENTS.glob('*.json')):
            reg = read(path)
            print(json.dumps(dict(registry=str(path), id=reg['id'], status=reg['status'], owner=reg.get('owner'),
                                  campaigns=[b['run'] for b in reg['batches']], yamls=len(reg['bundles']),
                                  server_run=reg['server_run'], memory=reg['memory'][-1:])))
        return
    print(json.dumps(stop(a.run, a.registry, a.force) if a.mode == 'stop' else smoke(a.run)))


if __name__ == '__main__':
    main()
