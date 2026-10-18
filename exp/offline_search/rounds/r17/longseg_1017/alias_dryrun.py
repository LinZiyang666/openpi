"""CPU dry run of one h100-alias lane prepare; every remote and transfer effect is stubbed.

Copies a built, never-prepared campaign directory to a scratch root (the live
paths inside its campaign.json stay read-only inputs), sets or clears
R17_H100_NODE/R17_H100_HOST/R17_DRIVER_PORT for this process and runs the real
filter_failed.stage on the copy. Only process boundaries are replaced: tether
and rsync subprocesses (recorded, canned replies), the rsync daemons and their
global lock, local immutable-release installs (verify only; a missing new
orchestration release is installed under <scratch>/qualification so the
recorded planner can run), worker deployment (no worker-node probes) and the
shared /data reader lock (a private file; the read throttle stays on). Writes under /data/mw_finetune/{eval,qualification}
raise, and the live campaign directory must stay byte-identical.

  python -m exp.offline_search.rounds.r17.longseg_1017.alias_dryrun run --name N --scratch S [--native]
  python -m exp.offline_search.rounds.r17.longseg_1017.alias_dryrun compare --name N --scratch S
"""
from __future__ import annotations

import argparse
from collections import Counter
import hashlib
import json
import os
from pathlib import Path
import shutil
import subprocess
import sys
from types import SimpleNamespace
from unittest.mock import patch

EVAL = Path('/data/mw_finetune/eval')
LIVE = ('/data/mw_finetune/eval/', '/data/mw_finetune/qualification/', '/data/r17_library/')
ALIAS = dict(R17_H100_NODE='h100-temp', R17_H100_HOST='149.165.172.126', R17_DRIVER_PORT='23194')
REMOTE = ('tether', 'rsync', 'ssh', 'scp')


def read(path):
    return json.loads(Path(path).read_text())


def sha(path):
    h = hashlib.sha256()
    with open(path, 'rb') as stream:
        for block in iter(lambda: stream.read(1 << 20), b''):
            h.update(block)
    return h.hexdigest()


def stamps(root):
    return {str(p.relative_to(root)): (p.stat().st_size, p.stat().st_mtime_ns)
            for p in sorted(Path(root).rglob('*')) if p.is_file()}


def forbidden(path):
    try:
        return os.path.abspath(os.fspath(path)).startswith(LIVE)
    except TypeError:
        return False


def hosts(value, path=''):
    """Every string that names a public H100 address, with its JSON path."""
    if isinstance(value, dict):
        return [h for k, v in value.items() for h in hosts(v, path+'/'+k)]
    if isinstance(value, list):
        return [h for i, v in enumerate(value) for h in hosts(v, path+'['+str(i)+']')]
    return [(path, value)] if isinstance(value, str) and '149.165.' in value else []


def guards(stack):
    """Refuse live writes and real remote subprocesses for the whole run."""
    import builtins, io
    def opened(native):
        def call(file, mode='r', *a, **kw):
            if isinstance(file, (str, os.PathLike)) and any(c in mode for c in 'wax+') and forbidden(file):
                raise AssertionError('dry run attempted a live write: '+str(file))
            return native(file, mode, *a, **kw)
        return call
    stack.enter_context(patch.object(builtins, 'open', opened(builtins.open)))
    stack.enter_context(patch.object(io, 'open', opened(io.open)))
    for name in ('rename', 'replace', 'link', 'symlink', 'mkdir', 'unlink', 'remove', 'rmdir', 'chmod'):
        native = getattr(os, name)
        def call(*a, _native=native, _name=name, **kw):
            targets = a[:2] if _name in ('rename', 'replace') else a[1:2] if _name in ('link', 'symlink') else a[:1]
            if _name == 'mkdir' and os.path.exists(a[0]):
                return _native(*a, **kw)  # raises FileExistsError, writes nothing
            if any(forbidden(t) for t in targets):
                raise AssertionError('dry run attempted a live '+_name+': '+str(targets))
            return _native(*a, **kw)
        stack.enter_context(patch.object(os, name, call))
    native_popen = subprocess.Popen
    class Popen(native_popen):
        def __init__(self, args, *a, **kw):
            words = [args] if isinstance(args, (str, bytes)) else list(args)
            if words and Path(str(words[0]).split()[0]).name in REMOTE:
                raise AssertionError('dry run attempted a real remote subprocess: '+str(words[:8]))
            super().__init__(args, *a, **kw)
    stack.enter_context(patch.object(subprocess, 'Popen', Popen))


def run(name, scratch, alias):
    for key in ALIAS:
        os.environ.pop(key, None)
    if alias:
        os.environ.update(ALIAS)
    variant = 'alias' if alias else 'native'
    live, root = EVAL/name, Path(scratch)/variant/name
    if any((live/n).exists() for n in ('PREPARED.json', 'LAUNCH.json', 'RECEIPT.json')):
        raise ValueError('choose a built, never-prepared campaign')
    if root.exists():
        shutil.rmtree(root)
    root.parent.mkdir(parents=True, exist_ok=True)
    before = stamps(live)
    pins = {n: sha(live/n) for n in ('campaign.json', 'EPISODE_PLAN.json')}
    # Bounded local copy (16 MiB/s) without the large pre-rebind backups.
    subprocess.run(['rsync', '-a', '--bwlimit=16384', *[f'--exclude={n}.json.*' for n in
                    ('EPISODE_PLAN', 'campaign', 'FFK_DEFINITION')], str(live)+'/', str(root)+'/'], check=True)
    from contextlib import ExitStack
    from exp.r17_integration.ops import direct_transfer as dt, driver_release as dr, formal_eval as f
    from exp.r17_integration.ops import formal_ports, immutable_release as ir
    from exp.offline_search.rounds.r17.filter_failed import stage as st, worker_resilience as wr
    from exp.offline_search.rounds.r17.longseg_1017 import io_budget
    calls, installs, admissions = [], [], []
    native_execute, native_verify, native_install, native_prepare = f.execute, ir.verify, ir.install, dr.prepare
    redirected = {}

    def answer(node, command):
        text = ' '.join(command)
        if command[:2] == ['mkdir', '-p']:
            return ''
        if command and command[0] == 'rsync':
            return ''
        if 'json.dumps(daemon(' in text:
            return json.dumps(dict(pid=1, root='/stub'))
        if 'json.dumps(close_daemon(' in text:
            return 'null'
        if 'present=d is not None' in text:
            return json.dumps(dict(present=False, installed='', reused=False))
        if any('release_install.py' in w for w in command):
            return json.dumps(dict(present=True, reused=False))
        if 'install_snapshot(' in text:
            return ''
        if 'm.get("digest")' in text:
            # As if the coordinator's copy already holds the sealed source tree: the
            # real stage would otherwise snapshot it (a new .eval.lock in that live
            # source directory) and push the archive with the same alias rsync.
            import re
            local = read(Path(re.search(r"p=Path\('([^']+)'\)", text).group(1))/'manifest.json')
            return json.dumps(dict(digest=local['digest'], files=local['files'], present=True))
        raise AssertionError('unexpected remote call on '+node+': '+text[:300])

    def execute(argv, seconds=60, env=None, cwd=None):
        argv = list(map(str, argv))
        program = Path(argv[0]).name
        if program == 'tether':
            i = argv.index('--')
            node, command = argv[i-1], argv[i+1:]
            url = next((w for w in command if w.startswith('rsync://')), None)
            what = ('rsync '+url if url else ' '.join(command)[:120])
            calls.append(dict(kind='tether', node=node, command=what))
            return answer(node, command)
        if program == 'rsync':
            url = next(w for w in argv if w.startswith('rsync://'))
            calls.append(dict(kind='rsync', host=url.split('//', 1)[1].split(':', 1)[0], url=url,
                              local=[w for w in argv[-2:] if not w.startswith('rsync://')][0]))
            return ''
        assert program not in REMOTE, argv
        return native_execute(argv, seconds=seconds, env=env, cwd=cwd)

    def daemon(root, module, port, source=None):
        return dict(pid=1, root=str(root))

    def close_daemon(server, destination=None):
        return None

    def install(archive, release, archive_sha, manifest_identity):
        found = native_verify(release, 'RUNTIME_RELEASE.json', manifest_identity)
        installs.append(dict(root=str(release), present_locally=found['present']))
        if found['present']:
            return found
        if Path(release).name.startswith('formal_driver_'):
            # The planner must import a new orchestration release: install it in scratch.
            redirected[str(release)] = str(Path(scratch)/'qualification'/Path(release).name)
            return native_install(archive, redirected[str(release)], archive_sha, manifest_identity)
        return dict(present=True, installed=str(release), reused=False, stub=True)

    def prepare_driver(campaign, directory, checkout):
        native_prepare(campaign, directory, checkout)
        if campaign['driver_runtime'] in redirected:
            campaign['orchestration']['driver_runtime'] = campaign['driver_runtime'] = redirected[campaign['driver_runtime']]
            (Path(directory)/'ORCHESTRATION.json').write_text(json.dumps(campaign['orchestration'], indent=2, sort_keys=True)+'\n')

    def admission(campaign):
        admissions.append(campaign['driver_port'])
        if alias:
            return native_admission(campaign)  # real local bind of the free lane port
    native_admission = formal_ports.prepare_admission

    def workers(campaign, directory, build):
        """Sealed source workers.tar; no worker-node probe, install or admission."""
        shutil.copyfile(Path(campaign['filter_failed_source'])/'workers.tar', Path(directory)/'workers.tar')
        deployment = {}
        for node in campaign.get('execution_worker_nodes', campaign['worker_nodes']):
            deployment[node] = dict(stage=f.worker_paths(campaign, node)['stage'])
            (Path(directory)/(node+'.json')).write_text(json.dumps(dict(campaign, worker_node=node)))
            (Path(directory)/(node+'_MEMORY_ADMISSION.json')).write_text(json.dumps(dict(admitted=True, stub=True)))
        return deployment

    with ExitStack() as stack:
        guards(stack)
        for target, attribute, value in (
                (f, 'execute', execute), (dt, 'daemon', daemon), (dt, 'close_daemon', close_daemon),
                (dt, 'fcntl', SimpleNamespace(flock=lambda *a: None, LOCK_EX=0)),
                (ir, 'install', install), (dr, 'install', install), (dr, 'prepare', prepare_driver),
                (formal_ports, 'prepare_admission', admission),
                (wr, 'prepare', workers), (io_budget, 'LOCK', Path(scratch)/'reader.lock')):
            stack.enter_context(patch.object(target, attribute, value))
        st.stage(root)
        c = read(root/'campaign.json')
        launch_view = dict(c)
        f.open_tunnels(launch_view, root)  # what launch_prepared recomputes before dispatch
        launch_view = {k: launch_view[k] for k in ('endpoint_host', 'endpoint', 'driver_host', 'driver_public_port')}
        launch_view['backend_control_host'] = f.backend_control_host(c)
        f.check_execution_host(c)
    old, new = read(live/'EPISODE_PLAN.json'), read(root/'EPISODE_PLAN.json')
    strip = lambda t: {k: v for k, v in t.items() if k not in ('server_host', 'server_port')}
    result = dict(
        name=name, variant=variant, env={k: os.environ.get(k) for k in ALIAS}, scratch=str(root),
        tether_nodes=dict(Counter(r['node'] for r in calls if r['kind'] == 'tether')),
        rsync_daemon_hosts=dict(Counter(r['host'] for r in calls if r['kind'] == 'rsync')),
        server_transfers=[r['local'] for r in calls if r['kind'] == 'rsync'],
        remote_calls=calls, local_release_installs=installs, driver_release_in_scratch=redirected,
        local_port_admission=admissions,
        campaign={k: c.get(k) for k in ('server_node', 'execution_host', 'endpoint_host', 'endpoint', 'endpoint_port',
                                         'driver_host', 'driver_port', 'driver_public_port', 'ports', 'backend_host',
                                         'server_campaign', 'runtime', 'runtime_identity')},
        campaign_h100_addresses=hosts(c), launch_view=launch_view,
        rebind_record=c.get('execution_host_rebind'),
        transport_rebind_kept=c.get('longseg_1017', {}).get('transport_rebind') == read(live/'campaign.json').get('longseg_1017', {}).get('transport_rebind'),
        plan=dict(tasks=len(new['tasks']), endpoint=new.get('endpoint'),
                  routing=Counter(f"{t['server_host']}:{t['server_port']}" for t in new['tasks']).most_common(),
                  task_uid_order_equal=[t['task_uid'] for t in old['tasks']] == [t['task_uid'] for t in new['tasks']],
                  non_routing_fields_equal=len(old['tasks']) == len(new['tasks']) and all(
                      strip(a) == strip(b) for a, b in zip(old['tasks'], new['tasks'])),
                  plan_fields_equal_except_endpoint={k: v for k, v in old.items() if k not in ('tasks', 'endpoint')} ==
                                                   {k: v for k, v in new.items() if k not in ('tasks', 'endpoint')},
                  bytes_equal_to_live=sha(root/'EPISODE_PLAN.json') == pins['EPISODE_PLAN.json'],
                  backups=sorted(p.name for p in root.glob('EPISODE_PLAN.json.before_alias_*'))),
        longseg_runtime_fix=(lambda d: dict(present=True, members=sorted(d['members']), identity=d['identity']))(
            read(root/'LONGSEG_RUNTIME_FIX.json')) if (root/'LONGSEG_RUNTIME_FIX.json').exists() else dict(present=False),
        prepared=read(root/'PREPARED.json').get('status'), preflight_ports=read(root/'PREFLIGHT.json')['ports'],
        live_unchanged=stamps(live) == before and {n: sha(live/n) for n in pins} == pins)
    (Path(scratch)/f'{name}.{variant}.json').write_text(json.dumps(result, indent=1, default=str)+'\n')
    return result


def compare(name, scratch):
    """Alias versus native prepare of the same campaign: which bytes differ."""
    a, n = Path(scratch)/'alias'/name, Path(scratch)/'native'/name
    ca, cn = read(a/'campaign.json'), read(n/'campaign.json')
    keys = sorted(k for k in set(ca) | set(cn) if ca.get(k) != cn.get(k))
    longseg = sorted(k for k in set(ca.get('longseg_1017', {})) | set(cn.get('longseg_1017', {}))
                     if ca.get('longseg_1017', {}).get(k) != cn.get('longseg_1017', {}).get(k))
    pa, pn = read(a/'EPISODE_PLAN.json'), read(n/'EPISODE_PLAN.json')
    fields = Counter(k for x, y in zip(pa['tasks'], pn['tasks']) for k in set(x) | set(y) if x.get(k) != y.get(k))
    files = sorted(p for p in {str(q.relative_to(a)) for q in a.rglob('*') if q.is_file()} &
                   {str(q.relative_to(n)) for q in n.rglob('*') if q.is_file()}
                   if not p.endswith(('.tar', '.lock')) and sha(a/p) != sha(n/p))
    result = dict(name=name, campaign_keys_differing=keys, longseg_1017_keys_differing=longseg,
                  values={k: dict(alias=ca.get(k), native=cn.get(k)) for k in keys if k not in ('longseg_1017',)},
                  plan_tasks=len(pa['tasks']), task_fields_differing=dict(fields),
                  plan_top_level_differing=sorted(k for k in set(pa) | set(pn) if k != 'tasks' and pa.get(k) != pn.get(k)),
                  common_files_differing=files)
    (Path(scratch)/f'{name}.compare.json').write_text(json.dumps(result, indent=1, default=str)+'\n')
    return result


if __name__ == '__main__':
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument('action', choices=('run', 'compare'))
    p.add_argument('--name', required=True)
    p.add_argument('--scratch', type=Path, required=True)
    p.add_argument('--native', action='store_true', help='leave the alias variables unset')
    a = p.parse_args()
    out = run(a.name, a.scratch, not a.native) if a.action == 'run' else compare(a.name, a.scratch)
    print(json.dumps({k: v for k, v in out.items() if k != 'remote_calls'}, indent=1, default=str))
