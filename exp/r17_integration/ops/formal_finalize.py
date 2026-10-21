"""CPU-only evidence sealing after receipt-verified GPU/fleet release.

PENDING_EVIDENCE is neither a paired control nor a resume source. A successful
launcher exit can hand off this CPU phase; wait-seal is the scientific barrier.
No function here starts or signals a model, worker, proxy or MPS process.
"""
from __future__ import annotations

import argparse
from datetime import datetime, timezone
import fcntl
import inspect
import io
import json
import os
from pathlib import Path
import subprocess
import tarfile
import time

from .formal_process import alive, ticks
from .build_client_bundle import sha256
from .immutable_release import archive_manifest, identity, install, sha, verify
from .sync_back import atomic_bytes, install_snapshot, snapshot_run

MAX_BYTES = 8 * 1024**3
OVERLAY = ('formal_finalize', 'formal_eval', 'formal_prestage', 'formal_process',
           'formal_runtime', 'formal_resume', 'driver_release', 'tether_retry',
           'evidence_transfer', 'transport_operations', 'sync_back', 'formal_ports','local_workers',
           'worker_storage','worker_disk','timan1_workers','incremental_sync','direct_transfer')


def read(path):
    return json.loads(Path(path).read_text())


def durable_json(path, value):
    atomic_bytes(path, (json.dumps(value, sort_keys=True, indent=2, allow_nan=False)+'\n').encode())


def record_close_gap(report, node, row):
    """A completed driver may defer a closed fleet's abnormal snapshot exit."""
    completion = report.get('driver_completion', {})
    if (completion.get('verified_completion') is not True or completion.get('accepted_episodes', 0) <= 0 or
            row.get('status') != 'CLOSED' or type(row.get('exit_code')) is not int or
            any(row.get(k) is not False for k in ('alive','child_alive','supervisor_alive')) or
            row.get('owned_worker_survivors') != []):
        pass
    report.setdefault('worker_close_gap', {})[node] = dict(
        phase='after_verified_driver_completion', exit_code=row['exit_code'],
        snapshot_verification_required=True, fleet_verified_gone=True)


def completed_close_error(directory, report):
    """Defer exactly the post-driver close failure; never general failures."""
    if report.get('error') != "RuntimeError('remote worker close/snapshot incomplete')":
        return False
    from .formal_eval import driver_completion
    completion = report.get('driver_completion', {})
    if (completion.get('verified_completion') is not True or report.get('all_owned_closed') is not True or
            any(report.get(k) for k in ('worker_fleet_failure','batching_audit_error','shard_transfer_error',
                                       'finalization_error')) or
            driver_completion(directory, completion['returncode']) != completion):
        pass
    for row in report.get('cleanup', []):
        if row.get('kind',row.get('label')) == 'workers':
            record_close_gap(report, row.get('fleet_key',row['node']), row)
    closed_proof(report)
    report['recovered_close_error'] = report.pop('error')
    return True


def closed_proof(report):
    """Require every recorded owner, not just the aggregate boolean, closed."""
    if report.get('all_owned_closed') is not True or not report.get('cleanup'):
        pass
    if any(row.get('closure_confirmed') is not True for row in report['cleanup']):
        raise ValueError('owned cleanup contains unconfirmed closure')
    for owned in report.get('owned_local', []):
        if alive(owned) or not any(row.get('node') is None and row.get('label') == owned['label']
                                   for row in report['cleanup']):
            raise ValueError('owned local process has not closed')
    fleets = {}
    for owned in report.get('owned_remote', []):
        rows = [row for row in report['cleanup'] if row.get('node') == owned['node'] and
                row.get('label') == owned['label']]
        if len(rows) != 1 or rows[0].get('status') != 'CLOSED' or rows[0].get('alive') is not False:
            raise ValueError('owned remote process closure missing')
        if rows[0].get('child_alive') is not False or rows[0].get('supervisor_alive') is not False:
            raise ValueError('owned remote child/supervisor closure missing')
        if owned.get('kind',owned['label']) == 'workers':
            row = rows[0]
            key = owned.get('fleet_key',owned['node'])
            if owned.get('retired_worker_node') and owned['node'] in report.get('worker_node_losses',{}):
                if (report.get('driver_completion',{}).get('verified_completion') is not True or
                        row.get('owned_worker_survivors') != []):
                    raise ValueError('retired worker requires full central evidence and proven closure')
                # Accepted per-step rows/results are durable in the driver.
                # A reboot can destroy optional worker-side diagnostics.
                continue
            allowed = (0, 2) if report.get('driver_completion', {}).get('recovered_worker_shutdown') else (0,)
            gap = report.get('worker_close_gap', {}).get(key, {})
            deferred = (gap.get('phase') == 'after_verified_driver_completion' and
                gap.get('exit_code') == row.get('exit_code') and gap.get('snapshot_verification_required') is True and
                gap.get('fleet_verified_gone') is True and
                report.get('driver_completion', {}).get('verified_completion') is True)
            if row.get('owned_worker_survivors') != [] or row.get('exit_code') not in allowed and not deferred:
                raise ValueError('owned fleet has survivors or an abnormal exit')
            if key in fleets:pass
            fleets[key] = row
    return fleets


def shutdown_gap(directory, report):
    completion = report.get('driver_completion', {})
    fleets = closed_proof(report)
    if not completion.get('recovered_worker_shutdown') and not report.get('worker_close_gap'):
        return None
    from .formal_eval import driver_completion
    if driver_completion(directory, completion['returncode']) != completion or not fleets:
        pass
    raw = read(Path(directory)/'driver/DRIVER_RECEIPT.json')
    return dict(schema='r17.worker_shutdown_gap.v1', recovered=True,
                worker_shutdown=raw.get('worker_shutdown'), worker_close_gap=report.get('worker_close_gap', {}),
                driver_returncode=completion['returncode'],
                accepted_episodes=completion['accepted_episodes'],
                error_attempts=completion['error_attempts'], resolved_retry_errors=completion['resolved_retry_errors'],
                fleet_closure={node:dict(exit_code=row['exit_code'], alive=False, child_alive=False,
                    supervisor_alive=False, owned_worker_survivors=[]) for node,row in fleets.items()})


def validate_fleet(fleet, gap):
    allowed = ('PASS','PARTIAL') if gap else ('PASS',)
    if (fleet.get('status') not in allowed or fleet.get('error') is not None or
            fleet.get('status') == 'PARTIAL' and fleet.get('reason') != 'interrupted'):
        pass
    failures = fleet.get('worker_failures', [])
    if any(not any(p.get('index') == f.get('index') and p.get('generation', -1) > f.get('generation', -1)
                   for p in fleet.get('pids', [])) for f in failures):
        pass
    return failures


def closed_worker_source(run, owners, manifest_sha, gap):
    """Read-only PID/start-tick and native fleet checks before rebuilding a tar."""
    run = Path(run)
    if not owners or any(alive(owner) for owner in owners):
        pass
    if sha256(run/'manifest.json') != manifest_sha:
        pass
    fleet = json.loads((run/'receipt.json').read_text())
    owner = fleet.get('fleet_owner')
    if not owner or not any(owner.get('pid') == o['pid'] and owner.get('start_ticks') == o['start_ticks'] for o in owners):
        pass
    if any(alive(entry) for entry in fleet.get('pids', [])):
        pass
    validate_fleet(fleet, gap)


def worker_snapshot(campaign, directory, node, item, closure, gap):
    """Rebuild only a verified gone campaign's missing snapshot, by value."""
    from . import formal_eval as launch
    key = Path(item['worker_run']).name
    archive = item['stage']+'/final_evidence_workers.tar.gz'
    manifest = Path(directory)/(node+'_launch.json')
    if not manifest.exists():
        manifest = Path(directory)/(node+'.json')
    owners = [closure[k] for k in ('child','supervisor') if k in closure]
    prelude = ('from pathlib import Path; import fcntl, hashlib, json, os, tarfile; '
        'from exp.r17_integration.ops.build_client_bundle import add_bytes,json_bytes,safe_name,sha256; '
        'from exp.r17_integration.ops.formal_process import alive; '
        'from exp.r17_integration.qualification import archive_stage; '
        'archive_stage.MAX_BYTES='+str(MAX_BYTES)+'\n')
    code = prelude+inspect.getsource(validate_fleet)+inspect.getsource(closed_worker_source)+inspect.getsource(snapshot_run)+(
        '\nclosed_worker_source('+repr(item['worker_run'])+','+repr(owners)+','+repr(sha(manifest))+','+repr(bool(gap))+'); '
        'r=snapshot_run('+repr(item['worker_run'])+','+repr(archive)+',compressed=True); '
        'r.update(archive_stage.stage_archive('+repr(archive)+','+repr(key)+',"workers",r["archive_sha256"])); '
        'r.update(compressed=True,formal_max_bytes='+str(MAX_BYTES)+',rebuilt_closed_snapshot=True); print(json.dumps(r))')
    value = json.loads(launch.remote(item.get('node',node), ['env','PYTHONPATH='+launch.pythonpath(item['payload']),
        item['python'],'-c',code],seconds=300,replay_safe=True))
    durable_json(Path(directory)/(node+'_REBUILT_SNAPSHOT.json'),value)
    return value


def server_snapshot(campaign, directory):
    """Send the compressed snapshot builder by value; native runtime is intact."""
    from . import formal_eval as launch
    key = campaign['run_name']+'_servers'
    archive = '/tmp/r17_formal/'+campaign['run_name']+'/server/final_evidence_servers.tar.gz'
    prelude = ('from pathlib import Path; import fcntl, hashlib, json, os, tarfile; '
        'from exp.r17_integration.ops.build_client_bundle import add_bytes,json_bytes,safe_name,sha256; '
        'from exp.r17_integration.qualification import archive_stage; '
        'archive_stage.MAX_BYTES='+str(MAX_BYTES)+'\n')
    code = prelude+inspect.getsource(snapshot_run)+(
        '\nr=snapshot_run('+repr(campaign['server_run'])+','+repr(archive)+',compressed=True); '
        'r.update(archive_stage.stage_archive('+repr(archive)+','+repr(key)+',"workers",r["archive_sha256"])); '
        'r.update(compressed=True,formal_max_bytes='+str(MAX_BYTES)+'); print(json.dumps(r))')
    return json.loads(launch.remote('h100', ['env','PYTHONPATH='+launch.pythonpath(campaign['runtime']),
        launch.host_python(campaign),'-c',code], seconds=300, replay_safe=True))


def pull_snapshot(node, value, destination, *, key, python, root, deadline, physical_node=None):
    from . import formal_eval as launch
    from .transport_operations import cleanup
    from exp.r17_integration.qualification.archive_stage import stage_identity
    if stage_identity(value['archive'], key, 'workers') != value['archive_sha256']:
        pass
    archive = destination.parent.parent/(node+'.tar') if node != 'h100_servers' else destination.parent/'servers.tar'
    source_node = physical_node or ('h100' if node == 'h100_servers' else node)
    def heartbeat():
        if time.monotonic() >= deadline:
            raise TimeoutError('CPU evidence sealing deadline exceeded')
    if not archive.exists():
        launch.transfer(['rsync','pull',source_node+':'+value['archive'],str(archive)], heartbeat=heartbeat)
    if 'archive_size_bytes' not in value:
        size = json.loads(launch.remote(source_node,[python,'-c','import json; from pathlib import Path; '
            'print(json.dumps(Path('+repr(value['archive'])+').stat().st_size))'],replay_safe=True))
        value['archive_size_bytes'] = size
    if archive.stat().st_size != value['archive_size_bytes']:
        pass
    verified = install_snapshot(archive, destination, expected_sha256=value['archive_sha256'])
    if verified['digest'] != value['manifest']['digest'] or verified['partials']:
        pass
    cleaned = cleanup(source_node,value['archive'],key,value['archive_sha256'],python=python,
        env=['PYTHONPATH='+launch.pythonpath(root)],remote=launch.remote,
        max_bytes=MAX_BYTES if node == 'h100_servers' or value.get('formal_max_bytes') == MAX_BYTES else None)
    return dict(remote=value,verified=verified,cleaned=cleaned,
                archive_sha256=sha(archive),after_gpu_release=True)


def native_batch_audit(campaign, directory):
    """Audit the resident evidence with the campaign's exact native code."""
    from . import formal_eval as launch
    root = campaign['runtime']
    if not verify(root,'RUNTIME_RELEASE.json',campaign['runtime_identity'])['present']:
        raise ValueError('frozen native audit runtime missing')
    if campaign.get('cache_study'):
        module, function = 'exp.r17_integration.ops.cache_study', 'audit_batches'
    else:
        module, function = 'exp.r17_integration.models.fastwam.formal_batching', 'audit_decisions'
    code = ('import json; from pathlib import Path; from '+module+' import '+function+'; '
        'c=json.loads(Path('+repr(str(directory/'campaign.json'))+').read_text()); '
        'print(json.dumps('+function+'(c,Path('+repr(str(directory/'servers'))+'))))')
    env = dict(os.environ,CUDA_VISIBLE_DEVICES='',JAX_PLATFORMS='cpu',PYTHONDONTWRITEBYTECODE='1',
        PYTHONPATH=launch.pythonpath(root),OMP_NUM_THREADS='1',OPENBLAS_NUM_THREADS='1')
    result = json.loads(launch.execute([launch.CONTROL_PYTHON,'-c',code],seconds=300,env=env,cwd=root))
    if result.get('status') != 'PASS' or not campaign.get('cache_study') and campaign.get('fastwam_batching') and (
            result.get('treatment_identity') != campaign['fastwam_batch_treatment_identity']):
        raise ValueError('native batching audit failed')
    return result


def finalize(campaign, directory, deployment, report, *, retain_worker_storage=False):
    """No process controls: only closed snapshots, verification and publication."""
    from . import formal_eval as launch
    directory = Path(directory)
    fleets = closed_proof(report)
    deadline = time.monotonic()+7200
    parity = campaign.get('native_parity_only')
    if not parity:
        completion = launch.driver_completion(directory,report['driver_completion']['returncode'])
        if completion != report['driver_completion']:
            pass
        gap = shutdown_gap(directory,report)
        if gap:
            report['worker_shutdown_gap'] = gap
    report.update(status='PENDING_EVIDENCE',evidence_pending=True,gpu_released=True)
    durable_json(directory/'RECEIPT.json',report)
    for node in fleets:
        item = deployment[node]
        old = report.get('transfers', {}).get(node)
        if old:
            value = old['remote']
            verified = install_snapshot(directory/(node+'.tar'),directory/'workers'/node,
                                         expected_sha256=value['archive_sha256'])
            if verified['digest'] != value['manifest']['digest'] or verified['partials'] or value['manifest'].get('partials'):
                pass
            old.update(verified=verified,archive_sha256=sha(directory/(node+'.tar')))
            if 'archive_size_bytes' in value and (directory/(node+'.tar')).stat().st_size != value['archive_size_bytes']:
                pass
        else:
            value = json.loads(launch.remote(item.get('node',node),[item['python'],'-c',
                'from pathlib import Path; p=Path('+repr(item['worker_run']+'.transfer.json')+'); '
                'print(p.read_text() if p.exists() else "null")'],
                replay_safe=True))
            if value is None:
                rebuilt = directory/(node+'_REBUILT_SNAPSHOT.json')
                value = read(rebuilt) if rebuilt.exists() else worker_snapshot(
                    campaign,directory,node,item,fleets[node],report.get('worker_shutdown_gap'))
            options=dict(physical_node=item['node']) if item.get('node') else {}
            old = pull_snapshot(node,value,directory/'workers'/node,key=Path(item['worker_run']).name,
                                python=item['python'],root=item['payload'],deadline=deadline,**options)
        worker_manifest = directory/(node+'_launch.json')
        if not worker_manifest.exists():
            worker_manifest = directory/(node+'.json')
        if value['manifest']['files'].get('manifest.json') != sha(worker_manifest):
            pass
        fleet = read(directory/'workers'/node/'receipt.json')
        from .incremental_sync import restore as restore_incremental
        old['incremental_sync'] = restore_incremental(directory,node)
        failures = validate_fleet(fleet, report.get('worker_shutdown_gap'))
        if failures:
            report.setdefault('resolved_worker_failures',{})[node] = failures
        report.setdefault('transfers',{})[node] = old
        durable_json(directory/'RECEIPT.json',report)
    if campaign['server_node'] == 'h100':
        old = report.get('transfers',{}).get('h100_servers')
        if old:
            verified = install_snapshot(directory/'servers.tar',directory/'servers',
                expected_sha256=old['remote']['archive_sha256'])
            if verified['partials'] or verified['digest'] != old['remote']['manifest']['digest']:
                pass
        else:
            value = server_snapshot(campaign,directory)
            report.setdefault('transfers',{})['h100_servers'] = pull_snapshot('h100_servers',value,
                directory/'servers',key=campaign['run_name']+'_servers',python=launch.host_python(campaign),
                root=campaign['runtime'],deadline=deadline)
        durable_json(directory/'RECEIPT.json',report)
    report['backend_evidence'] = launch.backend_evidence(directory)
    if campaign.get('fastwam_batching') or campaign.get('cache_study') and campaign.get('dp_batching'):
        report['batching_audit'] = native_batch_audit(campaign,directory)
    if not parity:
        aggregate_rc = report['driver_completion']['returncode']
        if campaign.get('resume_from'):
            from .formal_resume import merge
            if campaign.get('driver_mode') == 'group-commit':
                from .driver_release import preserve_resume_segment
                report['durable_resume_segment'] = preserve_resume_segment(directory)
            report['resume'] = merge(campaign,directory,report['driver_completion'])
            if campaign.get('driver_mode') == 'group-commit':
                from .driver_release import sync_resume_merge
                sync_resume_merge(directory)
            aggregate_rc = 0
        summary = launch.aggregate(directory,driver_returncode=aggregate_rc)
        report['aggregate_status'] = summary['status']  # recorded only; an evidence gap never fails the run
        if report.get('worker_shutdown_gap'):
            summary['worker_shutdown_gap'] = report['worker_shutdown_gap']
            durable_json(directory/'SUMMARY.json',summary)
        if campaign['smoke'] and (len(report['backend_evidence']) < min(2,campaign['replicas']) or
                len(summary['accepted_worker_ids']) < min(2,campaign['workers'])):
            raise ValueError('smoke did not exercise two workers and two backends')
        report['summary'] = str(directory/'SUMMARY.json')
        if campaign.get('cache_study'):
            report['segment_resources']=summary['segment_resources']
    if campaign.get('live_fleets'):
        from .live_fleets import validate_seal
        validate_seal(directory,deployment,report)
    report.update(status='PASS',evidence_pending=False,evidence_verified=True,
                  sealed_at=datetime.now(timezone.utc).isoformat())
    report.pop('server_transfer_error',None)
    # Immutable scientific receipt seals the verified evidence BEFORE deletion.
    # Keep the public barrier PENDING until cleanup has finished; frozen queue
    # waiters that only understand PASS/PENDING must never pin an interim PASS.
    scientific_seal = directory/'SEALED_EVIDENCE_RECEIPT.json'
    durable_json(scientific_seal,report)
    report.update(status='PENDING_EVIDENCE',evidence_pending=True,worker_storage_cleanup_pending=True,
                  scientific_evidence_seal=dict(path=str(scientific_seal),sha256=sha(scientific_seal)))
    durable_json(directory/'RECEIPT.json',report)
    try:
        if retain_worker_storage:
            storage_cleanup = None
        else:
            from .worker_storage import cleanup as storage_cleanup
    except ModuleNotFoundError as exc:
        # A launcher imported before this change may carry the old overlay
        # list into its new CPU release. Preserve its evidence, never break
        # scientific completion or read unpinned code from the working tree.
        if exc.name != 'exp.r17_integration.ops.worker_storage':
            raise
        report['worker_storage_cleanup'] = {node:dict(status='RETAINED',
            reason='legacy CPU overlay lacks worker_storage; remote evidence retained') for node in fleets}
    else:
        if storage_cleanup is None:
            report['worker_storage_cleanup'] = {node:dict(status='RETAINED',
                reason='CPU refinalization retains closed remote diagnostics') for node in fleets}
        else:
            storage_cleanup(campaign,directory,deployment,report)
    report.update(status='PASS',evidence_pending=False,worker_storage_cleanup_pending=False)
    durable_json(directory/'RECEIPT.json',report)
    return report


def pin_release(campaign, directory):
    """A separate CPU orchestration archive; resident/model/worker pins unchanged."""
    directory = Path(directory)
    archive = directory/'evidence_runtime.tar'
    base = archive_manifest(directory/'runtime.tar')
    if identity(base) != campaign['runtime_identity']:
        pass
    with tarfile.open(directory/'runtime.tar') as source,tarfile.open(archive,'w') as target:
        files = {m.name:source.extractfile(m).read() for m in source.getmembers()}
        overlay=(*OVERLAY,'cache_study','cache_resume') if campaign.get('cache_study') else OVERLAY
        if campaign.get('live_fleets'):overlay=(*overlay,'live_fleets','live_fleet_worker')
        for name in overlay:
            key = 'exp/r17_integration/ops/'+name+'.py'
            files[key] = Path(__file__).with_name(name+'.py').read_bytes()
        for name,data in sorted(files.items()):
            member = tarfile.TarInfo(name);member.size=len(data);member.mode=0o444
            target.addfile(member,io.BytesIO(data))
    release = archive_manifest(archive)
    root = '/data/mw_finetune/qualification/formal_evidence_'+identity(release)
    install(archive,root,sha(archive),identity(release))
    pin = dict(schema='r17.evidence_orchestration.v1',root=root,identity=identity(release),
        native_runtime_identity=campaign['runtime_identity'],archive_sha256=sha(archive),
        overlay_files={name:release['files'][name] for name in release['files']
            if base['files'].get(name) != release['files'][name]},cpu_only=True,
        model_server_worker_pins_unchanged=True)
    durable_json(directory/'EVIDENCE_ORCHESTRATION.json',pin)
    return pin


def handoff(campaign, directory, deployment, report):
    from . import formal_eval as launch
    directory = Path(directory)
    closed_proof(report)
    if launch.driver_completion(directory,report['driver_completion']['returncode']) != report['driver_completion']:
        pass
    gap = shutdown_gap(directory,report)
    if gap:
        report['worker_shutdown_gap'] = gap
    pin = pin_release(campaign,directory)
    report.update(status='PENDING_EVIDENCE',evidence_pending=True,gpu_released=True,
                  evidence_orchestration=pin)
    durable_json(directory/'RECEIPT.json',report)
    pins = {name:sha(directory/name) for name in ('campaign.json','EPISODE_PLAN.json','RECEIPT.json',
        'SERVER_ATTESTATIONS.json','WORKER_DEPLOYMENT.json','driver/DRIVER_RECEIPT.json',
        'driver/journal.jsonl','driver/attempts.jsonl','driver/per_step.jsonl')}
    if campaign.get('worker_node_failover'):
        pins['WORKER_NODE_STATE.json']=sha(directory/'WORKER_NODE_STATE.json')
    for node in deployment:
        name = node+'_launch.json' if (directory/(node+'_launch.json')).exists() else node+'.json'
        pins[name] = sha(directory/name)
    if campaign.get('live_fleets'):
        for name in ('LIVE_FLEETS.json','FLEET_EVENTS.jsonl','LIVE_FLEET_CAPABILITY.json'):
            pins[name]=sha(directory/name)
        for path in sorted((directory/'fleets').rglob('*')):
            if path.is_file():pins[str(path.relative_to(directory))]=sha(path)
    for path in sorted((directory/'assets').rglob('*')):
        if path.is_file():
            pins[str(path.relative_to(directory))] = sha(path)
    for name in ('runtime.tar','workers.tar','LAUNCH_WORKER_DEPLOYMENT.json','FULL_EPISODE_PLAN.json','RESUME.json'):
        if (directory/name).exists():
            pins[name] = sha(directory/name)
    release = dict(schema='r17.gpu_release.v1',status='GPU_RELEASED',cpu_only_remaining=True,
        released_at=datetime.now(timezone.utc).isoformat(),input_pins=pins,orchestration=pin,
        deployment=deployment,all_owned_closed=True,driver_completion=report['driver_completion'])
    durable_json(directory/'GPU_RELEASED.json',release)
    env = dict(os.environ,CUDA_VISIBLE_DEVICES='',JAX_PLATFORMS='cpu',PYTHONDONTWRITEBYTECODE='1',
        PYTHONPATH=launch.pythonpath(pin['root']),OMP_NUM_THREADS='1',OPENBLAS_NUM_THREADS='1')
    code = ('from pathlib import Path; from exp.r17_integration.ops import formal_eval as l; '
        'l.CONTROL_PYTHON='+repr(launch.CONTROL_PYTHON)+'; l.ROOT=Path('+repr(str(launch.ROOT))+'); '
        'from exp.r17_integration.ops.formal_finalize import background; '
        'background(Path('+repr(str(directory))+'))')
    with (directory/'evidence_finalizer.log').open('xb') as log:
        child = subprocess.Popen([launch.CONTROL_PYTHON,'-u','-c',code],cwd=pin['root'],env=env,
            stdout=log,stderr=subprocess.STDOUT,stdin=subprocess.DEVNULL,start_new_session=True,
            # Shell flock wrappers must release their GPU lock on launcher exit.
            close_fds=True)
    owner = dict(pid=child.pid,pgid=child.pid,start_ticks=ticks(child.pid),cpu_only=True,
                 seconds=7200,orchestration_identity=pin['identity'])
    durable_json(directory/'EVIDENCE_FINALIZER_PROCESS.json',owner)
    # Finalizer waits for this gate: the parent never edits its receipt again.
    durable_json(directory/'EVIDENCE_FINALIZER_START.json',dict(status='START',owner=owner))
    await_ready(child,directory)
    return report


def await_ready(child, directory, seconds=30):
    """Confirm CPU ownership/import; bulk verification stays behind the seal."""
    deadline=time.monotonic()+seconds
    while time.monotonic()<deadline:
        path=Path(directory)/'EVIDENCE_FINALIZATION.json'
        if path.exists():
            state=read(path)
            if state.get('status') in ('VERIFYING_INPUTS','RUNNING','PASS'):
                return
            if state.get('status') in ('FAIL','PARTIAL'):
                raise RuntimeError('CPU evidence finalizer failed before readiness: '+str(state.get('error')))
        returncode = child.poll()
        if returncode is not None:
            log = Path(directory)/'evidence_finalizer.log'
            detail = ''
            if log.exists():
                with log.open('rb') as stream:
                    stream.seek(max(0,log.stat().st_size-8192))
                    detail = stream.read(8192).decode('utf-8',errors='replace')
            raise RuntimeError(f'CPU evidence finalizer exited before readiness (rc={returncode}): '+detail)
        time.sleep(.05)
    raise TimeoutError('CPU evidence finalizer readiness deadline exceeded')


def background(directory):
    from .tether_retry import retry_scope
    directory = Path(directory)
    until = time.monotonic()+30
    while not (directory/'EVIDENCE_FINALIZER_START.json').exists():
        if time.monotonic() >= until:
            raise TimeoutError('finalizer parent did not publish handoff gate')
        time.sleep(.05)
    with (directory/'.evidence_seal.lock').open('a') as lock:
        fcntl.flock(lock,fcntl.LOCK_EX | fcntl.LOCK_NB)
        report = read(directory/'RECEIPT.json')
        try:
            release = read(directory/'GPU_RELEASED.json')
            pin = release['orchestration']
            if not verify(pin['root'],'RUNTIME_RELEASE.json',pin['identity'])['present']:
                raise ValueError('CPU finalizer release missing')
            owner = read(directory/'EVIDENCE_FINALIZER_PROCESS.json')
            if owner['pid'] != os.getpid() or owner['start_ticks'] != ticks(os.getpid()):
                raise ValueError('CPU finalizer ownership differs')
            # The two failed Cosmos releases pin 3.20 GB / .65 GB. Hashing
            # those is CPU evidence work, not a 30-second import handshake.
            # The public receipt remains PENDING until all audits pass.
            started=time.monotonic()
            durable_json(directory/'EVIDENCE_FINALIZATION.json',dict(status='VERIFYING_INPUTS',
                cpu_only=True,owner=owner,started_at=datetime.now(timezone.utc).isoformat()))
            if any(sha(directory/name) != digest for name,digest in release['input_pins'].items()):
                raise ValueError('GPU-release input changed before CPU evidence sealing')
            durable_json(directory/'EVIDENCE_FINALIZATION.json',dict(status='RUNNING',cpu_only=True,
                owner=owner,input_verification_seconds=time.monotonic()-started))
            with retry_scope(directory/'evidence_transport'):
                report = finalize(read(directory/'campaign.json'),directory,release['deployment'],report)
        except BaseException as exc:
            report.update(status='FAIL',evidence_pending=False,finalization_error=repr(exc))
            durable_json(directory/'RECEIPT.json',report)
        durable_json(directory/'EVIDENCE_FINALIZATION.json',dict(status=report['status'],cpu_only=True,
            owner=owner if 'owner' in locals() else None,finished_at=datetime.now(timezone.utc).isoformat(),
            receipt_sha256=sha(directory/'RECEIPT.json'),error=report.get('finalization_error')))
        print(json.dumps(dict(status=report['status'],run=str(directory))),flush=True)


def wait_seal(directory, *, seconds=7500, stopped=lambda:False):
    directory = Path(directory)
    deadline = time.monotonic()+seconds
    while True:
        report = read(directory/'RECEIPT.json')
        if (report.get('status') != 'PENDING_EVIDENCE' and
                (report.get('status') != 'PASS' or not report.get('worker_storage_cleanup_pending'))):
            if report.get('status') != 'PASS' or report.get('all_owned_closed') is not True:
                raise RuntimeError('CPU evidence seal failed: '+str(report.get('finalization_error') or report.get('status')))
            return dict(run=str(directory),status='PASS',receipt_sha256=sha(directory/'RECEIPT.json'))
        if stopped() or time.monotonic() >= deadline:
            raise TimeoutError('CPU evidence seal deadline exceeded')
        if (directory/'EVIDENCE_FINALIZER_PROCESS.json').exists() and not alive(read(directory/'EVIDENCE_FINALIZER_PROCESS.json')):
            raise RuntimeError('CPU evidence finalizer exited before seal')
        time.sleep(.2)


def retry_closed(directory):
    """Rerun only a closed, completed readiness failure; no experiment launch.

    Keep the failed public receipt and all existing frozen releases. Optional
    remote worker diagnostics are retained rather than deleted during recovery.
    The normal snapshot, native execution and batching audits still run.
    """
    from . import formal_eval as launch, cache_study
    from .tether_retry import retry_scope
    from unittest.mock import patch
    directory=Path(directory).resolve(strict=True)
    with (directory/'.evidence_seal.lock').open('a') as lock:
        fcntl.flock(lock,fcntl.LOCK_EX | fcntl.LOCK_NB)
        report=read(directory/'RECEIPT.json')
        if report.get('status')=='PASS':
            return dict(status='PASS',run=str(directory),already_sealed=True)
        if (report.get('status')!='FAIL' or report.get('finalization_error') !=
                "TimeoutError('CPU evidence finalizer readiness deadline exceeded')" or
                any(report.get(k) for k in ('error','batching_audit_error','shard_transfer_error'))):
            raise ValueError('CPU retry requires an isolated finalizer readiness failure')
        owner=directory/'EVIDENCE_FINALIZER_PROCESS.json'
        if owner.exists() and alive(read(owner)):
            raise ValueError('CPU evidence finalizer is still alive')
        closed_proof(report)
        completion=launch.driver_completion(directory,report['driver_completion']['returncode'])
        if completion!=report['driver_completion'] or completion.get('verified_completion') is not True:
            raise ValueError('CPU retry requires unchanged verified driver completion')
        campaign=read(directory/'campaign.json')
        deployment=read(directory/('LAUNCH_WORKER_DEPLOYMENT.json' if
            (directory/'LAUNCH_WORKER_DEPLOYMENT.json').exists() else 'WORKER_DEPLOYMENT.json'))
        stamp=datetime.now(timezone.utc).strftime('%Y%m%dT%H%M%S%fZ')
        backup=directory/('RECEIPT.before_cpu_refinalize_'+stamp+'.json')
        atomic_bytes(backup,(directory/'RECEIPT.json').read_bytes())
        proof=dict(schema='r17.cpu_refinalization.v1',cpu_only=True,started_at=stamp,
            original_receipt=str(backup),original_receipt_sha256=sha(backup),
            original_error=report.pop('finalization_error'),driver_completion=completion,
            model_server_worker_releases_unchanged=True,
            cpu_source_pins={n:sha(Path(__file__).with_name(n+'.py')) for n in ('formal_eval','formal_finalize')})
        report['cpu_refinalization']=proof
        try:
            with retry_scope(directory/'cpu_refinalize_transport'),patch.object(cache_study,'aggregate',cache_study.aggregate):
                path=directory/'assets/ffk_evidence.py'
                if campaign.get('ffk_evidence_sha256'):
                    if sha(path)!=campaign['ffk_evidence_sha256']:
                        raise ValueError('sealed FFK CPU evidence overlay changed')
                    scope={};exec(compile(path.read_text(),str(path),'exec'),scope);scope['install']()
                report=finalize(campaign,directory,deployment,report,retain_worker_storage=True)
                summary=read(directory/'SUMMARY.json')
                if summary.get('status')!='PASS' or summary.get('terminal_errors'):
                    raise ValueError('CPU refinalization did not verify complete native evidence')
        except BaseException as exc:
            report.update(status='FAIL',evidence_pending=False,finalization_error=repr(exc))
            durable_json(directory/'RECEIPT.json',report)
            raise
        proof.update(status='PASS',finished_at=datetime.now(timezone.utc).isoformat(),
            summary_sha256=sha(directory/'SUMMARY.json'))
        durable_json(directory/'CPU_REFINALIZATION.json',proof)
        report['cpu_refinalization']=proof
        durable_json(directory/'RECEIPT.json',report)
        return dict(status=report['status'],run=str(directory),original_receipt=str(backup),
            completed_episodes=summary['completed_episodes'])


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    mode=parser.add_mutually_exclusive_group(required=True)
    mode.add_argument('--wait',type=Path,nargs='+',help='Scientific completion barrier for closed campaigns')
    mode.add_argument('--retry-closed',type=Path,nargs='+',help='CPU-only recovery of completed readiness failures')
    args = parser.parse_args(argv)
    for directory in args.retry_closed or args.wait:
        print(json.dumps(retry_closed(directory) if args.retry_closed else wait_seal(directory)),flush=True)


if __name__ == '__main__':
    main()
