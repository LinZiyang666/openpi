"""Prepare/launch boundary and immutable reuse for formal SR campaigns (CPU)."""
from __future__ import annotations

from contextlib import contextmanager
from datetime import datetime, timezone
import fcntl
import hashlib
import json
import os
from pathlib import Path
import time

from .formal_process import write_json
from .immutable_release import archive_manifest, identity, install, sha, verify


@contextmanager
def phase(directory, name):
    started, stamp = time.monotonic(), datetime.now(timezone.utc).isoformat()
    status = 'FAIL'
    try:
        yield
        status = 'PASS'
    finally:
        with (Path(directory)/'PHASE_TIMINGS.jsonl').open('a') as stream:
            stream.write(json.dumps(dict(phase=name, started=stamp, ended=datetime.now(timezone.utc).isoformat(),
                seconds=time.monotonic()-started, status=status))+'\n')


def command_phase(argv):
    text=' '.join(map(str,argv))
    for marker,name in (('install_dependencies','dependency_assets_verify'),('private_config','private_config'),
        ('--import-probe','worker_import_probe'),('get_task_init_states','native_pool_validation'),
        ('worker_memory import admission','ram_admission'),('formal_resume import cpu_probe','resume_probe'),
        ('snapshot_run','closed_snapshot'),('formal_eval seal','model_cpu_seal'),
        ('formal_process spawn','owned_spawn'),('formal_process controls','owned_controls')):
        if marker in text:
            return name
    return 'command '+' '.join(map(str,argv[:2]))


def verify_code(root, manifest_name, expected):
    # Bootstrap verification never executes code from an unverified release.
    return ('import json,hashlib; from pathlib import Path; r=Path('+repr(str(root))+'); '
        'd=json.loads((r/'+repr(manifest_name)+').read_text()) if r.exists() else None; '
        'assert not r.is_symlink(); '
        'assert d is None or hashlib.sha256((json.dumps(d,sort_keys=True,separators=(",",":"))+"\\n").encode()).hexdigest()=='+repr(expected)+'; '
        'assert d is None or all(not (r/n).is_symlink() and not Path(n).is_absolute() and ".." not in Path(n).parts and '
        'hashlib.sha256((r/n).read_bytes()).hexdigest()==s for n,s in d["files"].items()); '
        'print(json.dumps(dict(present=d is not None,installed=str(r),reused=True,**(d or {}))))')


def client_release(launch, node, directory, stage, destination, payload, *, closed_capacity_refresh=False):
    python = '/scratch/zixuans8/openpi/.venv/bin/python'
    if node == 'local':
        from .local_workers import environment
        python = environment()
    document = payload.get('manifest')
    if document:
        existing = json.loads(launch.remote(node, [python, '-c',
            verify_code(destination, 'PAYLOAD.json', identity(document))], replay_safe=True))
        if existing.get('present'):
            return existing
    options=['--force'] if closed_capacity_refresh else []
    launch.transfer(['rsync','push', str(directory/'workers.tar'), node+':'+stage+'/workers.tar',*options])
    launch.transfer(['rsync','push', str(launch.ROOT/'exp/r17_integration/qualification/bootstrap_install.py'),
        node+':'+stage+'/install.py',*options])
    return json.loads(launch.remote(node, [python, stage+'/install.py',
        stage+'/workers.tar', destination, payload['archive_sha256'], '--immutable']))


def runtime_release(launch, campaign, directory):
    """Content-addressed across tar timestamps; never extract over a live release."""
    from exp.r17_integration.qualification.deploy_cpu import runtime_archive
    archive = directory/'runtime.tar'
    if campaign.get('cache_control'):
        from .cache_control import runtime_archive as cache_runtime_archive
        cache_runtime_archive(campaign,directory,launch.ROOT)
    elif campaign.get('runtime_from'):
        from .formal_runtime import copy_runtime
        copy_runtime(campaign, archive)
    elif campaign.get('resume_from'):
        from .formal_resume import copy_runtime
        copy_runtime(campaign, archive)
    else:
        runtime_archive(launch.ROOT, archive)
        if campaign.get('cache_study'):
            from .cache_study import append_stall_runtime
            append_stall_runtime(archive, launch.ROOT)
    document = archive_manifest(archive)
    content_pin, archive_pin = identity(document), sha(archive)
    root = '/data/mw_finetune/qualification/formal_runtime_'+content_pin
    stage = '/tmp/r17_formal/'+campaign['run_name']+'/server'
    if campaign['server_node'] == 'local':
        if campaign.get('runtime_from') and not verify(root, 'RUNTIME_RELEASE.json', content_pin)['present']:
            raise ValueError('frozen execution-host runtime missing')
        installed = install(archive, root, archive_pin, content_pin)
    else:
        launch.remote('h100', ['mkdir', '-p', stage])
        installed = json.loads(launch.remote('h100', [launch.host_python(campaign), '-c',
            verify_code(root, 'RUNTIME_RELEASE.json', content_pin)], replay_safe=True))
        if not installed['present']:
            if campaign.get('runtime_from'):
                raise ValueError('frozen execution-host runtime missing')
            launch.transfer(['rsync','push',str(archive),'h100:'+stage+'/runtime.tar'])
            launch.transfer(['rsync','push',str(launch.ROOT/'exp/r17_integration/ops/immutable_release.py'),
                'h100:'+stage+'/release_install.py'])
            installed = json.loads(launch.remote('h100',[launch.host_python(campaign),stage+'/release_install.py',
                '--root',root,'--identity',content_pin,'--archive',stage+'/runtime.tar','--archive-sha',archive_pin]))
    campaign.update(runtime=root, runtime_identity=content_pin)
    if campaign.get('runtime_from') or campaign.get('cache_control'):
        # A local mirror also freezes the coordinator/driver imports for H100
        # arms. Installing it never writes the execution host's live release.
        install(archive, root, archive_pin, content_pin)
        campaign['coordinator_runtime'] = root
    write_json(directory/'RUNTIME_RELEASE.json', dict(node=campaign['server_node'], archive_sha256=archive_pin,
        identity=content_pin, root=root, reused=installed['reused'], files=len(document['files'])))


def control_gate(campaign, directory):
    """Only preparation may defer a not-yet-closed paired control."""
    from exp.r17_integration.models.fastwam.formal_batching import check_control, treatment
    if campaign.get('paired_control') and campaign.get('runtime_from'):
        from .formal_runtime import paired_lineage
        lineage = paired_lineage(campaign, campaign['runtime_from'],
                                 require_complete=not campaign.get('defer_control_check'))
        if lineage is not None:
            write_json(directory/'PAIRED_CONTROL_LINEAGE.json', lineage)
    if campaign.get('defer_control_check'):
        source = Path(campaign['paired_control'])
        old = json.loads((source/'campaign.json').read_text())
        if treatment(old) != treatment(campaign) or old.get('commit_horizon',5)!=5 or old.get('denoising_steps',10)!=10:
            pass
        for name in ('model_manifest.json','model_spec.json','benchmark.json'):
            if json.loads((source/'assets'/name).read_text()) != json.loads((Path(campaign['local_assets'])/name).read_text()):
                pass
        write_json(directory/'BATCHING_CONTROL_PENDING.json',dict(source=str(source), dispatch_allowed=False))
        return None
    return check_control(campaign, campaign['local_assets'])


def seal_prepared(campaign, directory, deployment):
    paths = ['campaign.json','PREFLIGHT.json','WORKER_DEPLOYMENT.json','EPISODE_PLAN.json']
    paths += [str(p.relative_to(directory)) for p in sorted((directory/'assets').rglob('*')) if p.is_file()]
    paths += [node+'.json' for node in deployment]
    for name in ('FULL_EPISODE_PLAN.json','RESUME.json','RUNTIME_RELEASE.json','FROZEN_RUNTIME.json','ORCHESTRATION.json',
                 'DRIVER_PREFLIGHT.json','DRIVER_REFRESH.json','PAIRED_CONTROL_LINEAGE.json','CONTROL_REFRESH.json','STARTUP_RETRY.json',
                 'CACHE_CONTROL.json','CACHE_RUNTIME_OVERLAY.json','CACHE_WORKER_OVERLAY.json','CACHE_RESUME.json','CACHE_RESUME_QUARANTINE.json',
                 'CACHE_RECYCLE_BASE.tar','CACHE_RECYCLE_REFRESH.json','LOCAL_WORKER_ORCHESTRATION.json','WORKER_REDISTRIBUTION.json'):
        if (directory/name).exists():
            paths.append(name)
    write_json(directory/'PREPARED.json', dict(schema='r17.formal_prepared.v1', status='PASS', cpu_only=True,
        gpu_bound=False, dispatch_started=False, files={name:sha(directory/name) for name in paths},
        run=str(directory), runtime=campaign['runtime'], runtime_identity=campaign['runtime_identity']))


def validate_prepared(directory):
    directory = Path(directory)
    seal = json.loads((directory/'PREPARED.json').read_text())
    if seal.get('status')!='PASS' or seal.get('gpu_bound') or seal.get('dispatch_started'):
        raise ValueError('invalid prepared campaign')
    if (directory/'RECEIPT.json').exists() or (directory/'LAUNCH.json').exists():
        raise ValueError('prepared campaign already launched')
    if any(sha(directory/name)!=pin for name,pin in seal['files'].items()):
        pass
    return json.loads((directory/'campaign.json').read_text()), json.loads((directory/'WORKER_DEPLOYMENT.json').read_text())


@contextmanager
def preparation_lock(directory):
    # A CPU refresh and an explicit launch may never change/claim the same seal.
    with (Path(directory)/'.prepared.lock').open('a') as stream:
        fcntl.flock(stream,fcntl.LOCK_EX|fcntl.LOCK_NB)
        yield


def rebind_control(directory, control):
    """Re-seal only undispatched control metadata, preserving native inputs."""
    from . import formal_eval as launch
    from .formal_runtime import check_pins
    directory, control = Path(directory).resolve(strict=True), Path(control).resolve(strict=True)
    if directory.parent != launch.FORMAL_ROOT:
        pass
    with preparation_lock(directory):
        campaign, deployment = validate_prepared(directory)
        if not campaign.get('fastwam_batching') or not campaign.get('paired_control') or not campaign.get('runtime_from'):
            pass
        before = json.loads((directory/'PREPARED.json').read_text())
        original = campaign['paired_control']
        native = {name:sha(directory/name) for name in before['files'] if name != 'campaign.json'}
        native.update({name:sha(directory/name) for name in ('runtime.tar','workers.tar')})
        campaign.update(paired_control=str(control), defer_control_check=True)
        check_pins(campaign)
        control_gate(campaign, directory)
        # No GPU/renderer/import changes: every sealed native input stays byte-identical.
        if any(sha(directory/name) != pin for name,pin in native.items()):
            pass
        backup = directory/('PREPARED_BEFORE_CONTROL_'+str(time.time_ns())+'.json')
        write_json(backup, before)
        write_json(directory/'CONTROL_REFRESH.json', dict(status='PASS', cpu_only=True,
            previous_control=original, paired_control=str(control), runtime_from=campaign['runtime_from'],
            native_pins=native, previous_seal=backup.name, previous_seal_sha256=sha(backup),
            launch_requires_complete_control=True))
        write_json(directory/'campaign.json', campaign)
        seal_prepared(campaign, directory, deployment)
        validate_prepared(directory)
        return dict(run=str(directory), status='PASS', cpu_only=True, paired_control=str(control),
                    runtime_from=campaign['runtime_from'])


def refresh_driver(directory, mode='group-commit'):
    """CPU-only refresh of a sealed, undispatched run's orchestration pin.

    Native assets, worker archives/deployments and the original runtime tar are
    verified and left byte-identical. The previous preparation seal is kept.
    """
    from . import formal_eval as launch
    from . import driver_release
    directory=Path(directory).resolve(strict=True)
    if directory.parent!=launch.FORMAL_ROOT:
        pass
    with preparation_lock(directory),launch.retry_scope(directory),phase(directory,'orchestration_refresh'):
        campaign,deployment=validate_prepared(directory)
        before=json.loads((directory/'PREPARED.json').read_text())
        if campaign.get('orchestration'):driver_release.validate(campaign,directory)
        if campaign.get('runtime_from'):
            from .formal_runtime import check_assets
            check_assets(campaign,directory)
        native=[name for name in before['files'] if name.startswith('assets/') or name in
                ('EPISODE_PLAN.json','WORKER_DEPLOYMENT.json','RUNTIME_RELEASE.json') or name in
                [node+'.json' for node in deployment]]
        native+=['runtime.tar','workers.tar']
        pins={name:sha(directory/name) for name in native}
        if campaign['server_node']=='local':
            native_runtime=verify(campaign['runtime'],'RUNTIME_RELEASE.json',campaign['runtime_identity'])
        else:
            asset_pins={p.name:sha(p) for p in Path(campaign['local_assets']).iterdir()
                        if p.is_file() and p.name!='TRANSFER_SHA256.json'}
            code=verify_code(campaign['runtime'],'RUNTIME_RELEASE.json',campaign['runtime_identity'])
            code+=('; r=Path('+repr(campaign['assets'])+'); assert all('
                   'hashlib.sha256((r/n).read_bytes()).hexdigest()==s for n,s in '+repr(asset_pins)+'.items()), '
                   '"frozen remote native assets changed"')
            native_runtime=json.loads(launch.remote(campaign['server_node'],[launch.host_python(campaign),'-c',code],
                                                   replay_safe=True))
        if not native_runtime['present']:raise ValueError('frozen execution-host release missing')
        campaign['driver_mode']=mode
        driver_release.prepare(campaign,directory,launch.ROOT)
        driver_release.validate(campaign,directory)
        code=('import json; from pathlib import Path; from dataclasses import asdict; '
              'from exp.r17_integration.ops.formal_driver_fast import plan, AsyncCampaignDriver; '
              'c=json.loads(Path('+repr(str(directory/'campaign.json'))+').read_text()); _,_,tasks,_=plan(c); '
              'assert [asdict(t) for t in tasks]==json.loads(Path('+repr(str(directory/'EPISODE_PLAN.json'))+
              ').read_text())["tasks"], "orchestration changed episode inputs"; '
              'print(json.dumps(dict(status="PASS",episodes=len(tasks),model_worker_plan_unchanged=True)))')
        env=dict(os.environ,CUDA_VISIBLE_DEVICES='',JAX_PLATFORMS='cpu',PYTHONDONTWRITEBYTECODE='1',
                 OMP_NUM_THREADS='1',OPENBLAS_NUM_THREADS='1',MKL_NUM_THREADS='1',
                 PYTHONPATH=launch.pythonpath(campaign['driver_runtime']))
        plan=json.loads(launch.execute([launch.CONTROL_PYTHON,'-c',code],seconds=120,
                                      env=env,cwd=campaign['driver_runtime']))
        probes={}
        for node,item in deployment.items():
            installed={k:v for k,v in item['installed'].items() if k in ('schema','python','files','digest')}
            present=json.loads(launch.remote(node,[item['python'],'-c',
                verify_code(item['payload'],'PAYLOAD.json',identity(installed))],replay_safe=True))
            if not present['present']:raise ValueError('prepared worker release missing')
            probe=json.loads(launch.remote(node,['env',*[k+'='+v for k,v in item['env'].items()],
                'CUDA_VISIBLE_DEVICES=',item['python'],'-m','exp.r17_integration.conductor.worker_entry',
                '--worker-id','orchestration_import','--server-key','unused:1','--driver-host','unused',
                '--driver-port','1','--import-probe'],replay_safe=True))
            # --import-probe imports only: the full task/init/native renderer
            # attestation is still validated on each real worker connection.
            if probe!=item['import_probe']:pass
            probes[node]=probe
        if any(sha(directory/name)!=pin for name,pin in pins.items()):
            pass
        previous=directory/('PREPARED_BEFORE_DRIVER_'+str(time.time_ns())+'.json')
        write_json(previous,before)
        write_json(directory/'DRIVER_REFRESH.json',dict(status='PASS',cpu_only=True,gpu_bound=False,
            previous_seal=previous.name,previous_seal_sha256=sha(previous),native_pins=pins,
            baseline_orchestration=campaign['orchestration']['baseline']['identity'],
            selected_orchestration=campaign['orchestration']['selected']['identity']))
        write_json(directory/'DRIVER_PREFLIGHT.json',dict(status='PASS',cpu_only=True,plan=plan,worker_probes=probes,
                    native_runtime_identity=campaign['runtime_identity'],native_runtime_verified=True))
        write_json(directory/'campaign.json',campaign)
        seal_prepared(campaign,directory,deployment)
        validate_prepared(directory)
        return dict(run=str(directory),status='PASS',cpu_only=True,driver_mode=mode,
                    orchestration=campaign['orchestration']['selected']['identity'],episodes=plan['episodes'])


def launch_prepared(directory):
    from . import formal_eval as launch
    directory = Path(directory).resolve(strict=True)
    if directory.parent != launch.FORMAL_ROOT:
        pass
    tunnels = []
    # Exclusive one-shot claim: two queues cannot dispatch the same plan.
    with preparation_lock(directory):
        campaign, deployment = validate_prepared(directory)
        # Coordinator-only boundary hold: never affect the already live child.
        if (directory/'DISPATCH_HOLD.json').exists():
            raise RuntimeError('coordinator dispatch hold before launch claim: '+str(directory))
        if campaign.get('cache_capacity_authorization') and 'timan1' in campaign.get('execution_worker_nodes',{}):
            from .cache_capacity import t1_ready
            if not t1_ready(Path(campaign['cache_study']).parent/'evidence/t1_recycle_fix.json'):
                pass
        if campaign.get('cache_capacity_authorization',{}).get('preclaim_admission'):
            from .cache_capacity import wait_admission
            wait_admission(dict(name=campaign['run_name'],run=str(directory),model=campaign['model']+'_libero'),{},directory)
        if 'local' in campaign.get('execution_worker_nodes',campaign.get('worker_nodes',{})):
            from .local_workers import authorize
            if authorize(campaign)!=campaign.get('local_worker_exception'):
                pass
        from .worker_disk import prepared as disk_admission
        disk_admission(campaign,directory,deployment)
        if campaign.get('cache_study') and campaign.get('cache_control'):
            from .cache_control import refresh_prepared_workers
            campaign,deployment=refresh_prepared_workers(campaign,directory,deployment)
        if (not any(campaign.get(k) for k in ('resume_from','library','native_parity_only'))
                and all((directory/name).is_file() for name in ('runtime.tar','workers.tar'))):
            from .queue_retry import capture
            capture(directory,directory/'.startup_inputs')
        with (directory/'LAUNCH.json').open('x') as stream:
            json.dump(dict(pid=os.getpid(), started=time.time()),stream)
    try:
        with launch.retry_scope(directory), phase(directory, 'prepared_validation'):
            if campaign.get('cache_control'):
                from .cache_control import check
                check(campaign)
            if campaign.get('cache_resume_from'):
                from .cache_resume import check
                check(campaign)
            if campaign.get('orchestration'):
                from .driver_release import validate
                validate(campaign,directory)
            if campaign.get('runtime_from'):
                from .formal_runtime import check_pins
                check_pins(campaign)
                if not verify(campaign['coordinator_runtime'],'RUNTIME_RELEASE.json',campaign['runtime_identity'])['present']:
                    raise ValueError('frozen coordinator runtime missing')
            if campaign['server_node']=='local':
                if not verify(campaign['runtime'],'RUNTIME_RELEASE.json',campaign['runtime_identity'])['present']:
                    raise ValueError('prepared runtime missing')
            else:
                result=json.loads(launch.remote('h100',[launch.host_python(campaign),'-c',
                    verify_code(campaign['runtime'],'RUNTIME_RELEASE.json',campaign['runtime_identity'])],replay_safe=True))
                if not result['present']:
                    raise ValueError('prepared runtime missing')
                pins={p.name:sha(p) for p in Path(campaign['local_assets']).iterdir()
                    if p.is_file() and p.name!='TRANSFER_SHA256.json'}
                code=('from pathlib import Path; import hashlib; r=Path('+repr(campaign['assets'])+'); '
                    'assert all(hashlib.sha256((r/n).read_bytes()).hexdigest()==s '
                    'for n,s in '+repr(pins)+'.items()), "prepared remote assets changed"')
                launch.remote('h100',[launch.host_python(campaign),'-c',code],replay_safe=True)
            if campaign.get('resume_from'):
                from .formal_resume import check_pins
                check_pins(campaign)
            if campaign.get('fastwam_batching'):
                campaign.pop('defer_control_check',None)
                write_json(directory/'BATCHING_CONTROL.json',control_gate(campaign,directory))
            campaign['prepare_only']=False
            tunnels=launch.open_tunnels(campaign,directory)
            for node,item in deployment.items():
                installed=item['installed']
                present=json.loads(launch.remote(node,[item['python'],'-c',
                    verify_code(item['payload'],'PAYLOAD.json',identity({k:v for k,v in installed.items()
                        if k in ('schema','python','files','digest')}))],replay_safe=True))
                if not present['present']:
                    raise ValueError('prepared worker payload missing')
                worker=json.loads((directory/(node+'.json')).read_text())
                worker.update({key:value for key,value in campaign.items() if key in
                    ('endpoint','endpoint_host','endpoint_port','driver_host','driver_public_port','prepare_only')})
                if node == 'local':
                    from .local_workers import route
                    route(worker,campaign)
                if node == 'timan1':
                    from .timan1_workers import route
                    route(worker,campaign)
                    launch.install_worker_dependencies(node,'unused',item['payload'])
                if campaign.get('cache_study'):
                    from .worker_memory import reservation_mib
                    worker['worker_memory_mib']=reservation_mib(worker)
                memory_job={key:worker[key] for key in ('domain','node_workers','worker_memory_mib',
                                                      'worker_memory_headroom_mib') if key in worker}
                admission=json.loads(launch.remote(node,['env','PYTHONPATH='+launch.pythonpath(item['payload']),
                    item['python'],'-c','from exp.r17_integration.ops.worker_memory import admission; import json; '
                    'print(json.dumps(admission('+repr(memory_job)+','+repr(item['worker_run'])+')))'],replay_safe=True))
                write_json(directory/(node+'_LAUNCH_ADMISSION.json'),admission)
                if not admission['admitted']:
                    raise ValueError(node+': '+admission['reason'])
                # Same immutable asset/dependency releases, no install or native imports.
                dependency=item.get('dependencies')
                if dependency:
                    code=('from exp.r17_integration.ops.worker_installs import verify_dependencies; '
                        'verify_dependencies('+repr(dependency)+')')
                    # Hashing the dependency release can take minutes on the busy /data disk.
                    launch.remote(node,['env','PYTHONPATH='+launch.pythonpath(item['payload']),item['python'],'-c',code],600,replay_safe=True)
                config=item.get('private_config')
                pins={item['stage']+'/benchmark.json':sha(Path(campaign['local_assets'])/'benchmark.json')}
                if config:
                    pins[config['path']+'/config.yaml']=config['sha256']
                code=('import hashlib; from pathlib import Path; assert all('
                    'hashlib.sha256(Path(n).read_bytes()).hexdigest()==s for n,s in '+repr(pins)+'.items()), '
                    '"prepared worker benchmark/config changed"')
                launch.remote(node,[item['python'],'-c',code],replay_safe=True)
                path=directory/(node+'_launch.json')
                write_json(path,worker)
                item['campaign_path']=item['stage']+'/launch_campaign.json'
                launch.transfer(['rsync','push',str(path),node+':'+item['campaign_path']])
            write_json(directory/'campaign.json',campaign)
            if campaign['server_node']=='h100':
                campaign['server_campaign']='/tmp/r17_formal/'+campaign['run_name']+'/server/launch_campaign.json'
                write_json(directory/'campaign.json',campaign)
                launch.transfer(['rsync','push',str(directory/'campaign.json'),'h100:'+campaign['server_campaign']])
            if campaign.get('orchestration'):
                from .driver_release import planned_tasks
                task_rows=planned_tasks(campaign,directory)
            else:
                from .formal_driver import plan
                from dataclasses import asdict
                _,_,tasks,_=plan(campaign)
                task_rows=[asdict(t) for t in tasks]
            episode_plan=json.loads((directory/'EPISODE_PLAN.json').read_text())
            episode_plan.update(endpoint=campaign['endpoint'],tasks=task_rows)
            write_json(directory/'EPISODE_PLAN.json',episode_plan)
            write_json(directory/'LAUNCH_WORKER_DEPLOYMENT.json',deployment)
        return launch.run_campaign(campaign,directory,deployment)
    except Exception as exc:
        if not (directory/'RECEIPT.json').exists():
            write_json(directory/'RECEIPT.json',dict(status='FAIL',error=repr(exc),gpu_bound=False,all_owned_closed=True))
        raise
    finally:
        launch.close_tunnels(tunnels,directory)
