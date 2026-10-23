"""Closed, zero-acceptance startup retries using sealed campaign inputs."""
from __future__ import annotations

import argparse
import json
from pathlib import Path
import shutil
import time
import uuid

from .formal_process import write_json, alive
from .immutable_release import sha


def capture(directory, destination):
    from .formal_prestage import validate_prepared
    directory,destination = Path(directory),Path(destination)
    validate_prepared(directory)
    seal = json.loads((directory/'PREPARED.json').read_text())
    destination.mkdir(parents=True,exist_ok=False)
    for name in [*seal['files'],'PREPARED.json','runtime.tar','workers.tar']:
        source = directory/name
        target = destination/name
        target.parent.mkdir(parents=True,exist_ok=True)
        shutil.copyfile(source,target)
    write_json(destination/'STARTUP_INPUTS.json',dict(source=str(directory.resolve()),
        files={name:sha(destination/name) for name in [*seal['files'],'PREPARED.json','runtime.tar','workers.tar']}))
    validate_inputs(destination)
    return destination


def validate_inputs(directory):
    from .formal_prestage import validate_prepared
    directory = Path(directory)
    return validate_prepared(directory)


def zero_acceptance(directory):
    """Require confirmed closure; never retry partial scientific results."""
    directory = Path(directory)
    receipt = json.loads((directory/'RECEIPT.json').read_text())
    if receipt.get('all_owned_closed') is not True or receipt.get('status') not in ('FAIL','PARTIAL'):
        pass
    campaign_path = directory/'campaign.json'
    if not campaign_path.exists():
        raise RuntimeError('queue refuses switch: missing campaign evidence for startup retry')
    campaign = json.loads(campaign_path.read_text())
    if campaign.get('resume_from') or campaign.get('library'):
        raise ValueError('startup retry is restricted to fresh pure-inference campaigns')
    journal = directory/'driver/journal.jsonl'
    driver_owned = any(p.get('label')=='driver' for p in receipt.get('owned_local',[])+receipt.get('owned_remote',[]))
    if driver_owned and not journal.exists():
        raise ValueError('no journal to prove zero driver acceptance')
    paths = [journal,directory/'journal.jsonl',directory/'driver/attempts.jsonl']
    for path in paths:
        if path.exists():
            for line in path.read_text().splitlines():
                if line and json.loads(line).get('accepted') is True:
                    raise ValueError('campaign already accepted an episode/attempt; use explicit resume')
    for name in ('SUMMARY.json','driver/DRIVER_RECEIPT.json'):
        path = directory/name
        if path.exists():
            value = json.loads(path.read_text())
            if any(value.get(key,0) for key in ('completed_episodes','accepted_episodes','accepted','durable_accepted')):
                raise ValueError('campaign reports accepted episodes; no startup retry')
    return dict(status='PASS',source=str(directory.resolve()),accepted_episodes=0,
                all_owned_closed=True,original_failure=receipt.get('error'))


def clone(snapshot, name, failure, *, refresh_cache_workers=False,worker_counts=None,size_lane_authorization=None,capacity_authorization=None,replicas=None,local_vram_mib=None,local_server_growth_mib=None,campaign_overrides=None):
    """CPU-only fresh paths; keep native files, episode inputs, and driver pin."""
    from . import formal_eval as launch
    from .formal_prestage import seal_prepared, validate_prepared
    from .formal_runtime import check_pins, attach
    from .immutable_release import install, nvme_dir
    from .driver_release import validate as validate_driver
    campaign,_ = validate_inputs(snapshot)
    if (not name or len(name)>150 or any(c not in 'abcdefghijklmnopqrstuvwxyzABCDEFGHIJKLMNOPQRSTUVWXYZ0123456789_-' for c in name)):
        raise ValueError('startup retry requires a plain run identifier')
    if campaign.get('resume_from') or campaign.get('library'):
        raise ValueError('only fresh pure inference is supported by startup retry')
    snapshot = Path(snapshot)
    if campaign.get('cache_study'):
        # Cache inputs already carry separate reviewed framework/control pins.
        # Never reinterpret them as an ordinary pure-inference source.
        from .cache_control import check as check_cache_pins
        check_cache_pins(campaign)
    elif not campaign.get('runtime_from'):
        # Even a first control has an immutable native release at preparation.
        # Inherit its captured archives rather than rebuild the working tree.
        attach(campaign,snapshot)
        install(snapshot/'runtime.tar',campaign['runtime'],sha(snapshot/'runtime.tar'),campaign['runtime_identity'])
        campaign['coordinator_runtime'] = campaign['runtime']
    if campaign.get('runtime_from'):check_pins(campaign)
    directory = launch.FORMAL_ROOT/name
    nvme_dir(directory)  # real dir on /nvme, /data path a symlink
    seal = json.loads((snapshot/'PREPARED.json').read_text())
    for filename in [*seal['files'],'runtime.tar','workers.tar']:
        target = directory/filename
        target.parent.mkdir(parents=True,exist_ok=True)
        shutil.copyfile(snapshot/filename,target)
    old_name = campaign['run_name']
    campaign.update(run_name=name,prepare_only=True,local_assets=str(directory/'assets'),
        assets=str(directory/'assets'),server_run=str(directory/'servers'))
    if campaign['server_node']=='h100':
        campaign['server_campaign'] = '/tmp/r17_formal/'+name+'/server/campaign.json'
    # Continuation overrides (worker devices, wire codec, batching) must reach the worker deployment below.
    campaign.update(campaign_overrides or {})
    if size_lane_authorization is not None:
        if not campaign.get('cache_study') or worker_counts is None:
            pass
        campaign['cache_size_lane_authorization']=dict(size_lane_authorization)
        campaign['local_worker_server_vram_headroom_mib']=size_lane_authorization['local_server_growth_mib']
        resources=size_lane_authorization.get('resource_reservations')
        if resources is not None:
            campaign.update(resources)
    if capacity_authorization is not None:
        from .cache_capacity import resize
        resize(campaign,capacity_authorization,directory)
        worker_counts=campaign['execution_worker_nodes']
        from .driver_release import prepare
        prepare(campaign,directory,launch.ROOT)
    elif worker_counts is not None:
        from .worker_topology import redistribute
        redistribute(campaign,worker_counts,directory,replicas=replicas,
            local_vram_mib=local_vram_mib,local_server_growth_mib=local_server_growth_mib)
        if replicas is not None:
            from .driver_release import prepare
            prepare(campaign,directory,launch.ROOT)
    elif replicas is not None or local_vram_mib is not None or local_server_growth_mib is not None:
        pass
    if campaign.get('fastwam_batching'):
        campaign['defer_control_check'] = True
    write_json(directory/'campaign.json',campaign)
    if campaign.get('orchestration'):
        validate_driver(campaign,directory)
    if campaign['server_node'] == 'h100':
        from .sync_back import snapshot_run
        archive = directory/'startup_assets.tar'
        bundle = snapshot_run(str(directory/'assets'),str(archive))
        stage = '/tmp/r17_formal/'+name+'/server'
        launch.remote('h100',['mkdir','-p',stage],replay_safe=True)
        launch.transfer(['rsync','push',str(archive),'h100:'+stage+'/startup_assets.tar'])
        code = ('from exp.r17_integration.ops.sync_back import install_snapshot; '
                'install_snapshot('+repr(stage+'/startup_assets.tar')+','+repr(campaign['assets'])+
                ',expected_sha256='+repr(bundle['archive_sha256'])+')')
        launch.remote('h100',['env','PYTHONPATH='+launch.pythonpath(campaign['runtime']),
            launch.host_python(campaign),'-c',code])
        launch.transfer(['rsync','push',str(directory/'campaign.json'),'h100:'+campaign['server_campaign']])
    if refresh_cache_workers:
        if not campaign.get('cache_study') or not campaign.get('cache_control'):
            pass
        campaign['cache_worker_refresh']=dict(archive=str((snapshot/'workers.tar').resolve()),
            archive_sha256=sha(snapshot/'workers.tar'),files={n:sha(launch.ROOT/n) for n in (
                'exp/r17_integration/ops/formal_workers.py','exp/r17_integration/ops/worker_gpu.py',
                'exp/r17_integration/ops/worker_disk.py')})
        write_json(directory/'campaign.json',campaign)
    if not campaign.get('cache_control') and not campaign.get('runtime_from'):
        # The plain path rebuilds the worker payload; drop the copy cloned above (MT50 cache resumes).
        (directory/'workers.tar').unlink(missing_ok=True)
    with launch.retry_scope(directory):
        deployment = launch.prepare_workers(campaign,directory)
    driver_unchanged=(not (snapshot/'ORCHESTRATION.json').exists() or
                      sha(snapshot/'ORCHESTRATION.json')==sha(directory/'ORCHESTRATION.json'))
    write_json(directory/'STARTUP_RETRY.json',dict(schema='r17.startup_retry.v1',
        previous_run=old_name,input_snapshot=str(snapshot.resolve()),failure=failure,
        runtime_identity=campaign['runtime_identity'],native_files_unchanged=True,
        driver_pin_unchanged=driver_unchanged,driver_orchestration_refreshed=not driver_unchanged,
        episode_plan_sha256=sha(directory/'EPISODE_PLAN.json')))
    if worker_counts is not None:
        # The model preflight is inherited, but its worker summary must describe
        # the new execution fleet rather than the source's canonical topology.
        preflight=json.loads((directory/'PREFLIGHT.json').read_text())
        if replicas is not None:preflight['source_replicas']=preflight.get('replicas')
        preflight.update(worker_nodes=list(deployment),execution_worker_nodes=campaign['execution_worker_nodes'],
            execution_workers=campaign.get('execution_workers',campaign['workers']),replicas=campaign['replicas'],
            memory_admissible=all(json.loads((directory/(node+'_MEMORY_ADMISSION.json')).read_text())['admitted']
                                  for node in deployment))
        write_json(directory/'PREFLIGHT.json',preflight)
    seal_prepared(campaign,directory,deployment)
    validate_prepared(directory)
    return directory


def run_job(job, runner, limit=2, delay=10, initial_failure=False, resume_handler=None):
    from . import formal_eval as launch
    base = job['name']
    current = job
    snapshot = Path(job['run'])/'.startup_inputs'
    attempts = []
    for index in range(limit+1):
        failed = initial_failure and index == 0
        if not failed:
            try:
                runner.invoke(current,'run',['--run','--prepared-run',current['run']])
                receipt = json.loads((Path(current['run'])/'RECEIPT.json').read_text())
                if receipt.get('status') == 'PENDING_EVIDENCE':
                    from .formal_finalize import closed_proof
                    closed_proof(receipt)
                    if receipt.get('gpu_released') is not True or not (Path(current['run'])/'GPU_RELEASED.json').is_file():
                        pass
                    return dict(name=base,run=current['run'],status='PENDING_EVIDENCE',startup_retries=attempts)
                if receipt.get('status') != 'PASS' or receipt.get('all_owned_closed') is not True:
                    pass
                return dict(name=base,run=current['run'],status='PASS',startup_retries=attempts)
            except Exception:
                if index >= limit or getattr(runner,'stopped',False):
                    raise
                try:
                    failure = zero_acceptance(current['run'])
                except ValueError:
                    if resume_handler is None:
                        raise
                    fresh = resume_handler(current)
                    runner.event(event='cache_resume',campaign=base,failed_run=current['run'],fresh_run=str(fresh))
                    attempts.append(dict(failed_run=current['run'],fresh_run=str(fresh),cache_resume=True))
                    current=dict(job,name=Path(fresh).name,run=str(fresh))
                    snapshot=Path(fresh)/'.startup_inputs'
                    continue
        else:
            failure = zero_acceptance(current['run'])
            if limit == 0:
                raise RuntimeError('startup retries disabled')
        validate_inputs(snapshot)
        name = base+'_r'+str(index+1)+'_'+uuid.uuid4().hex[:4]
        # Validate private-pipe length before provisioning anything remotely.
        campaign,_ = validate_inputs(snapshot)
        if campaign.get('mps'):
            from exp.r17_integration.models.fastwam.ops.mps import paths
            paths(launch.FORMAL_ROOT/name,'GPU-placeholder')
        runner.event(event='startup_retry',campaign=base,failed_run=current['run'],retry_name=name,failure=failure)
        time.sleep(delay)
        fresh = clone(snapshot,name,failure)
        attempts.append(dict(failed_run=current['run'],fresh_run=str(fresh),proof=failure))
        current = dict(job,name=name,run=str(fresh))


def continued_jobs(directory, wait_seconds=172800):
    """Wait for the old queue's natural closure, then select only remaining jobs."""
    from . import formal_eval as launch
    from .formal_prestage import validate_prepared
    directory = Path(directory).resolve(strict=True)  # may resolve to /nvme (owner 10-20); no parent check
    deadline = time.monotonic()+wait_seconds
    while True:
        path = directory/'QUEUE_RECEIPT.json'
        owned = directory/'QUEUE_OWNED.json'
        closed = not owned.exists() or all(not alive(p) for p in json.loads(owned.read_text()))
        if path.exists() and closed:
            receipt = json.loads(path.read_text())
            if receipt.get('all_owned_closed') is not True:
                pass
            break
        if time.monotonic() >= deadline:
            raise TimeoutError('old queue did not close naturally; nothing launched')
        time.sleep(1)
    if receipt.get('cpu_only'):
        raise ValueError('use --prepared-queue for a CPU-only source')
    plan = json.loads((directory/'QUEUE_PLAN.json').read_text())
    jobs = []
    for original in plan['jobs']:
        job = dict(original)
        run = Path(job['run'])
        receipt_path = run/'RECEIPT.json'
        if receipt_path.exists():
            value = json.loads(receipt_path.read_text())
            if value.get('status')=='PASS' and value.get('all_owned_closed') is True:
                continue
            zero_acceptance(run)
            campaign,_ = validate_inputs(run/'.startup_inputs')
            job['initial_failure'] = True
        else:
            campaign,_ = validate_prepared(run)
        job['args'] = argparse.Namespace(seconds=campaign['seconds'])
        jobs.append(job)
    return jobs
