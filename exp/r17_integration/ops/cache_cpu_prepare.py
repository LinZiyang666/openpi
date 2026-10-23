"""Finish interrupted cache CPU preparation from verified native seals.

Only the worker preparation tail is retried. No resident, fleet, tunnel, model
load or renderer is launched; worker disk admission remains mandatory.
"""
from pathlib import Path
import shutil

from .formal_process import write_json
from .immutable_release import sha, verify


def finish(directory, manifest):
    from . import cache_study as study, formal_eval as launch
    from .build_client_bundle import safe_name
    from .cache_control import check
    from .driver_release import validate as validate_driver, planned_tasks
    from .formal_prestage import preparation_lock, phase, seal_prepared, validate_prepared

    directory=Path(directory).resolve(strict=True)
    with preparation_lock(directory),launch.retry_scope(directory):
        if any((directory/n).exists() for n in ('LAUNCH.json','RECEIPT.json','PREPARED.json','OWNED.json')):
            pass
        campaign=study.read(directory/'campaign.json')
        if (campaign.get('prepare_only') is not True or campaign.get('cache_study')!=str(Path(manifest).resolve()) or
                not campaign.get('cache_control') or campaign.get('resume_from') or
                campaign.get('local_assets')!=str(directory/'assets')):
            pass
        source=study.validate_manifest(manifest)
        sealed=study.read(directory/'assets/CACHE_STUDY.json')
        if sealed['source_study_digest']!=source['digest']:
            pass

        # Check the transferred snapshot's membership and closure before retrying.
        snapshot=study.read(directory/'assets/TRANSFER_SHA256.json')
        transfer=study.read(directory/'SERVER_ASSET_TRANSFER.json')
        if (transfer.get('status')!='verified' or transfer.get('digest')!=snapshot['digest'] or
                snapshot.get('partials') or transfer.get('partials')):
            pass
        assets=directory/'assets'
        names={str(p.relative_to(assets)) for p in assets.rglob('*') if p.is_file() and p.name!='.eval.lock'}
        if names!=set(snapshot['files'])|{'TRANSFER_SHA256.json'}:
            pass
        for name,pin in snapshot['files'].items():
            path=assets/safe_name(name)
            if not path.resolve().is_relative_to(assets):
                pass
        release=study.read(directory/'RUNTIME_RELEASE.json')
        if not verify(campaign.get('coordinator_runtime',campaign['runtime']),
                      'RUNTIME_RELEASE.json',release['identity'])['present']:
            pass
        check(campaign)

        # prepare() persists orchestration after worker preparation, so its
        # verified sidecar can outlive the older campaign.json on interruption.
        orchestration=study.read(directory/'ORCHESTRATION.json')
        campaign.update(orchestration=orchestration,driver_runtime=orchestration['driver_runtime'],
            driver_module=orchestration['selected']['module'],driver_mode=orchestration['selected']['mode'])
        validate_driver(campaign,directory)
        write_json(directory/'campaign.json',campaign)
        episode_plan=study.read(directory/'EPISODE_PLAN.json')
        if planned_tasks(campaign,directory)!=episode_plan['tasks']:
            pass
        native={str(p.relative_to(directory)):sha(p) for p in assets.rglob('*') if p.is_file()}
        native.update({n:sha(directory/n) for n in ('runtime.tar','EPISODE_PLAN.json','ORCHESTRATION.json')})
        if (directory/'workers.tar').exists():
            # Reuse the already frozen worker/client payload, retaining its
            # original archive and refreshing only the allowed recycle helpers.
            from .build_client_bundle import verified_members
            base=directory/'CPU_WORKER_PREPARE_BASE.tar'
            if not campaign.get('cache_worker_refresh'):
                verified_members(directory/'workers.tar','PAYLOAD.json')
                shutil.copyfile(directory/'workers.tar',base)
            else:
                base=Path(campaign['cache_worker_refresh']['archive'])
            campaign['cache_worker_refresh']=dict(archive=str(base),archive_sha256=sha(base),
                files={n:sha(launch.ROOT/n) for n in ('exp/r17_integration/ops/formal_workers.py',
                    'exp/r17_integration/ops/worker_gpu.py','exp/r17_integration/ops/worker_disk.py')})
            write_json(directory/'campaign.json',campaign)
        write_json(directory/'CPU_WORKER_PREPARE_RECOVERY.json',dict(status='WORKER_PREPARE_PENDING',
            cpu_only=True,gpu_launched=False,study_digest=source['digest'],native_pins=native))
        with phase(directory,'cpu_worker_prepare_recovery'):
            deployment=launch.prepare_workers(campaign,directory)
        write_json(directory/'campaign.json',campaign)
        write_json(directory/'PREFLIGHT.json',dict(status='PASS',cpu_only=True,
            episodes=episode_plan['expected_episodes'],worker_nodes=list(deployment),
            replicas=campaign['replicas'],ports=campaign['ports'],
            model_identity=episode_plan['model']['checkpoint_identity'],
            protocol=episode_plan['benchmark_protocol'],
            memory_admissible=all(study.read(directory/(node+'_MEMORY_ADMISSION.json'))['admitted']
                                  for node in deployment)))
        seal_prepared(campaign,directory,deployment)
        validate_prepared(directory)
        write_json(directory/'CPU_WORKER_PREPARE_RECOVERY.json',dict(status='PASS',
            cpu_only=True,gpu_launched=False,study_digest=source['digest'],native_pins=native))
        return campaign,deployment
