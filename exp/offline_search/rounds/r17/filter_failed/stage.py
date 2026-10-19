"""Remote CPU staging for the existing prepared launcher.

It starts no model server, worker, driver or chain.
Native server/runtime and scientific worker bytes are reused from the baseline;
new dynamic campaigns receive a separately pinned orchestration overlay.
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path
import shutil

from .audit_counts import HERE, REPO, read
from .build import write
from . import timan1_gpu2, worker_resilience


def stage_worker_inputs(launch, campaign, root, deployment):
    """Keep the fixed filenames read by launch_prepared beside live controls.

    prepare_workers(payload_override=...) stages content-addressed JSON, while
    launch_prepared hashes stage/benchmark.json against the *original bytes* of
    assets/benchmark.json. Re-serializing that JSON produces a different hash.
    The launcher replaces launch_campaign.json with its routed launch config.
    """
    for node, item in deployment.items():
        launch.remote(node, ['mkdir', '-p', item['stage']], replay_safe=True)
        for source, name in ((Path(campaign['local_assets'])/'benchmark.json', 'benchmark.json'),
                             (root/(node+'.json'), 'campaign.json'),
                             (root/(node+'.json'), 'launch_campaign.json')):
            launch.transfer(['rsync', 'push', str(source), node+':'+item['stage']+'/'+name])
        # Repair /tmp after loss of either spelling, including the benchmark
        # path embedded in the frozen-worker campaign. Never recompute controls.
        for key in ('benchmark_transfer', 'campaign_transfer'):
            if item.get(key):
                ref = item[key]
                launch.transfer(['rsync', 'push', str(root/'control_inputs'/(ref['sha256']+'.json')),
                                 node+':'+ref['path']])


def server_asset_destinations(campaign):
    """The server opens assets directly and also receives local_assets YAML paths.

    Short-name campaigns retain the original address inside frozen recipes.
    Both addresses must therefore contain the same sealed tree on H100.
    """
    return list(dict.fromkeys(campaign[key] for key in ('assets', 'local_assets')))


def stage_server_inputs(launch, campaign, root):
    if campaign['server_node'] != 'h100':
        return
    from exp.r17_integration.ops.sync_back import snapshot_run
    from exp.r17_integration.ops.immutable_release import sha
    archive = root/'startup_assets.tar'
    asset_source=root/'assets'
    if campaign.get('longseg_1017'):
        from exp.offline_search.rounds.r17.longseg_1017.staging import metadata_tree
        asset_source=metadata_tree(root)
    snapshot_run(str(asset_source), str(archive))
    directory = '/tmp/r17_formal/'+campaign['run_name']+'/server'
    launch.remote('h100', ['mkdir', '-p', directory], replay_safe=True)
    launch.transfer(['rsync', 'push', str(archive), 'h100:'+directory+'/startup_assets.tar'])
    code = ('from exp.r17_integration.ops.sync_back import install_snapshot; '+
            '; '.join('install_snapshot('+repr(directory+'/startup_assets.tar')+','+repr(destination)+
                      ',expected_sha256='+repr(sha(archive))+')'
                      for destination in server_asset_destinations(campaign)))
    launch.remote('h100', ['env', 'PYTHONPATH='+launch.pythonpath(campaign['runtime']),
                         launch.host_python(campaign), '-c', code], seconds=600, replay_safe=True)
    # New long-segment reruns preserve the original YAML bytes/task namespace.
    # Their frozen recipes address the original immutable cache_inputs trees;
    # stage those exact source trees, rather than rewriting the YAML or fits.
    for i, path in enumerate(campaign.get('longseg_1017_input_roots', [])):
        from exp.offline_search.rounds.r17.longseg_1017.staging import reusable_inputs
        if reusable_inputs(launch,path):
            continue
        inputs = root/('longseg_inputs_'+str(i)+'.tar')
        snapshot_run(path, str(inputs))
        remote_archive=directory+'/'+inputs.name
        launch.transfer(['rsync', 'push', str(inputs), 'h100:'+remote_archive])
        install='from exp.r17_integration.ops.sync_back import install_snapshot; install_snapshot('+repr(remote_archive)+','+repr(path)+',expected_sha256='+repr(sha(inputs))+')'
        launch.remote('h100', ['env', 'PYTHONPATH='+launch.pythonpath(campaign['runtime']),
                             launch.host_python(campaign), '-c', install], seconds=600, replay_safe=True)
    campaign['server_campaign'] = directory+'/campaign.json'
    write(root/'campaign.json', campaign)
    for name in ('campaign.json', 'launch_campaign.json'):
        launch.transfer(['rsync', 'push', str(root/'campaign.json'), 'h100:'+directory+'/'+name])


def stage(root):
    from exp.r17_integration.ops.formal_prestage import preparation_lock
    root = Path(root).resolve()
    with preparation_lock(root):
        if read(root/'campaign.json').get('longseg_1017'):
            from exp.offline_search.rounds.r17.longseg_1017.io_budget import filesystem_budget
            from exp.r17_integration.ops import direct_transfer
            from unittest.mock import patch
            with filesystem_budget(),patch.object(direct_transfer,'BWLIMIT_KIBPS',4096),patch.object(shutil,'_USE_CP_SENDFILE',False):
                return _stage(root)
        return _stage(root)


@timan1_gpu2.coordinator()
@worker_resilience.coordinator()
def _stage(root):
    from exp.r17_integration.ops import formal_eval as launch
    from exp.r17_integration.ops.formal_runtime import attach
    from exp.r17_integration.ops.driver_release import prepare, planned_tasks
    from exp.r17_integration.ops.formal_prestage import seal_prepared
    root = Path(root).resolve()
    campaign = read(root/'campaign.json')
    environment_file = HERE/'SERVER_ENVIRONMENTS.json'
    if environment_file.exists():
        selected = read(environment_file)['jobs'].get(campaign['run_name'])
        if selected:
            campaign['server_python'] = selected['python']
    source = Path(campaign['filter_failed_source'])
    continuation=campaign.get('cache_resume_selection')
    if continuation:
        source=Path(continuation['source'])
        if continuation.get('source_closure_required_at_prepare'):
            from .resume import prove_closed,refresh_remaining
            # Same native closure proof used by ordinary continuations. An
            # error-only draft may be built while its parent is still running.
            closure=prove_closed(source)
            write(root/'SOURCE_CLOSURE.json',closure)
            campaign=refresh_remaining(root,closure)
            continuation=campaign['cache_resume_selection']
    if any((root/name).exists() for name in ('LAUNCH.json', 'RECEIPT.json')):
        raise ValueError('campaign was already launched; use filter_failed.reset for a pre-start failure')
    # R17_H100_NODE lane: serve this H100 job on the alias machine (routing fields only).
    from exp.offline_search.rounds.r17.longseg_1017.ports import execution_host
    campaign = execution_host(campaign, root)
    from exp.offline_search.rounds.r17.autoscale_1017.defaults import apply
    apply(campaign, root)
    from exp.r17_integration.ops.formal_ports import prepare_admission
    prepare_admission(campaign)
    # The standard runtime_release step also makes this coordinator mirror.
    # Reuse the original archive, without cache_control rebuilding any overlay.
    from exp.r17_integration.ops.immutable_release import install, sha
    runtime_sha = sha(source/'runtime.tar')
    if campaign.get('longseg_1017'):
        # Long-segment campaigns run the archive with the segment-boundary fix
        # written into its sources (longseg_1017/runtime_fix.py).
        if sha(root/'runtime.tar') != runtime_sha and not (root/'LONGSEG_RUNTIME_FIX.json').exists():
            raise ValueError('runtime.tar is neither the source release nor its fixed copy')
        from exp.offline_search.rounds.r17.longseg_1017.runtime_fix import stage as fixed_runtime
        runtime_sha = fixed_runtime(launch, campaign, root)
    install(root/'runtime.tar', campaign['runtime'], runtime_sha, campaign['runtime_identity'])
    campaign['coordinator_runtime'] = campaign['runtime']
    if campaign.get('local_worker_exception'):
        # payload_override skips local_workers.payload, which normally writes
        # this receipt. run_campaign reads it before spawning anything. Its
        # original archive pin remains correct because that archive is reused.
        if (source/'LOCAL_WORKER_ORCHESTRATION.json').exists():
            shutil.copyfile(source/'LOCAL_WORKER_ORCHESTRATION.json', root/'LOCAL_WORKER_ORCHESTRATION.json')
    write(root/'campaign.json', campaign)
    if (root/'PREPARED.json').exists():
        # PREPARED seals local inputs, not ephemeral /tmp copies. A repeated
        # prepare must repair those copies rather than silently skip staging.
        deployment = read(root/'WORKER_DEPLOYMENT.json')
        with launch.retry_scope(root):
            stage_server_inputs(launch, campaign, root)
            stage_worker_inputs(launch, campaign, root, deployment)
        seal_prepared(campaign, root, deployment)
        print(json.dumps(dict(status='PREPARED', run=str(root), reused=True,
                              restaged=True, server_started=False, workers_started=False)))
        return
    if not continuation:
        attach(campaign,source)
        from exp.r17_integration.ops.cache_control import attach as attach_control
        attach_control(campaign,root)
        # Existing orchestration overlay supports task/init subsets. It leaves
        # model code and renderer/worker payloads in their original releases.
        prepare(campaign,root,REPO)
    elif campaign.get('autoscale'):
        # A dynamic continuation needs the driver's autoscale hook: rebuild the
        # driver over the same remaining plan (checked below).
        prepare(campaign,root,REPO)
    # A fixed continuation inherits the exact source driver/control pins;
    # provision only its new worker/server paths, without rebuilding them.
    write(root/'campaign.json',campaign)
    assert planned_tasks(campaign,root)==read(root/'EPISODE_PLAN.json')['tasks']
    shutil.copyfile(source/'WORKER_DEPLOYMENT.json',root/'WORKER_DEPLOYMENT.json')
    with launch.retry_scope(root):
        deployment=worker_resilience.prepare(campaign,root,
            lambda c,d:launch.prepare_workers(c,d,payload_override=source/'workers.tar'))
        # Worker preparation can add routing/lease metadata to the campaign.
        # The H100 copy must contain the final prepared document.
        stage_server_inputs(launch, campaign, root)
        stage_worker_inputs(launch, campaign, root, deployment)
    write(root/'campaign.json',campaign)
    if not campaign.get('autoscale'):
        assert sha(root/'workers.tar')==sha(source/'workers.tar')
    assert sha(root/'runtime.tar')==runtime_sha
    write(root/'PREFLIGHT.json',dict(status='PASS',cpu_only=True,episodes=read(root/'EPISODE_PLAN.json')['expected_episodes'],
          worker_nodes=list(deployment),replicas=campaign['replicas'],ports=campaign['ports'],
          model_identity=read(root/'assets/model_spec.json')['checkpoint_identity'],
          protocol=read(root/'assets/CACHE_STUDY.json')['benchmark_protocol'],
          memory_admissible=all(read(root/(node+'_MEMORY_ADMISSION.json'))['admitted'] for node in deployment)))
    seal_prepared(campaign,root,deployment)
    print(json.dumps(dict(status='PREPARED',run=str(root),server_started=False,workers_started=False)))


if __name__=='__main__':
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--run',type=Path,required=True)
    args=p.parse_args();stage(args.run)
