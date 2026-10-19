"""Client-only resident worker fleet; no cache, debug capture or model load."""
from __future__ import annotations

import argparse
import fcntl
import json
import os
from pathlib import Path
import signal
import subprocess
import sys
import time

from .formal_process import ticks, write_json
from .worker_disk import admission as disk_admission, minimum_bytes


LIVE_FLEET_PROTOCOL = 1


def stage_snapshot(archive, key, transfer, *, cache_study=False):
    """Use the finalizer's bounded size for cache evidence, in this process only.

    Avoid rebuilding a second tar when a long N4 fleet exceeds Q's default
    512MiB stage. The qualification/default path keeps that original bound.
    """
    from exp.r17_integration.qualification import archive_stage
    previous = archive_stage.MAX_BYTES
    try:
        if cache_study:
            archive_stage.MAX_BYTES = 8 * 1024**3
        result = archive_stage.stage_archive(archive,key,'workers',transfer['archive_sha256'])
        if cache_study:
            result['formal_max_bytes'] = archive_stage.MAX_BYTES
        return result
    finally:
        archive_stage.MAX_BYTES = previous


def worker_command(campaign, index, python=sys.executable):
    worker_id = f"{campaign['run_name']}_{campaign['worker_node']}_{index}"
    device = (campaign['worker_gpu_assignments'][index]['device'] if campaign.get('worker_gpu_assignments')
              else campaign['worker_egl_device'])
    argv = [python, '-u', '-m', 'exp.r17_integration.conductor.worker_entry',
        '--worker-id', worker_id, '--server-key', campaign['endpoint'],
        '--driver-host', campaign['driver_host'], '--driver-port', str(campaign.get('driver_public_port', campaign['driver_port'])),
        '--benchmark', campaign['domain'], '--benchmark-spec', campaign['worker_benchmark'],
        '--renderer', 'egl', '--egl-device', str(device)]
    if campaign.get('worker_max_episodes', 0):
        argv[argv.index('-m')+1] = 'exp.r17_integration.ops.worker_lifetime'
        argv += ['--max-episodes', str(campaign['worker_max_episodes'])]
        if campaign.get('cache_study'):
            argv += ['--cache-study']
    elif campaign.get('cache_study'):
        argv[argv.index('-m')+1] = 'exp.r17_integration.ops.cache_study_worker'
    if campaign.get('domain') == 'libero' and (campaign.get('library') or campaign.get('eval_pool')=='B'):
        argv += ['--init-states-dir',campaign['worker_init_states_dir']]
    if campaign.get('worker_node') in ('local', 'h100'):
        # Same-machine workers reach their model server over loopback.
        argv[argv.index('-m')+1] = 'exp.r17_integration.ops.local_worker_entry'
    if campaign.get('worker_node') == 'timan1':
        from .timan1_workers import wrapper,route
        route(dict(campaign),campaign)
        argv[0]=wrapper(campaign)
        argv[argv.index('-m')+1]='exp.r17_integration.ops.timan1_workers'
    return worker_id, argv


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--campaign', required=True)
    args = p.parse_args()
    campaign = json.loads(Path(args.campaign).read_text())
    if campaign.get('live_fleet_authorization'):
        from .live_fleet_worker import authorization
        authorization(campaign)
    run = Path(campaign['worker_run'])
    disk = disk_admission(run,minimum_bytes(campaign))
    if not disk['admitted']:
        print(json.dumps(dict(status='PARTIAL',reason=disk['reason'],disk_admission=disk)),
              file=sys.stderr,flush=True)
        raise SystemExit(2)
    run.mkdir(parents=True, exist_ok=False)
    stopped = False
    def stop(*unused):
        nonlocal stopped
        stopped = True
    for sig in (signal.SIGINT, signal.SIGTERM):
        signal.signal(sig, stop)
    report = dict(status='RUNNING', pids=[], debug_capture=False, workers=campaign['node_workers'])
    if campaign.get('cache_study'):
        campaign['worker_memory_mib']=max(4096,campaign.get('worker_memory_mib',4096))
    report.update(status='STARTING', worker_failures=[], worker_recycles=[],
        disk_admission=disk,
        node=campaign.get('worker_node'),policy_endpoint=campaign.get('policy_endpoint',campaign.get('endpoint')),
        fleet_owner=dict(pid=os.getpid(), start_ticks=ticks(os.getpid())),
        worker_memory_mib=campaign.get('worker_memory_mib', 4096 if campaign['domain']=='libero' else 2048))
    write_json(run / 'manifest.json', campaign)
    children, logs = [], {}
    lock = (run / '.eval.lock').open('a')
    fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
    started = time.monotonic()
    from exp.offline_search.rounds.r17.scheduler.node import NodeExecutor
    executor = NodeExecutor(campaign, run)
    drain_requested = lambda:False
    class DrainedBeforeStart(Exception):
        pass
    if campaign.get('live_fleets'):
        from .live_fleet_worker import draining
        drain_requested = lambda:draining(campaign)
    def launch(index, generation=0):
        report['disk_admission'] = disk_admission(run,minimum_bytes(campaign))
        worker_id, argv = worker_command(campaign, index)
        workspace = run/'scratch'/('worker_'+str(index))/('generation_'+str(generation))
        workspace.mkdir(parents=True, exist_ok=False)
        env = dict(os.environ, TMPDIR=str(workspace), XDG_CACHE_HOME=str(workspace/'cache'),
                   HF_HOME=str(workspace/'hf_cache'), R17_WORKER_RECYCLE_RECEIPT=str(workspace/'RECYCLE.json'))
        if campaign.get('worker_node')=='timan1':
            import secrets
            env['R17_T1_GENERATION_TOKEN']=secrets.token_hex(32)
        if campaign.get('worker_node') in ('local', 'h100'):
            target = ('exp.r17_integration.ops.worker_lifetime' if campaign.get('worker_max_episodes',0) else
                      'exp.r17_integration.ops.cache_study_worker' if campaign.get('cache_study') else
                      'exp.r17_integration.conductor.worker_entry')
            env.update(R17_LOCAL_WORKER_TARGET=target,R17_CANONICAL_POLICY_ENDPOINT=campaign['endpoint'],
                       R17_LOCAL_POLICY_ENDPOINT=campaign['policy_endpoint'])
        assignment = (campaign['worker_gpu_assignments'][index] if campaign.get('worker_gpu_assignments')
                      else None)
        if 'worker_egl_device' in campaign:
            device = assignment['device'] if assignment else campaign['worker_egl_device']
            # robosuite reads this at import, before its process-global EGL display.
            env.update(MUJOCO_EGL_DEVICE_ID=str(device), MUJOCO_GL='egl', PYOPENGL_PLATFORM='egl')
            if campaign['domain'] == 'libero':
                env['CUDA_VISIBLE_DEVICES'] = ''  # render ordinal stays physical; no worker CUDA model
        env.update(executor.worker_env(worker_id), OPENPI_AUTOSCALE_PARENT_PID=str(os.getpid()))
        if assignment:env['OPENPI_AUTOSCALE_GPU_UUID'] = assignment['uuid']
        log = (run / f'worker_{index}_generation_{generation}.log').open('wb')
        logs[index] = log
        # The bootstrap sets parent-death ownership in the child (the fleet is multithreaded).
        argv = [sys.executable, '-B', '-m', 'exp.offline_search.rounds.r17.scheduler.bootstrap', '--', *argv]
        child = subprocess.Popen(argv, stdout=log, stderr=subprocess.STDOUT, cwd=workspace, env=env)
        entry = dict(pid=child.pid, start_ticks=ticks(child.pid), argv=argv, worker_id=worker_id,
                     node=campaign.get('worker_node'),policy_endpoint=campaign.get('policy_endpoint',campaign.get('endpoint')),
                     index=index, generation=generation, exit_code=None, recycle_receipt=str(workspace/'RECYCLE.json'))
        if campaign.get('worker_node')=='timan1':
            entry.update(wrapped_generation_token=env['R17_T1_GENERATION_TOKEN'],
                         maximum_episodes=campaign['worker_max_episodes'])
        if 'worker_egl_device' in campaign:
            entry.update(egl_device=device, gpu_uuid=assignment['uuid'] if assignment else None,
                         worker_vram_mib=assignment['worker_vram_mib'] if assignment else None)
        report['pids'].append(entry)
        write_json(run/'receipt.json', report)
        return child, entry
    if campaign.get('live_fleets'):
        native_launch=launch
        def launch(index,generation=0):
            from .live_fleet_worker import lifecycle
            with lifecycle(campaign):
                if drain_requested():return None
                return native_launch(index,generation)
    try:
        if drain_requested():raise DrainedBeforeStart
        from exp.offline_search.rounds.r17.scheduler.fleet import run_fleet
        run_fleet(campaign, run, report, children, logs, launch, lambda: stopped, drain_requested, executor)
    except DrainedBeforeStart:
        report.update(status='PASS',reason='fleet_drained',
            fleet_drain=dict(requested=True,no_replacements=True,finish_current_episode=True))
    except BaseException as exc:
        report.update(status='PARTIAL', reason=repr(exc))
    finally:
        for child, entry in children:
            if child.poll() is None:
                child.terminate()
        until = time.monotonic() + 8
        while any(c.poll() is None for c, _ in children) and time.monotonic() < until:
            time.sleep(.1)
        for child, entry in children:
            if child.poll() is None:
                child.kill()
                entry['forced'] = True
            entry['exit_code'] = child.wait(timeout=5)
        from exp.offline_search.rounds.r17.scheduler.kernel_evidence import collect
        collect(run, executor.started_wall)
        if executor.slots is not None:
            executor.slots.close()
        for log in logs.values():
            log.close()
        report['elapsed_seconds'] = time.monotonic()-started
        write_json(run / 'receipt.json', report)
        lock.close()
    print(json.dumps(dict(fleet_receipt=str(run/'receipt.json'), status=report['status'],
                         reason=report.get('reason'), worker_failures=report['worker_failures'])), flush=True)
    from .sync_back import snapshot_run
    archive = run.with_suffix('.tar')
    # Thousands of N4 generations exceed the Q stage's uncompressed bound.
    # Compression preserves every file and the same per-file SHA manifest.
    transfer = snapshot_run(run, archive, compressed=bool(campaign.get('cache_study')))
    transfer.update(stage_snapshot(archive,run.name,transfer,cache_study=bool(campaign.get('cache_study'))))
    write_json(run.with_suffix('.transfer.json'), transfer)
    print(json.dumps(transfer), flush=True)
    raise SystemExit(0 if report['status'] == 'PASS' else 2)


if __name__ == '__main__':
    main()
