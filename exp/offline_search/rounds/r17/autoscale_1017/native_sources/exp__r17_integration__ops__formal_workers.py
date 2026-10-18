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


RECYCLE_TIMEOUT_SECONDS = 600
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


def recycle_slot(report, child, entry, gpu_lock):
    """Reap the old child, then reuse its already-admitted RAM/EGL slot."""
    if child.poll() is None:
        return dict(admitted=False, reason='old worker has not exited')
    entry['exit_code'] = child.wait(timeout=5)
    if not report['memory_admission'].get('admitted'):
        pass
    if hasattr(gpu_lock, 'reuse'):
        gpu = gpu_lock.reuse(entry['index'], entry)
    else:
        # CPU fixtures / exclusive non-LIBERO lease retain the same fleet slot.
        gpu = dict(admitted=True, reused=True)
    return dict(admitted=gpu['admitted'], ram_slot_reused=True, requested_extra_ram_bytes=0,
                requested_extra_gpu_mib=0, gpu=gpu)


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
    if campaign.get('worker_node') == 'local':
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
        from .build_client_bundle import sha256
        authorization(campaign)
        if sha256(args.campaign)!=os.environ.get('R17_LIVE_FLEET_CONFIG_SHA256'):
            pass
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
    restarts, pending = {}, {}
    gpu_lock = None
    drain_requested = lambda:False
    class DrainedBeforeStart(Exception):
        pass
    if campaign.get('live_fleets'):
        from .live_fleet_worker import draining
        drain_requested = lambda:draining(campaign)
    def launch(index, generation=0):
        report['disk_admission'] = disk_admission(run,minimum_bytes(campaign))
        if not report['disk_admission']['admitted']:
            pass
        worker_id, argv = worker_command(campaign, index)
        workspace = run/'scratch'/('worker_'+str(index))/('generation_'+str(generation))
        workspace.mkdir(parents=True, exist_ok=False)
        env = dict(os.environ, TMPDIR=str(workspace), XDG_CACHE_HOME=str(workspace/'cache'),
                   HF_HOME=str(workspace/'hf_cache'), R17_WORKER_RECYCLE_RECEIPT=str(workspace/'RECYCLE.json'))
        if campaign.get('worker_node')=='timan1':
            import secrets
            env['R17_T1_GENERATION_TOKEN']=secrets.token_hex(32)
        if campaign.get('worker_node') == 'local':
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
        log = (run / f'worker_{index}_generation_{generation}.log').open('wb')
        logs[index] = log
        def parent_death():
            import ctypes
            ctypes.CDLL(None).prctl(1, signal.SIGTERM)
        child = subprocess.Popen(argv, stdout=log, stderr=subprocess.STDOUT,
            preexec_fn=parent_death, cwd=workspace, env=env)
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
        from .worker_gpu import acquire
        gpu_lock = acquire(campaign, run)
        if hasattr(gpu_lock, 'result'):
            report['gpu_admission'] = gpu_lock.result
            campaign['worker_gpu_assignments'] = report['worker_gpu_assignments'] = gpu_lock.result['assignments']
            write_json(run/'receipt.json', report)
        from .worker_memory import admission
        with (run.parent/'.memory_admission.lock').open('a') as memory_lock:
            fcntl.flock(memory_lock, fcntl.LOCK_EX)
            report['memory_admission'] = admission(campaign, run)
            if not report['memory_admission']['admitted']:
                report.update(status='PARTIAL', reason=report['memory_admission']['reason'])
            else:
                report['status'] = 'RUNNING'
                for index in range(campaign['node_workers']):
                    if drain_requested():break
                    spawned=launch(index)
                    if spawned is None:break
                    children.append(spawned)
        if campaign.get('cache_study'):
            print(json.dumps(dict(fleet_receipt=str(run/'receipt.json'),status=report['status'],
                workers_spawned=len(children),worker_logs=str(run/'worker_*_generation_*.log'),
                awaiting_driver=dict(host=campaign['driver_host'],
                    port=campaign.get('driver_public_port',campaign['driver_port'])))),flush=True)
        until = started + campaign['seconds']
        while report['status']=='RUNNING' and not stopped and time.monotonic() < until:
            drain = drain_requested()
            if drain:
                report['fleet_drain']=dict(requested=True,no_replacements=True,
                    finish_current_episode=True)
            codes = [c.poll() for c, _ in children]
            for index, code in enumerate(codes):
                if code is None or code == 0:
                    continue
                child, entry = children[index]
                entry['exit_code'] = code
                from .worker_lifetime import planned_recycle
                planned = planned_recycle(entry, code)
                if not planned and entry.get('node')=='timan1':
                    from .timan1_workers import planned_recycle as wrapped_recycle
                    planned=wrapped_recycle(entry,code)
                if drain or drain_requested():
                    if planned:
                        entry['drained_boundary']=True
                        child.wait(timeout=5)
                        continue
                    report.update(status='PARTIAL',reason='worker_failed_during_fleet_drain')
                    report['worker_failures'].append(dict(index=index,pid=child.pid,exit_code=code,
                        generation=entry['generation']))
                    break
                if index not in pending:
                    failure = dict(index=index, pid=child.pid, exit_code=code,
                        generation=entry['generation'], signal=-code if code < 0 else None)
                    report['worker_recycles' if planned else 'worker_failures'].append(failure)
                    if not planned and restarts.get(index, 0) >= campaign.get('worker_restart_limit', 2):
                        report.update(status='PARTIAL', reason='worker_restart_budget_exhausted')
                        break
                    pending[index] = time.monotonic()+RECYCLE_TIMEOUT_SECONDS
                with (run.parent/'.memory_admission.lock').open('a') as memory_lock:
                    fcntl.flock(memory_lock, fcntl.LOCK_EX)
                    if planned:
                        allowed = recycle_slot(report,child,entry,gpu_lock)
                        gpu_allowed = allowed['admitted']
                    else:
                        allowed = admission(dict(campaign, node_workers=1), run, replacement=True)
                        gpu_allowed = not hasattr(gpu_lock, 'check') or gpu_lock.check(index)
                    report['last_replacement'] = dict(index=index,generation=entry['generation'],
                        planned_recycle=planned,old_pid=child.pid,admission=allowed,gpu_admitted=gpu_allowed,
                        remaining_seconds=max(0,pending[index]-time.monotonic()))
                    disk = disk_admission(run,minimum_bytes(campaign))
                    report['disk_admission'] = report['last_replacement']['disk_admission'] = disk
                    if not disk['admitted']:
                        if report.get('dispatch_pause',{}).get('reason') != 'worker_disk_low':
                            print('worker recycle paused: '+disk['reason'],flush=True)
                            report.setdefault('worker_disk_events',[]).append(dict(event='replacement_paused',
                                index=index,generation=entry['generation'],disk_admission=disk))
                        report['dispatch_pause'] = dict(reason='worker_disk_low',disk_admission=disk,
                            remaining_seconds=max(0,pending[index]-time.monotonic()))
                    elif report.get('dispatch_pause',{}).get('reason') == 'worker_disk_low':
                        report.setdefault('worker_disk_events',[]).append(dict(event='replacement_resumed',
                            index=index,generation=entry['generation'],disk_admission=disk))
                        report.pop('dispatch_pause')
                    if allowed['admitted'] and gpu_allowed and disk['admitted']:
                        # The generation file remains on disk for audit. Its
                        # parent FD must not accumulate across N4 replacements.
                        child.wait(timeout=5)
                        logs[index].close()
                        if drain_requested():
                            entry['drained_boundary']=True
                            continue
                        if not planned:
                            restarts[index] = restarts.get(index, 0)+1
                        spawned=launch(index, entry['generation']+1)
                        if spawned is None:
                            entry['drained_boundary']=True
                            continue
                        children[index] = spawned
                        pending.pop(index)
                    elif time.monotonic() >= pending[index]:
                        report.update(status='PARTIAL', reason=(disk['reason'] if not disk['admitted'] else
                                      'worker_restart_resource_timeout'),
                                      memory_admission=allowed, gpu_admitted=gpu_allowed)
                        break
                write_json(run/'receipt.json', report)
            if report['status'] != 'RUNNING':
                break
            if all(c == 0 or e.get('drained_boundary') for c,(_,e) in zip(codes,children)):
                report.update(status='PASS', reason='fleet_drained' if drain_requested() else 'normal_exit')
                break
            time.sleep(.2)
        else:
            if report['status']=='RUNNING':
                report.update(status='PARTIAL', reason='interrupted' if stopped else 'worker_deadline')
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
        for log in logs.values():
            log.close()
        report['elapsed_seconds'] = time.monotonic()-started
        write_json(run / 'receipt.json', report)
        lock.close()
        if gpu_lock is not None:
            gpu_lock.close()
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
