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


def worker_command(campaign, index, python=sys.executable):
    worker_id = f"{campaign['run_name']}_{campaign['worker_node']}_{index}"
    argv = [python, '-u', '-m', 'exp.r17_integration.conductor.worker_entry',
        '--worker-id', worker_id, '--server-key', campaign['endpoint'],
        '--driver-host', campaign['driver_host'], '--driver-port', str(campaign.get('driver_public_port', campaign['driver_port'])),
        '--benchmark', campaign['domain'], '--benchmark-spec', campaign['worker_benchmark'],
        '--renderer', 'egl', '--egl-device', str(campaign['worker_egl_device'])]
    if campaign.get('worker_max_episodes', 0):
        argv[argv.index('-m')+1] = 'exp.r17_integration.ops.worker_lifetime'
        argv += ['--max-episodes', str(campaign['worker_max_episodes'])]
    if campaign.get('library'):
        argv += ['--init-states-dir',campaign['worker_init_states_dir']]
    return worker_id, argv


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--campaign', required=True)
    args = p.parse_args()
    campaign = json.loads(Path(args.campaign).read_text())
    run = Path(campaign['worker_run'])
    run.mkdir(parents=True, exist_ok=False)
    stopped = False
    def stop(*unused):
        nonlocal stopped
        stopped = True
    for sig in (signal.SIGINT, signal.SIGTERM):
        signal.signal(sig, stop)
    report = dict(status='RUNNING', pids=[], debug_capture=False, workers=campaign['node_workers'])
    report.update(status='STARTING', worker_failures=[], worker_recycles=[],
        fleet_owner=dict(pid=os.getpid(), start_ticks=ticks(os.getpid())),
        worker_memory_mib=campaign.get('worker_memory_mib', 4096 if campaign['domain']=='libero' else 2048))
    write_json(run / 'manifest.json', campaign)
    children, logs = [], []
    lock = (run / '.eval.lock').open('a')
    fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
    started = time.monotonic()
    restarts, pending = {}, {}
    def launch(index, generation=0):
        worker_id, argv = worker_command(campaign, index)
        workspace = run/'scratch'/('worker_'+str(index))/('generation_'+str(generation))
        workspace.mkdir(parents=True, exist_ok=False)
        env = dict(os.environ, TMPDIR=str(workspace), XDG_CACHE_HOME=str(workspace/'cache'),
                   HF_HOME=str(workspace/'hf_cache'), R17_WORKER_RECYCLE_RECEIPT=str(workspace/'RECYCLE.json'))
        log = (run / f'worker_{index}_generation_{generation}.log').open('wb')
        logs.append(log)
        def parent_death():
            import ctypes
            ctypes.CDLL(None).prctl(1, signal.SIGTERM)
        child = subprocess.Popen(argv, stdout=log, stderr=subprocess.STDOUT,
            preexec_fn=parent_death, cwd=workspace, env=env)
        entry = dict(pid=child.pid, start_ticks=ticks(child.pid), argv=argv, worker_id=worker_id,
                     index=index, generation=generation, exit_code=None, recycle_receipt=str(workspace/'RECYCLE.json'))
        report['pids'].append(entry)
        write_json(run/'receipt.json', report)
        return child, entry
    try:
        from .worker_memory import admission
        with (run.parent/'.memory_admission.lock').open('a') as memory_lock:
            fcntl.flock(memory_lock, fcntl.LOCK_EX)
            report['memory_admission'] = admission(campaign, run)
            if not report['memory_admission']['admitted']:
                report.update(status='PARTIAL', reason=report['memory_admission']['reason'])
            else:
                report['status'] = 'RUNNING'
                for index in range(campaign['node_workers']):
                    children.append(launch(index))
        until = started + campaign['seconds']
        while report['status']=='RUNNING' and not stopped and time.monotonic() < until:
            codes = [c.poll() for c, _ in children]
            for index, code in enumerate(codes):
                if code is None or code == 0:
                    continue
                child, entry = children[index]
                entry['exit_code'] = code
                from .worker_lifetime import planned_recycle
                planned = planned_recycle(entry, code)
                if index not in pending:
                    failure = dict(index=index, pid=child.pid, exit_code=code,
                        generation=entry['generation'], signal=-code if code < 0 else None)
                    report['worker_recycles' if planned else 'worker_failures'].append(failure)
                    if not planned and restarts.get(index, 0) >= campaign.get('worker_restart_limit', 2):
                        report.update(status='PARTIAL', reason='worker_restart_budget_exhausted')
                        break
                    pending[index] = time.monotonic()+120
                with (run.parent/'.memory_admission.lock').open('a') as memory_lock:
                    fcntl.flock(memory_lock, fcntl.LOCK_EX)
                    allowed = admission(dict(campaign, node_workers=1), run, replacement=True)
                    if allowed['admitted']:
                        if not planned:
                            restarts[index] = restarts.get(index, 0)+1
                        children[index] = launch(index, entry['generation']+1)
                        pending.pop(index)
                    elif time.monotonic() >= pending[index]:
                        report.update(status='PARTIAL', reason='worker_restart_memory_timeout', memory_admission=allowed)
                        break
                write_json(run/'receipt.json', report)
            if report['status'] != 'RUNNING':
                break
            if all(c == 0 for c in codes):
                report.update(status='PASS', reason='normal_exit')
                break
            time.sleep(.2)
        else:
            if report['status']=='RUNNING':
                report.update(status='PARTIAL', reason='interrupted' if stopped else 'worker_deadline')
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
        for log in logs:
            log.close()
        report['elapsed_seconds'] = time.monotonic()-started
        write_json(run / 'receipt.json', report)
        lock.close()
    print(json.dumps(dict(fleet_receipt=str(run/'receipt.json'), status=report['status'],
                         reason=report.get('reason'), worker_failures=report['worker_failures'])), flush=True)
    from .sync_back import snapshot_run
    from exp.r17_integration.qualification.archive_stage import stage_archive
    archive = run.with_suffix('.tar')
    transfer = snapshot_run(run, archive)
    transfer.update(stage_archive(archive, run.name, 'workers', transfer['archive_sha256']))
    write_json(run.with_suffix('.transfer.json'), transfer)
    print(json.dumps(transfer), flush=True)
    raise SystemExit(0 if report['status'] == 'PASS' else 2)


if __name__ == '__main__':
    main()
