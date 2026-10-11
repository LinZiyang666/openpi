"""Owned timan supervisor, independently bounded even when tether disconnects."""
from __future__ import annotations

import argparse
import fcntl
import json
import os
from pathlib import Path
import signal
import subprocess
import time


def worker_command(args,run,index):
    worker_id=run.name+'_worker_'+str(index)
    return worker_id,[os.sys.executable,'-u','-m','exp.r17_integration.conductor.worker_entry',
        '--worker-id',worker_id,'--server-key',args.endpoint,'--driver-host',args.driver_host,
        '--driver-port',str(args.driver_port),'--renderer','egl','--egl-device',str(index),
        '--debug-directory',str(run/('physical_'+str(index)))]


def wait_workers(children,deadline,stopped):
    """Normal exit is observed explicitly; timeout/interrupt never becomes PASS."""
    while True:
        if stopped():return dict(status='PARTIAL',completion_reason='interrupted')
        if time.monotonic()>=deadline:return dict(status='PARTIAL',completion_reason='worker_deadline')
        codes=[c.poll() for c in children]
        if any(code is not None and code!=0 for code in codes):
            return dict(status='PARTIAL',completion_reason='worker_nonzero_exit')
        if children and all(code==0 for code in codes):
            return dict(status='PASS',completion_reason='normal_exit')
        time.sleep(min(.1,max(0,deadline-time.monotonic())))


def snapshot_closed(run,seconds):
    """Bound P6 archive creation and record only our own producer PID."""
    code=('import sys,json; sys.stdin.read(1); '
          'from exp.r17_integration.ops.sync_back import snapshot_run; '
          'from exp.r17_integration.qualification.archive_stage import stage_archive; '
          'r=snapshot_run(sys.argv[1],sys.argv[2]); '
          'r.update(stage_archive(sys.argv[2],sys.argv[3],"workers",r["archive_sha256"])); '
          'print(json.dumps(r))')
    argv=[os.sys.executable,'-u','-c',code,str(run),str(run.with_suffix('.tar')),run.name]
    def parent_death():
        import ctypes
        ctypes.CDLL(None).prctl(1,signal.SIGTERM)
    child=subprocess.Popen(argv,stdin=subprocess.PIPE,stdout=subprocess.PIPE,stderr=subprocess.PIPE,
        text=True,preexec_fn=parent_death)
    ledger=run.with_suffix('.snapshot_pid.json')
    entry=dict(pid=child.pid,argv=argv,exit_code=None,
        start_ticks=Path('/proc/'+str(child.pid)+'/stat').read_text().split(') ',1)[1].split()[19])
    ledger.write_text(json.dumps(entry)+'\n')
    try:
        stdout,stderr=child.communicate('1',timeout=seconds)
        if child.returncode:raise RuntimeError('closed snapshot failed: '+stderr[-1500:])
        return json.loads(stdout)
    finally:
        if child.poll() is None:
            child.terminate()
            try:child.wait(timeout=2)
            except subprocess.TimeoutExpired:child.kill();child.wait(timeout=2)
        entry['exit_code']=child.returncode
        ledger.write_text(json.dumps(entry)+'\n')


def main(argv=None):
    p = argparse.ArgumentParser()
    p.add_argument('--endpoint', required=True)
    p.add_argument('--driver-host', required=True)
    p.add_argument('--driver-port', type=int, required=True)
    p.add_argument('--run', required=True)
    p.add_argument('--seconds', type=int, default=600)
    p.add_argument('--import-probe', action='store_true')
    a = p.parse_args(argv)
    if a.import_probe:
        from exp.r17_integration.conductor.worker_entry import import_probe
        from exp.r17_integration.debug.capture_bridge import CaptureBridge, PhysicalSink
        from openpi_client.websocket_client_policy import WebsocketClientPolicy
        from exp.r17_integration.benchmarks.metaworld import MetaWorldBenchmarkAdapter
        from exp.r17_integration.conductor.manifest import validate_worker_probe
        imports=import_probe()
        benchmark_probe=MetaWorldBenchmarkAdapter().worker_probe()
        validate_worker_probe(benchmark_probe,require_renderer=False,require_task_data=False)
        print(json.dumps(dict(imports,debug_payload=True,executable=os.sys.executable,
            packages=benchmark_probe['packages'],benchmark_probe=benchmark_probe,
            package_protocol_validation=True,renderer_checked=False,task_data_checked=False)))
        return
    if not 1 <= a.seconds <= 650:
        p.error('remote wall bound <=650 seconds')
    run = Path(a.run)
    run.mkdir(parents=True, exist_ok=False)
    report = dict(status='PARTIAL', pids=[], supervisor_pid=os.getpid(),
                  supervisor_start_ticks=Path('/proc/self/stat').read_text().split(') ',1)[1].split()[19],endpoint=a.endpoint,
                  driver_host=a.driver_host, driver_port=a.driver_port, deadline_seconds=a.seconds)
    def save():
        temp = run / 'receipt.json.part'
        temp.write_text(json.dumps(report, indent=2)+'\n')
        os.replace(temp, run / 'receipt.json')
    (run / 'manifest.json').write_text(json.dumps(report)+'\n')
    save()
    stopped = False
    def stop(*args):
        nonlocal stopped
        stopped = True
    previous_handlers={sig:signal.signal(sig,stop) for sig in (signal.SIGTERM,signal.SIGINT)}
    children, handles = [], []
    started = time.monotonic()
    lock=(run/'.eval.lock').open('a')
    fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
    try:
        for i in range(2):
            worker_id,argv=worker_command(a,run,i)
            handle = (run / ('worker_'+str(i)+'.log')).open('wb')
            handles.append(handle)
            def parent_death():
                import ctypes
                ctypes.CDLL(None).prctl(1,signal.SIGTERM)
            child = subprocess.Popen(argv, stdout=handle, stderr=subprocess.STDOUT,preexec_fn=parent_death)
            children.append(child)
            report['pids'].append(dict(pid=child.pid, argv=argv, worker_id=worker_id))
            save()
        print('Q_REMOTE_PIDS '+json.dumps(report), flush=True)
        # Reserve archive time inside the remote wall bound. The stock workers
        # exit themselves on MSG_SHUTDOWN; their clean exit requires no signal.
        report.update(wait_workers(children,started+a.seconds-min(20,a.seconds*.1),lambda:stopped))
    except BaseException as exc:
        report.update(status='FAIL', error=str(exc))
    finally:
        for c in children:
            if c.poll() is None:
                c.terminate()
        grace = time.monotonic()+3
        while time.monotonic() < grace and any(c.poll() is None for c in children):
            time.sleep(.1)
        for c in children:
            if c.poll() is None:
                c.kill()
                report.setdefault('forced_pids', []).append(c.pid)
            c.wait()
        for entry, child in zip(report['pids'], children):
            entry['exit_code'] = child.returncode
        for handle in handles:
            handle.close()
        report['elapsed_seconds'] = time.monotonic()-started
        save()
        lock.close()
    try:
        transfer=snapshot_closed(run,max(.1,min(20,started+a.seconds-time.monotonic())))
        if stopped:report.update(status='PARTIAL',completion_reason='interrupted')
        print('Q_REMOTE_ARCHIVE '+json.dumps(transfer),flush=True)
    except subprocess.TimeoutExpired:
        report.update(status='PARTIAL',completion_reason='snapshot_deadline')
        save()
    except BaseException as exc:
        report.update(status='FAIL',error='snapshot: '+str(exc));save()
    finally:
        for sig,handler in previous_handlers.items():signal.signal(sig,handler)
    print('Q_REMOTE_RECEIPT '+json.dumps(report),flush=True)
    print(report['status'],flush=True)
    return 0 if report['status'] == 'PASS' else 2


if __name__ == '__main__':
    raise SystemExit(main())
