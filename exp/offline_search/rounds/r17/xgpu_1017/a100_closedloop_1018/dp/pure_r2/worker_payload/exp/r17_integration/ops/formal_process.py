"""Receipt-bound remote process ownership and expiring coordinator leases."""
from __future__ import annotations

import argparse
import json
import os
from pathlib import Path
import signal
import subprocess
import sys
import time


def write_json(path, value):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    part = path.with_suffix(path.suffix + '.part')
    part.write_text(json.dumps(value, indent=2, sort_keys=True, allow_nan=False) + '\n')
    os.replace(part, path)


def ticks(pid):
    try:
        return Path(f'/proc/{pid}/stat').read_text().split(') ', 1)[1].split()[19]
    except (FileNotFoundError, ProcessLookupError):
        return None


def alive(entry):
    if ticks(entry['pid']) != entry['start_ticks']:
        return False
    try:
        return Path(f"/proc/{entry['pid']}/stat").read_text().split(') ', 1)[1].split()[0] != 'Z'
    except FileNotFoundError:
        return False


def stop_group(entry, grace=12):
    """Signal only a still-identical session leader recorded in our receipt."""
    if not alive(entry):
        return {'status': 'closed', 'pid': entry['pid'], 'already_closed': True}
    if os.getpgid(entry['pid']) != entry['pgid'] or entry['pgid'] != entry['pid']:
        raise RuntimeError('owned process group identity changed')
    os.killpg(entry['pgid'], signal.SIGTERM)
    until = time.monotonic() + grace
    while alive(entry) and time.monotonic() < until:
        time.sleep(.1)
    forced = alive(entry)
    if forced:
        os.killpg(entry['pgid'], signal.SIGKILL)
    return {'status': 'closed', 'pid': entry['pid'], 'forced': forced}


def lease_expired(request, *, now=None):
    """Old requests retain120s; new launcher requests carry bounded grace."""
    seconds = request.get('lease_seconds', 120)
    if type(seconds) not in (int, float) or not 120 <= seconds <= 600:
        raise ValueError('remote lease grace must be120..600 seconds')
    lease = Path(request['lease'])
    return not lease.exists() or (time.time() if now is None else now)-lease.stat().st_mtime > seconds


def supervise(request):
    receipt = Path(request['receipt'])
    stopped = False
    def stop(*unused):
        nonlocal stopped
        stopped = True
    for sig in (signal.SIGINT, signal.SIGTERM):
        signal.signal(sig, stop)
    report = dict(status='RUNNING', supervisor={'pid': os.getpid(), 'pgid': os.getpgid(0),
        'start_ticks': ticks(os.getpid())}, request=request)
    child = None
    started = time.monotonic()
    try:
        with Path(request['log']).open('ab') as log:
            child = subprocess.Popen(request['argv'], env={**os.environ, **request['env']},
                cwd=request['cwd'], stdout=log, stderr=subprocess.STDOUT,
                stdin=subprocess.DEVNULL, start_new_session=True)
            report['child'] = dict(pid=child.pid, pgid=child.pid, start_ticks=ticks(child.pid), argv=request['argv'])
            write_json(receipt, report)
            while child.poll() is None and not stopped:
                if (time.monotonic() - started > request['seconds'] or
                    lease_expired(request)):
                    report['stop_reason'] = 'deadline_or_expired_lease'
                    break
                time.sleep(.2)
            if child.poll() is None:
                report['cleanup'] = stop_group(report['child'])
            child.wait(timeout=15)
            report.update(status='CLOSED', exit_code=child.returncode, elapsed_seconds=time.monotonic()-started)
    except BaseException as exc:
        report.update(status='FAIL', error=repr(exc))
        if child is not None and child.poll() is None:
            report['cleanup'] = stop_group(report['child'])
            child.wait(timeout=15)
    finally:
        write_json(receipt, report)


def spawn(request):
    if Path(request['receipt']).exists():
        raise FileExistsError(request['receipt'])
    Path(request['lease']).touch()
    Path(request['log']).parent.mkdir(parents=True, exist_ok=True)
    with Path(request['log']).with_suffix('.supervisor.log').open('ab') as log:
        child = subprocess.Popen([sys.executable, '-u', '-m', 'exp.r17_integration.ops.formal_process', 'supervise', '--request',
            json.dumps(request)], env={**os.environ, **request['env']}, cwd=request['cwd'],
            stdout=log, stderr=subprocess.STDOUT, stdin=subprocess.DEVNULL, start_new_session=True)
    until = time.monotonic() + 10
    while not Path(request['receipt']).exists() and child.poll() is None and time.monotonic() < until:
        time.sleep(.05)
    if not Path(request['receipt']).exists():
        stop_group(dict(pid=child.pid, pgid=child.pid, start_ticks=ticks(child.pid)))
        raise RuntimeError('remote supervisor did not publish ownership receipt')
    return json.loads(Path(request['receipt']).read_text())


def manage(request, mode):
    if mode == 'renew':
        Path(request['lease']).touch()
        return {'renewed': True}
    result = json.loads(Path(request['receipt']).read_text())
    if mode == 'stop':
        result['stop'] = stop_group(result['supervisor'], grace=20)
        result = {**json.loads(Path(request['receipt']).read_text()), 'stop': result['stop']}
        if result.get('child') and alive(result['child']):
            result['orphan_child_cleanup'] = stop_group(result['child'])
    result['supervisor_alive'] = alive(result['supervisor'])
    result['child_alive'] = bool(result.get('child') and alive(result['child']))
    result['alive'] = result['supervisor_alive'] or result['child_alive']
    from .worker_gpu import fleet_evidence
    try:
        evidence = fleet_evidence(request)
        result.update(evidence)
        result['alive'] = result['alive'] or bool(evidence.get('owned_worker_survivors'))
    except (OSError, ValueError, KeyError, IndexError) as exc:
        result.update(alive=True, worker_closure_error=repr(exc))
    return result


def controls(entries):
    """One remote session renews/checks all owned services on this node."""
    result = {}
    for entry in entries:
        request = entry['request']
        manage(request,'renew')
        result[entry['label']] = manage(request,'status')
    return result


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('mode', choices=('spawn', 'supervise', 'status', 'stop', 'renew'))
    p.add_argument('--request', required=True)
    args = p.parse_args()
    request = json.loads(args.request)
    if args.mode == 'spawn':
        result = spawn(request)
    elif args.mode == 'supervise':
        supervise(request)
        return
    elif args.mode == 'renew':
        result = manage(request,'renew')
    else:
        result = manage(request,args.mode)
    print(json.dumps(result))


if __name__ == '__main__':
    main()
