#!/usr/bin/env python3
"""Read-only /proc and socket sampling. Writes JSON only to stdout.

Run --remote through one persistent tether exec; no campaign attachment,
signals, environment changes, or GPU runtime imports. Kernel stack/syscall
permissions may prevent exact attribution; retain that limitation explicitly.
"""
import argparse
from collections import Counter
import json
import os
from pathlib import Path
import shutil
import subprocess
import time


def read(path):
    try:
        return Path(path).read_text().strip()
    except OSError as exc:
        return {"error": str(exc)}


def sockets():
    result = {}
    for kind in ("tcp", "tcp6"):
        value = read('/proc/net/' + kind)
        if not isinstance(value, str):
            continue
        for line in value.splitlines()[1:]:
            fields = line.split()
            if len(fields) >= 10:
                tx, rx = fields[4].split(':')
                result[fields[9]] = dict(local=fields[1], remote=fields[2], state=fields[3],
                    tx_bytes=int(tx, 16), rx_bytes=int(rx, 16))
    return result


def proc(pid, tcp, details=False):
    root = Path('/proc') / str(pid)
    try:
        fields = (root/'stat').read_text().rsplit(') ', 1)[1].split()
        threads = []
        for task in (root/'task').iterdir():
            try:
                stat = (task/'stat').read_text().rsplit(') ', 1)[1].split()
                threads.append(dict(tid=int(task.name), state=stat[0], ticks=int(stat[11])+int(stat[12]),
                    wchan=read(task/'wchan')))
            except (OSError, IndexError, ValueError):
                continue
        fds = []
        for fd in (root/'fd').iterdir():
            try:
                link = os.readlink(fd)
                if link.startswith('socket:['):
                    fds.append(dict(fd=int(fd.name), inode=link[8:-1], **tcp.get(link[8:-1], {})))
                elif details and ('per_step.jsonl' in link or 'worker_dispatch.jsonl' in link or 'journal.jsonl' in link):
                    fds.append(dict(fd=int(fd.name), file=link))
            except OSError:
                continue
        out = dict(pid=pid, state=fields[0], start_ticks=int(fields[19]), ticks=int(fields[11])+int(fields[12]),
            main_wchan=read(root/'wchan'), threads=threads, sockets=fds,
            io=read(root/'io'))
        if details:
            out.update(syscall=read(root/'syscall'), stack=read(root/'stack'))
        return out
    except (OSError, IndexError, ValueError):
        return None


def worker_pids(campaign):
    result = []
    for path in Path('/proc').iterdir():
        if not path.name.isdigit():
            continue
        try:
            argv = (path/'cmdline').read_bytes().decode(errors='replace').split('\0')
            if '--worker-id' not in argv:
                continue
            worker = argv[argv.index('--worker-id')+1]
            if not worker.startswith(campaign+'_timan107_') or '-m' not in argv:
                continue
            module = argv[argv.index('-m')+1]
            if module not in ('exp.r17_integration.ops.worker_lifetime', 'exp.r17_integration.conductor.worker_entry'):
                continue
            result.append((int(path.name), worker))
        except (OSError, ValueError, IndexError):
            continue
    return result


def gpu():
    try:
        p = subprocess.run(['nvidia-smi', '--query-gpu=index,utilization.gpu,power.draw,memory.used',
            '--format=csv,noheader,nounits'], capture_output=True, text=True, timeout=3)
        return dict(stdout=p.stdout.strip(), error=p.stderr.strip())
    except (OSError, subprocess.TimeoutExpired) as exc:
        return dict(error=str(exc))


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--campaign', required=True)
    parser.add_argument('--remote', action='store_true')
    parser.add_argument('--seconds', type=float, default=360)
    parser.add_argument('--interval', type=float, default=1)
    parser.add_argument('--pids', type=int, nargs='*', default=[2020942, 2022835, 2022909])
    args = parser.parse_args()
    start = time.monotonic()
    pids = []
    next_census = next_gpu = 0
    print(json.dumps(dict(kind='meta', time=time.time(), node='timan107' if args.remote else 'local',
        uid=os.getuid(), clock_ticks=os.sysconf('SC_CLK_TCK'), yama=read('/proc/sys/kernel/yama/ptrace_scope'),
        py_spy=shutil.which('py-spy'), interval=args.interval)), flush=True)
    while time.monotonic()-start < args.seconds:
        tick = time.monotonic()
        if tick >= next_census:
            pids = worker_pids(args.campaign) if args.remote else [(pid, None) for pid in args.pids]
            next_census = tick + 5
        tcp = sockets()
        rows = []
        for pid, worker in pids:
            row = proc(pid, tcp, details=tick-start < args.interval*2 or not args.remote)
            if row:
                row['worker_id'] = worker
                rows.append(row)
        out = dict(kind='sample', time=time.time(), processes=rows)
        if tick >= next_gpu:
            out['gpu'] = gpu()
            next_gpu = tick + (10 if args.remote else 1)
        if args.remote and int(tick-start) % 30 == 0:
            path = Path('/scratch/zixuans8/mw_r17/formal/runs')/(args.campaign+'_timan107')/'receipt.json'
            receipt = read(path)
            try:
                receipt = json.loads(receipt)
                out['fleet'] = {key: receipt.get(key) for key in ('status', 'worker_recycles', 'worker_failures', 'reason')}
            except (ValueError, TypeError):
                out['fleet_error'] = receipt
        out['probe_seconds'] = time.monotonic()-tick
        print(json.dumps(out), flush=True)
        time.sleep(max(0, args.interval-(time.monotonic()-tick)))


if __name__ == '__main__':
    main()
