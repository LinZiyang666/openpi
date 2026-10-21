"""Periodic server-load check (owner 10-20: analyse every server's load, do not starve servers).

Every PERIOD_S: per-second GPU utilization of the 4090 and the H100, 4090/H100 ingress, the scheduler's view of
every batch (desired, active, target per node), each batch's accepted-episode rate (new journal bytes only),
and every worker machine's load / free RAM / uplink. One JSON line per round goes to LOG. The script prints an
ALERT summary and exits when a server is starved (GPU below STARVED_UTIL while worker machines have room), a
batch's rate halves, or a running batch holds no workers for two rounds; the coordinator then adjusts and
restarts it.
"""
import json
import os
import shlex
import subprocess
import sys
import time
from pathlib import Path

PERIOD_S = 600
STARVED_UTIL = 60
LOG = Path('/home/weiland/projects/openpi_ext/scratch/r17_ops/server_watch.log')
EVAL = (Path('/data/mw_finetune/eval'), Path('/nvme/weiland/openpi/eval'))
WORKER_NODES = ('timan107', 'timan1', 'a100')
CHAIN_LOG = Path('/home/weiland/projects/openpi_ext/scratch/r17_ops/mw/chain.log')


def sh(argv, timeout=60):
    try:
        return subprocess.run(argv, capture_output=True, text=True, timeout=timeout).stdout
    except subprocess.TimeoutExpired:
        return ''


def remote(node, script, timeout=60):
    return sh(['timeout', str(timeout), 'tether', 'exec', node, '--', 'bash', '-lc', script], timeout + 5)


PROBE = r"""
import json, os, subprocess, time
def run(*a):
    try: return subprocess.run(a, capture_output=True, text=True, timeout=20).stdout
    except Exception: return ''
iface = run('ip', 'route', 'get', '8.8.8.8').split('dev ')[1].split()[0]
net = lambda k: int(open('/sys/class/net/%s/statistics/%s_bytes' % (iface, k)).read())
pids = [int(x) for x in run('nvidia-smi', '--query-compute-apps=pid', '--format=csv,noheader').split()]
def ticks(pid):
    try:
        f = open('/proc/%d/stat' % pid).read().rsplit(')', 1)[1].split()
        return int(f[11]) + int(f[12])
    except Exception: return None
t0, r0, x0, c0 = time.time(), net('rx'), net('tx'), {p: ticks(p) for p in pids}
gpu = []
for _ in range(10):
    out = run('nvidia-smi', '--query-gpu=utilization.gpu', '--format=csv,noheader,nounits').split()
    if out: gpu.append(max(float(v) for v in out))
    time.sleep(1)
dt, hz = time.time() - t0, os.sysconf('SC_CLK_TCK')
cpu = sorted(round(100 * (ticks(p) - c0[p]) / hz / dt) for p in pids if c0[p] is not None and ticks(p) is not None)
mem = {l.split(':')[0]: int(l.split()[1]) for l in open('/proc/meminfo')}
print(json.dumps(dict(rx_mb=round((net('rx') - r0) / dt / 2**20, 1), tx_mb=round((net('tx') - x0) / dt / 2**20, 1),
    gpu_mean=round(sum(gpu) / len(gpu), 1) if gpu else None, gpu_min=min(gpu) if gpu else None,
    gpu_max=max(gpu) if gpu else None, server_cpu=cpu, load=os.getloadavg()[0], cores=os.cpu_count(),
    ram_free_gib=mem['MemAvailable'] // 2**20)))
"""


def probe(node):
    argv = ['python3', '-c', PROBE]
    out = sh(argv, 40) if node == 'local' else remote(node, 'python3 -c ' + shlex.quote(PROBE))
    line = next((l for l in out.splitlines()[::-1] if l.startswith('{')), None)
    return json.loads(line) if line else None


def scheduler():
    out = remote('a100', 'curl -s http://127.0.0.1:23500/v1/state')
    try:
        return json.loads(out[out.index('{'):])
    except ValueError:
        return None


class Rates:
    """Accepted episodes per minute from the bytes appended to each batch journal since the last round."""

    def __init__(self):
        self.offset, self.seen = {}, {}

    def journal(self, batch):
        for root in EVAL:
            for rel in ('driver/journal.jsonl', 'journal.jsonl'):
                p = root / batch / rel
                if p.exists():
                    return p
        return None

    def rate(self, batch, now):
        p = self.journal(batch)
        if p is None:
            return None
        size = p.stat().st_size
        start = self.offset.get(batch)
        self.offset[batch] = size
        if start is None or size < start:
            self.seen[batch] = now
            return None
        with open(p, 'rb') as f:
            f.seek(start)
            data = f.read(size - start).decode(errors='replace')
        n = data.count('"accepted": true') or (data.count('"status": "done"') + data.count('"status": "failed"'))
        minutes = (now - self.seen[batch]) / 60
        self.seen[batch] = now
        return round(n / minutes, 1) if minutes > 0 else None


def main():
    rates, last, idle, starved = Rates(), {}, {}, {}
    chain_offset = CHAIN_LOG.stat().st_size  # only failures from now on
    while True:
        now = time.time()
        servers = {'local': probe('local'), 'h100': probe('h100')}
        workers = {n: probe(n) for n in WORKER_NODES}
        state = scheduler() or {}
        batches = {}
        for b, r in (state.get('reports') or {}).items():
            active = sum((r.get('active') or {}).values())
            batches[b] = dict(server=r.get('server_host'), desired=r.get('desired'), active=active,
                              target=sum((state.get('targets', {}).get(b) or {}).values()), done=r.get('done'),
                              rate=rates.rate(b, now))
        row = dict(time=time.strftime('%m-%d %H:%M:%S'), servers=servers, workers=workers, batches=batches)
        LOG.parent.mkdir(parents=True, exist_ok=True)
        with LOG.open('a') as out:
            out.write(json.dumps(row) + '\n')
        alerts = []
        # A chain that stops on a failure leaves its machine idle until someone restarts it (10-21: 4090 idle 3.5 h).
        size = CHAIN_LOG.stat().st_size
        with CHAIN_LOG.open() as f:
            f.seek(chain_offset)
            alerts += [l.strip() for l in f.read().splitlines() if 'STOPPED after failure' in l or 'FAILED' in l]
        chain_offset = size
        room = [n for n, w in workers.items() if w and w['load'] < 0.8 * w['cores'] and w['ram_free_gib'] > 30]
        for s, d in servers.items():
            serving = [b for b, v in batches.items() if v['server'] == s and not v['done']]
            # A replica is one host process; at ~100% CPU it is saturated even with the GPU half idle (GR00T).
            replicas_busy = bool(d and d['server_cpu']) and sum(c >= 85 for c in d['server_cpu']) >= len(d['server_cpu']) / 2
            hungry = bool(d and serving and d['gpu_mean'] is not None and d['gpu_mean'] < STARVED_UTIL and not replicas_busy and room)
            # Two rounds in a row: an R15 arm or a batch starting from zero workers is not starvation.
            starved[s] = starved.get(s, 0) + 1 if hungry else 0
            if starved[s] >= 2:
                alerts.append(f'{s} GPU {d["gpu_mean"]}% (min {d["gpu_min"]}), replica CPU {d["server_cpu"]} '
                              f'serving {serving}; room on {room} (2 rounds)')
        for b, v in batches.items():
            if v['done']:
                continue
            if v['rate'] is not None and last.get(b) and v['rate'] < 0.5 * last[b]:
                alerts.append(f'{b} rate {last[b]} -> {v["rate"]}/min (active {v["active"]}, target {v["target"]})')
            idle[b] = idle.get(b, 0) + 1 if v['active'] == 0 else 0
            if idle[b] >= 2:
                alerts.append(f'{b} has no active workers for {idle[b]} rounds (desired {v["desired"]})')
            if v['rate'] is not None:
                last[b] = v['rate']
        if alerts:
            print(row['time'], 'ALERT', *alerts, sep='\n  ', flush=True)
            print('  batches', json.dumps(batches), flush=True)
            print('  servers', json.dumps(servers), '\n  workers', json.dumps(workers), flush=True)
            return 0
        time.sleep(max(0, PERIOD_S - (time.time() - now)))


if __name__ == '__main__':
    sys.exit(main())
