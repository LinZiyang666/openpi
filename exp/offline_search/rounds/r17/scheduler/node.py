"""Node-side executor of the central scheduler: one per batch per node.

Runs the scheduler's target, holds the node's memory safety line and reports
node_state in AUTOSCALE_NODE.json. No remote calls; the only signal it sends
is SIGKILL to its own workers' process trees below critical / 2.
"""
from __future__ import annotations

from collections import deque
from contextlib import contextmanager
import fcntl
import json
import os
from pathlib import Path
import signal
import statistics
import time

GIB = 1024 ** 3
SAMPLE_S = 2  # fleet tick; spawns per tick: 1 + planned recycles since the previous tick
STALE_S = 90  # an older control keeps the previous target and never grows
SETTLED_S = 120  # live workers this old join the per-worker medians
HISTORY = 32  # ... with the last samples of this many finished generations
GROWTH_S = 300  # a worker younger than this may still grow to the per-worker RAM
RESOURCE_S = 10  # CPU and node-wide worker census period
# (floor, critical) bytes; campaign scheduler_node_config[node] overrides.
FLOORS = {'a100': (16 * GIB, 6 * GIB), 'timan107': (24 * GIB, 16 * GIB)}
FLOOR = (20 * GIB, 10 * GIB)
WORKER_RAM = {'libero': int(2.2 * GIB), 'metaworld': GIB}  # until a worker here is measured
OBSERVER = 'exp.offline_search.rounds.r17.scheduler.worker:Observer'
MARKER = ('OPENPI_WORKER_OBSERVER=' + OBSERVER).encode()
ROOT = Path('/tmp/openpi-autoscale-%d' % os.getuid())  # node lock, per-GPU EGL creation locks
PAGE = os.sysconf('SC_PAGE_SIZE')
HZ = os.sysconf('SC_CLK_TCK')


def read(path, default=None):
    try:
        return json.loads(Path(path).read_text())
    except (OSError, ValueError):
        return default


def atomic(path, value):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    part = path.with_name(path.name + '.%d.part' % os.getpid())
    part.write_text(json.dumps(value, sort_keys=True, allow_nan=False) + '\n')
    os.replace(part, path)


def event(path, node, delta, reason, signals, counts):
    """Append only; lock serializes supervisor/driver writers, including threads."""
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    with Path(path).open('a') as out:
        fcntl.flock(out, fcntl.LOCK_EX)
        out.write(json.dumps(dict(time=time.time(), node=node, delta=delta, reason=reason,
                                  signals=signals, counts=counts), sort_keys=True, allow_nan=False) + '\n')
        out.flush()


def stat(pid):
    """(start ticks, CPU ticks including reaped children, RSS bytes); None once gone or a zombie."""
    try:
        fields = Path('/proc/%d/stat' % pid).read_text().rsplit(')', 1)[1].split()
    except (OSError, IndexError):
        return None
    return None if fields[0] == 'Z' else (fields[19], sum(map(int, fields[11:15])), int(fields[21]) * PAGE)


def thread_budget(pid):
    """Cores a process may compute on: its OMP_NUM_THREADS, else every core."""
    try:
        environ = Path('/proc/%d/environ' % pid).read_bytes().split(b'\0')
        return max(1, int(next(e for e in environ if e.startswith(b'OMP_NUM_THREADS=')).split(b'=', 1)[1]))
    except (OSError, StopIteration, ValueError):
        return os.cpu_count()


def listeners(ports):
    """Socket inodes listening on any of `ports` (TCP, v4 and v6)."""
    inodes = set()
    for table in ('/proc/net/tcp', '/proc/net/tcp6'):
        try:
            rows = Path(table).read_text().splitlines()[1:]
        except OSError:
            continue
        for row in rows:
            fields = row.split()
            if fields[3] == '0A' and int(fields[1].rsplit(':', 1)[1], 16) in ports:
                inodes.add(fields[9])
    return inodes


def identity(pid):
    sample = stat(pid)
    return sample[0] if sample else None


def process_tree(pid, start_ticks):
    """One worker generation's process tree (PID-reuse safe): pids, summed RSS and CPU ticks."""
    if identity(pid) != str(start_ticks):
        return None
    tree = dict(pids=[], rss=0, cpu=0)
    todo = [pid]
    while todo:
        current = todo.pop()
        sample = None if current in tree['pids'] else stat(current)
        if sample is None:
            continue
        tree['pids'].append(current)
        tree['cpu'] += sample[1]
        tree['rss'] += sample[2]
        try:
            todo.extend(map(int, Path('/proc/%d/task/%d/children' % (current, current)).read_text().split()))
        except (OSError, ValueError):
            pass
    return tree


def stop_tree(pid, start_ticks):
    """SIGKILL one owned worker generation's process tree (PID-reuse safe)."""
    for current in (process_tree(pid, start_ticks) or dict(pids=[]))['pids']:
        try:
            os.kill(current, signal.SIGKILL)
        except ProcessLookupError:
            pass


def memory():
    fields = dict(line.split(':', 1) for line in Path('/proc/meminfo').read_text().splitlines())
    available = int(fields['MemAvailable'].split()[0]) * 1024
    # Also respect a tighter cgroup limit; /proc MemAvailable is host-wide.
    for line in Path('/proc/self/cgroup').read_text().splitlines():
        _, controllers, relative = line.split(':', 2)
        if not controllers:
            base, limit, used = Path('/sys/fs/cgroup') / relative.lstrip('/'), 'memory.max', 'memory.current'
        elif 'memory' in controllers.split(','):
            base, limit, used = Path('/sys/fs/cgroup/memory') / relative.lstrip('/'), 'memory.limit_in_bytes', 'memory.usage_in_bytes'
        else:
            continue
        try:
            available = min(available, max(0, int((base / limit).read_text()) - int((base / used).read_text())))
        except (OSError, ValueError):
            pass
    return available


def cpu_times(cpus):
    """Busy and total jiffies summed over the given CPUs."""
    busy = total = 0
    for line in Path('/proc/stat').read_text().splitlines():
        name, *values = line.split()
        if name[3:].isdigit() and name.startswith('cpu') and int(name[3:]) in cpus:
            values = list(map(int, values[:8]))
            total += sum(values)
            busy += sum(values) - values[3] - values[4]  # minus idle, iowait
    return busy, total


def card_reserve(total_mib):
    # Card reserve: 10% capped at 2 GiB (an 80 GB card does not need 8 GB idle).
    return max(512, min(2048, (total_mib + 9) // 10))


def median(values):
    return statistics.median(values) if values else None


class NodeExecutor:
    def __init__(self, campaign, run):
        self.campaign, self.run = campaign, Path(run)
        self.node = campaign['worker_node']
        self.maximum = campaign['node_workers']
        config = campaign.get('scheduler_node_config', {}).get(self.node)
        self.floor, self.critical = (config['floor'], config['critical']) if config else FLOORS.get(self.node, FLOOR)
        self.run.mkdir(parents=True, exist_ok=True)
        ROOT.mkdir(mode=0o700, parents=True, exist_ok=True)
        from .kernel_evidence import begin
        begin(self.run)
        self.started_wall = time.time()
        # A replacement fleet (swap_node_fleet.py) keeps the channel files the launcher feeds and polls.
        channel = Path(campaign.get('scheduler_channel_run') or self.run)
        self.snapshot_path = channel / 'AUTOSCALE_NODE.json'
        self.control_path = channel / 'AUTOSCALE_CONTROL.json'
        self.events = self.run / 'AUTOSCALE_DECISIONS.jsonl'
        self.entries, self.retired = {}, set()
        self.slots = None
        self.snapshot = {}
        self.target = 0
        self.done = self.fresh = False
        self.budget = self.recycled = 0
        self.available = self.growth = self.cost = 0
        self.history = deque(maxlen=HISTORY)  # last samples (ram, vram, cpu) of finished generations
        self.known = {}  # (pid, start ticks) -> is a scheduler worker
        self.cpu_sample, self.cpu = None, dict(capacity=None, own={})
        self.server_cpu = None  # per server process of this batch: CPU in % of its thread budget
        self.vram_capacity = None
        self.last_resources = float('-inf')

    @contextmanager
    def lock(self):
        with (ROOT / 'scheduler.lock').open('a') as stream:
            fcntl.flock(stream, fcntl.LOCK_EX)
            yield

    def counts(self):
        return dict(live=len(self.entries), retired=len(self.retired), target=self.target)

    def register(self, entry):
        entry = dict(entry)
        entry.setdefault('born', time.monotonic())
        entry.setdefault('rss', 0)
        entry.setdefault('pids', [entry['pid']])
        self.entries[entry['worker_id']] = entry
        self.budget -= 1
        self.growth += self.cost  # its whole footprint is still to come
        event(self.events, self.node, 1, 'spawn', dict(worker_id=entry['worker_id'], pid=entry['pid']), self.counts())

    def finished(self, worker_id, normal=False):
        entry = self.entries.pop(worker_id)
        self.retired.discard(worker_id)
        if self.snapshot_path.exists():
            # The next generation of this slot reuses the worker id: it must not find itself retired.
            self.snapshot['retired'] = sorted(self.retired)
            atomic(self.snapshot_path, self.snapshot)
        self.recycled += bool(normal)
        if entry.get('last'):
            self.history.append(entry['last'])
        marker = read(Path(entry['recycle_receipt']).with_name('PEAK.json'), {}) if entry.get('recycle_receipt') else {}
        peak = marker.get('peak') if (marker.get('pid'), marker.get('start_ticks')) == (entry['pid'], entry['start_ticks']) else None
        event(self.events, self.node, -1, 'generation_exit',
              dict(worker_id=worker_id, pid=entry['pid'], peak=peak, rss=entry['rss'], complete=normal,
                   killed=bool(entry.get('killed'))), self.counts())

    def _retire(self, entries, reason, signals):
        ids = sorted(e['worker_id'] for e in entries if e['worker_id'] not in self.retired)
        if ids:
            self.retired.update(ids)
            event(self.events, self.node, -len(ids), reason, dict(signals, workers=ids), self.counts())

    def _shed(self, newest, available, line, kill):
        """Newest workers first, until their RSS (plus what is already leaving) covers line - available."""
        leaving = (lambda e: e.get('killed')) if kill else (lambda e: e['worker_id'] in self.retired)
        released = sum(e['rss'] for e in newest if leaving(e))
        chosen = []
        for e in newest:
            if released >= line - available:
                break
            if not leaving(e):
                chosen.append(e)
                released += e['rss']
        signals = dict(available=available, line=line, released=released)
        if not kill:
            self._retire(chosen, 'retire_memory', signals)
            return
        for e in chosen:
            # Its episode is re-dispatched by the driver's disconnect fence; retired, so
            # the fleet does not count the exit as a failure.
            stop_tree(e['pid'], e['start_ticks'])
            e['killed'] = True
            self.retired.add(e['worker_id'])
        if chosen:
            event(self.events, self.node, -len(chosen), 'kill_memory',
                  dict(signals, workers=[e['worker_id'] for e in chosen]), self.counts())

    def _worker(self, pid, start):
        """Any batch's scheduler worker, by the Observer in its (inherited) environment."""
        key = (pid, start)
        if key not in self.known:
            try:
                self.known[key] = MARKER in Path('/proc/%d/environ' % pid).read_bytes().split(b'\0')
            except OSError:
                self.known[key] = False  # another user's process
        return self.known[key]

    def _vram(self, trees):
        """Usable renderer VRAM of this node's cards, and this batch's per-worker usage (MiB)."""
        view = self.slots.view() if self.slots is not None else None
        if view is None:
            return None, {}
        headroom = self.campaign.get('worker_server_vram_headroom_mib', 0)
        owner = {pid: worker_id for worker_id, tree in trees.items() for pid in tree['pids']}
        own, capacity = dict.fromkeys(trees, 0), 0
        for card in view['cards']:
            workers = 0
            for p in card['processes']:
                used = p.get('used_mib') or 0
                if p['pid'] in owner:
                    own[owner[p['pid']]] += used
                sample = stat(p['pid'])
                if p['pid'] in owner or sample and self._worker(p['pid'], sample[0]):
                    workers += used
            capacity += max(0, card['total_mib'] - card_reserve(card['total_mib']) - headroom
                            - (card['used_mib'] - workers))
        return capacity, own

    def _resources(self, now, trees):
        """CPU over the last RESOURCE_S: node capacity for workers and this batch's per-worker cores."""
        cpus = os.sched_getaffinity(0)
        busy, total = cpu_times(cpus)
        boot = float(Path('/proc/uptime').read_text().split()[0]) * HZ
        marked, seen = {}, set()
        for name in os.listdir('/proc'):
            sample = stat(int(name)) if name.isdigit() else None
            if sample is not None:
                key = (int(name), sample[0])
                seen.add(key)
                if self._worker(*key):
                    marked[key] = sample[1]
        self.known = {k: v for k, v in self.known.items() if k in seen}
        own = {self._generation(w): tree['cpu'] for w, tree in trees.items()}
        # This batch's model servers: our processes listening on its backend ports (never another batch's on the
        # same machine). A replica using its whole thread budget is CPU-bound however idle the GPU is (GR00T B1
        # replicas with OMP=1 at ~104%, the 4090 GPU at 72%, 10-20). Without known ports (R15/R16) nothing is reported.
        ports = set(self.campaign.get('backend_ports') or ())
        inodes = listeners(ports) if ports and self.node == self.campaign.get('server_host') else set()
        servers = {}
        for name in os.listdir('/proc') if inodes else ():
            if not name.isdigit():
                continue
            try:
                if Path('/proc/' + name).stat().st_uid != os.getuid() or not any(
                        os.readlink('/proc/%s/fd/%s' % (name, fd))[8:-1] in inodes for fd in os.listdir('/proc/%s/fd' % name)):
                    continue
            except OSError:
                continue
            sample = stat(int(name))
            if sample:
                servers[(int(name), sample[0])] = (sample[1], thread_budget(int(name)))
        previous, self.cpu_sample = self.cpu_sample, dict(time=now, busy=busy, total=total, boot=boot,
                                                          marked=marked, own=own, servers=servers)
        if previous is None:
            return
        if inodes:
            span = (now - previous['time']) * HZ / 100
            self.server_cpu = sorted(round((ticks - previous['servers'][key][0]) / span / budget)
                                     for key, (ticks, budget) in servers.items() if key in previous['servers'])
        def cores(old, key, ticks):
            # A process first seen in this window counts whole only if it started inside it.
            delta = max(0, ticks - old[key]) if key in old else ticks if int(key[1]) >= previous['boot'] else 0
            return delta / HZ / (now - previous['time'])
        workers = sum(cores(previous['marked'], k, t) for k, t in marked.items())
        load = (busy - previous['busy']) / max(1, total - previous['total']) * len(cpus)
        self.cpu = dict(capacity=len(cpus) - max(0, load - workers),
                        own={k: cores(previous['own'], k, t) for k, t in own.items()})

    def _generation(self, worker_id):
        entry = self.entries[worker_id]
        return entry['pid'], str(entry['start_ticks'])

    def _sample(self):
        now, wall = time.monotonic(), time.time()
        self.available = available = memory()
        trees = {}
        for worker_id, entry in self.entries.items():
            tree = process_tree(entry['pid'], entry['start_ticks'])
            if tree is not None:
                trees[worker_id] = tree
                entry['rss'], entry['pids'] = tree['rss'], tree['pids']
        if now - self.last_resources >= RESOURCE_S:
            self.last_resources = now
            self._resources(now, trees)
        vram_capacity, vram = self._vram(trees)
        # A missed inventory keeps the last capacity: an empty reading would zero the node for every batch.
        self.vram_capacity = self.vram_capacity if vram_capacity is None else vram_capacity
        cpu = {w: self.cpu['own'][self._generation(w)] for w in trees if self._generation(w) in self.cpu['own']}
        for worker_id, tree in trees.items():
            last = self.entries[worker_id].setdefault('last', {})
            last['ram'] = tree['rss']
            if worker_id in vram:
                last['vram'] = vram[worker_id]
            if worker_id in cpu:
                last['cpu'] = cpu[worker_id]
        samples = [self.entries[w]['last'] for w in trees if now - self.entries[w]['born'] >= SETTLED_S]
        samples += self.history
        worker = {k: median([s[k] for s in samples if k in s]) for k in ('ram', 'vram', 'cpu')}
        control = read(self.control_path, {}) or {}
        self.fresh = (control.get('campaign') == self.campaign['run_name'] and
                      0 <= wall - control.get('time', 0) <= STALE_S)
        self.done = self.done or self.fresh and bool(control.get('done'))
        if self.fresh:
            target = 0 if self.done else min(self.maximum, max(0, int(control.get('targets', {}).get(self.node, 0))))
            if target != self.target:
                event(self.events, self.node, target - self.target, 'target',
                      dict(epoch=control.get('epoch'), sequence=control.get('sequence')), dict(self.counts(), target=target))
                self.target = target
        newest = sorted(self.entries.values(), key=lambda e: e['born'], reverse=True)
        if available < self.critical / 2:
            self._shed(newest, available, self.critical, kill=True)
        if available < self.critical:
            self._shed(newest, available, self.floor, kill=False)
        active = [e for e in newest if e['worker_id'] not in self.retired]
        if len(active) > self.target:
            self._retire(active[:len(active) - self.target], 'retire_target', dict(target=self.target))
            active = active[len(active) - self.target:]
        # A young worker has not reached its footprint yet: keep room for the rest of it.
        self.cost = worker['ram'] or WORKER_RAM[self.campaign['domain']]
        self.growth = sum(max(0, self.cost - e['rss']) for e in active if now - e['born'] < GROWTH_S)
        # Replacing planned recycles keeps pace with them; growth stays at one per tick.
        self.budget, self.recycled = 1 + self.recycled, 0
        self.snapshot = dict(
            node=self.node, time=wall, campaign=self.campaign['run_name'], active=len(active), target=self.target,
            retired=sorted(self.retired), done=self.done,
            control_epoch=control.get('epoch'), control_sequence=control.get('sequence', 0),
            ram_available=available, ram_floor=self.floor,
            workers_ram=sum(tree['rss'] for tree in trees.values()), worker_ram=worker['ram'],
            vram_capacity=self.vram_capacity, workers_vram=sum(vram.values()), worker_vram=worker['vram'],
            cpu_capacity=self.cpu['capacity'], workers_cpu=sum(cpu.values()), worker_cpu=worker['cpu'],
            gpu_util=self.slots.gpu_util() if self.slots is not None else None, server_cpu=self.server_cpu)
        # The scheduler takes a node's capacity from its freshest snapshot: publish only once measured.
        if self.vram_capacity is not None and self.cpu['capacity'] is not None:
            atomic(self.snapshot_path, self.snapshot)
        if self.last_resources == now:
            event(self.run / 'AUTOSCALE_SAMPLES.jsonl', self.node, 0, 'node_sample',
                  {k: v for k, v in self.snapshot.items() if k != 'retired'},
                  dict(self.counts(), growth=self.growth, history=len(self.history)))
        return self.snapshot

    def tick(self):
        with self.lock():
            return self._sample()

    @contextmanager
    def permit(self):
        """Hold the node lock through Popen + register. Fresh target only; each spawn uses the
        tick's budget and charges its RAM, so several in one tick still pass the memory check."""
        with self.lock():
            yield (self.fresh and self.budget > 0 and len(self.entries) - len(self.retired) < self.target
                   and self.available - self.floor - self.growth >= self.cost)

    def worker_env(self, worker_id):
        return dict(OPENPI_WORKER_OBSERVER=OBSERVER,
                    OPENPI_AUTOSCALE_NODE=str(self.snapshot_path), OPENPI_AUTOSCALE_CONTROL=str(self.control_path),
                    OPENPI_AUTOSCALE_CAMPAIGN=self.campaign['run_name'],
                    OPENPI_AUTOSCALE_MAX_EPISODES=str(self.campaign.get('worker_max_episodes', 32)),
                    OPENPI_AUTOSCALE_MAX_SECONDS=str(self.campaign.get('worker_max_seconds', 1800)))
