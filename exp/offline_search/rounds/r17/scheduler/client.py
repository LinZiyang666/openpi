"""Driver-side hook (conductor `_autoscaler`): the batch's desire, its report to the central scheduler, and
AUTOSCALE_CONTROL.json carrying the scheduler's targets to the batch's NodeExecutors."""
from __future__ import annotations

from collections import deque
import json
import math
import os
from pathlib import Path
import threading
import time
import urllib.request


TICK_S = 2
REPORT_S = 15
DESIRE_S = 30
FRESH_S = 90  # a node snapshot older than this no longer counts
START = 64  # first desire (10-20: 16 left a fresh LIBERO batch ~15 min below its ~100-worker need while it doubled)
SETTLE_S = 120  # after a step, wait at most this long for live workers to reach 95% of desired
MEASURE_S = 180  # accepted-rate window after a step
HOLD_S = 300  # time in hold before one probe step (owner 10-20: keep exploring upward)
STABLE_TOLERANCE = .15  # the shared pool is steady once its active workers reach 85% of the targets' total
MIX_TOLERANCE = .2  # windows whose cohort (yaml) shares differ more than this are not comparable
STATE = ('time', 'ram_available', 'ram_floor', 'workers_ram', 'worker_ram', 'vram_capacity', 'workers_vram',
         'worker_vram', 'cpu_capacity', 'workers_cpu', 'worker_cpu', 'gpu_util', 'server_cpu')
OPENER = urllib.request.build_opener(urllib.request.ProxyHandler({}))  # public address directly, never a proxy


def read(path, default=None):
    try:
        return json.loads(Path(path).read_text())
    except (OSError, ValueError):
        return default


def atomic(path, value):
    path = Path(path)
    part = path.with_name(path.name + '.%d.part' % os.getpid())
    part.write_text(json.dumps(value, sort_keys=True, allow_nan=False) + '\n')
    os.replace(part, path)


def helped(before, after):
    """A step helped if the accepted rate rose at least 5%.

    The old bar (a third of the proportional share) failed every large step: workers added at the step have not
    finished their first episode inside the 180 s window (10-20: 20 -> 92 workers, 29.5 -> 45.4 per minute, judged
    a failure and cut back to 20)."""
    (rate0, active0), (rate1, active1) = before[:2], after[:2]
    return active0 <= 0 or rate1 >= 1.05 * rate0


def comparable(before, after):
    """Rates of two windows compare only if no cohort's share moved more than MIX_TOLERANCE (empty windows do)."""
    return not before or not after or all(abs(before.get(c, 0) - after.get(c, 0)) <= MIX_TOLERANCE
                                          for c in set(before) | set(after))


class BatchClient:
    def __init__(self, driver, campaign, directory, *, clock=time.monotonic, wall=time.time):
        self.driver, self.campaign, self.root = driver, campaign, Path(directory)
        self.clock, self.wall = clock, wall
        self.batch = campaign['run_name']
        self.url = campaign['scheduler_url'].rstrip('/') + '/v1/report'
        self.nodes = campaign['scheduler_nodes']
        self.lock = threading.RLock()
        self.results_lock = threading.Lock()
        self.accepted = deque()
        self.snapshots = {}
        self.retired = set()
        self.targets = dict.fromkeys(self.nodes, 0)
        self.done = False
        self.desired, self.phase, self.since, self.base, self.samples = 0, 'start', self.clock(), None, []
        self.failed = False  # a step has not paid off yet: growth doubles
        self.stable_since = None
        self.last_tick = self.last_desire = self.last_report = float('-inf')
        self.posting = False
        self.epoch = '%d-%d' % (self.wall() * 1000000, os.getpid())
        self.sequence = 0
        self.control = {}
        self.events = self.root / 'SCHEDULER_DECISIONS.jsonl'
        self.root.mkdir(parents=True, exist_ok=True)
        # A later process of the same batch (R15 arm, restarted driver) starts where the last one ended.
        lines = self.events.read_text().splitlines() if self.events.is_file() else []
        self.resume = next((row['desired'] for row in map(json.loads, reversed(lines))
                            if row.get('kind') == 'desire' and row.get('desired')), START)
        self.publish()

    def note(self, kind, **fields):
        with self.events.open('a') as out:
            out.write(json.dumps(dict(fields, time=self.wall(), kind=kind), sort_keys=True) + '\n')

    def ingest(self, data):
        row = data.get('node')
        if (row and row.get('node') in self.nodes and row.get('campaign') == self.batch
                and row.get('time', 0) >= self.snapshots.get(row['node'], {}).get('time', 0)):
            self.snapshots[row['node']] = row

    def pull(self, payload):
        with self.lock:
            data = payload.get('_autoscale', {})
            self.ingest(data)
            self.tick()
            return bool(data.get('retire')), dict(self.control)

    def result(self, payload, accepted):
        if accepted:
            cohort = (payload.get('_autoscale_episode') or {}).get('cohort') or ''
            with self.results_lock:
                self.accepted.append((self.clock(), cohort))

    def fresh(self, snapshot):
        return self.wall() - snapshot.get('time', 0) <= FRESH_S

    def window(self, start, end):
        """Accepted episodes per minute in (start, end] and each cohort's share of them."""
        with self.results_lock:
            while self.accepted and self.accepted[0][0] < end - 2 * MEASURE_S:
                self.accepted.popleft()
            cohorts = [c for t, c in self.accepted if start < t <= end]
        return len(cohorts) * 60 / max(end - start, 1), {c: cohorts.count(c) / len(cohorts) for c in set(cohorts)}

    def tick(self):
        with self.lock:
            now = self.clock()
            done = self.driver.scheduler.all_done()
            if now - self.last_tick < TICK_S and done == self.done:
                return
            self.last_tick = now
            for row in (read(self.root / 'AUTOSCALE_NODES.json') or {}).values():
                self.ingest(dict(node=row))
            self.retired = set((read(self.root / 'WORKER_NODE_STATE.json') or {}).get('retired', []))
            active = {n: self.snapshots[n]['active'] if n in self.snapshots and self.fresh(self.snapshots[n]) else 0
                      for n in self.nodes}
            # Totals over the shared pool: per-node equality never held, since the scheduler moves a few
            # targets every round (10-20: shared_stable_s stayed 0, so no server host ever got local workers).
            shared = [n for n, v in self.nodes.items() if v['class'] == 'shared']
            want = sum(self.targets[n] for n in shared)
            steady = sum(active[n] for n in shared) >= want * (1 - STABLE_TOLERANCE)
            self.stable_since = (self.stable_since if self.stable_since is not None else now) if steady else None
            if done and not self.done:
                self.done, self.desired = True, 0
                self.note('done')
                self.apply({})
            elif not done and now - self.last_desire >= DESIRE_S:
                self.last_desire = now
                self.evaluate(now, sum(active.values()))
            if now - self.last_report >= REPORT_S and not self.posting:
                self.last_report, self.posting = now, True
                threading.Thread(target=self.exchange, args=(self.report(now, active),), daemon=True).start()

    def evaluate(self, now, active):
        """HPA-style desire: start, grow while steps pay off, hold, probe; re-evaluated every 30 s."""
        progress = self.driver.scheduler.progress()
        remaining = progress['episodes_total'] - progress['episodes_done'] - progress['episodes_failed']
        host = self.snapshots.get(self.campaign['server_host'])
        # Only the measured step verdict moves the desire. A "busy" shortcut that froze it at the active count
        # misread both signals: replica CPU against OMP_NUM_THREADS (Fast-WAM at 115% of 4 threads with 12 workers,
        # GPU 8%) and GPU utilization (H100 at 99% and 215 W while fa_ls_fw_f_f_l_l fell from 57-75 to 30 workers
        # and 82 to 36/min, 10-21). A saturated server fails its probes and returns to the last count that paid off.
        before = self.phase, self.desired
        if self.phase == 'start':
            self.desired, self.phase, self.since = min(remaining, max(START, self.resume)), 'settle', now
        elif self.phase == 'settle':
            if active >= .95 * self.desired or now - self.since >= SETTLE_S:
                self.phase, self.since, self.samples = 'measure', now, [active]
        elif self.phase == 'measure':
            self.samples.append(active)
            if now - self.since >= MEASURE_S:
                rate, mix = self.window(self.since, now)
                point = (rate, sum(self.samples) / len(self.samples), mix)
                mixed = self.base is not None and not comparable(self.base[2], mix)
                if mixed:
                    # Another yaml mix, not this step, may have moved the rate: no verdict, re-base and go on.
                    self.note('incomparable', base_mix=self.base[2], mix=mix, rate=rate, base_rate=self.base[0])
                if self.base is None or mixed or helped(self.base, point):
                    self.base = point
                    # Double until a step first fails to pay off, then grow by a quarter.
                    step = max(4, math.ceil(active / 4)) if self.failed else max(8, active)
                    self.desired, self.phase, self.since = active + step, 'settle', now
                else:
                    # The first claim is the last count that paid off; the workers above it stay through the
                    # allocator's work-conserving pass while nobody else needs them, so nothing is cut back
                    # (owner 10-20) but idle-waiting workers no longer outrank another batch's starving server
                    # (10-21: pod workers at 0.006 cores held timan107's RAM, the 4090 R15 batch got 26).
                    self.failed = True
                    self.desired, self.phase, self.since, self.samples = round(self.base[1]), 'hold', now, []
                self.note('step', rate=point[0], mean_active=point[1], base=self.base)
        elif now - self.since < HOLD_S:
            self.samples.append(active)
        else:
            # Base on the mean count over the rate window: workers leaving during a hold still deliver their last
            # episodes, so rate over the instantaneous count read a drain as a fast small fleet and every later step
            # as a loss (10-23 w3_pi_s10: 43 -> 4 workers, 18 -> 15 per minute at a server-bound 4090).
            rate, mix = self.window(now - MEASURE_S, now)
            recent = self.samples[-max(1, MEASURE_S // DESIRE_S):] or [active]
            self.base = (rate, sum(recent) / len(recent), mix)
            self.desired, self.phase, self.since = active + max(2, math.ceil(active / 10)), 'settle', now
        ceiling = min(remaining, sum(v['max'] for n, v in self.nodes.items() if n not in self.retired))
        self.desired = max(1, min(self.desired, ceiling))
        if (self.phase, self.desired) != before:
            self.note('desire', phase=self.phase, desired=self.desired, active=active, remaining=remaining,
                      gpu_util=host.get('gpu_util') if host else None)

    def report(self, now, active):
        nodes = {n: v for n, v in self.nodes.items() if n not in self.retired}
        return dict(batch=self.batch, worker_type=self.campaign['domain'], server_host=self.campaign['server_host'],
                    nodes=nodes, desired=self.desired, active={n: active[n] for n in nodes},
                    shared_stable_s=now - self.stable_since if self.stable_since is not None else 0., done=self.done,
                    node_state={n: {k: s.get(k) for k in STATE} for n, s in self.snapshots.items()
                                if n in nodes and self.fresh(s)})

    def exchange(self, report):
        """POST outside the lock (pulls never wait on the network); unreachable keeps the last targets."""
        targets = error = None
        try:
            request = urllib.request.Request(self.url, json.dumps(report).encode(), {'Content-Type': 'application/json'})
            with OPENER.open(request, timeout=10) as response:
                targets = json.loads(response.read())['targets']
        except Exception as exc:
            error = repr(exc)
        with self.lock:
            self.posting = False
            if error:
                self.note('report_failed', error=error, url=self.url)
            if targets is None:
                self.publish()
            else:
                self.apply(targets)

    def apply(self, targets):
        targets = {n: 0 if self.done or n in self.retired else int(targets.get(n, 0)) for n in self.nodes}
        if targets != self.targets:
            self.note('targets', targets=targets, previous=self.targets)
        self.targets = targets
        self.publish()

    def publish(self):
        self.sequence += 1
        self.control = dict(campaign=self.batch, epoch=self.epoch, sequence=self.sequence,
                            time=max(self.wall(), self.control.get('time', 0) + .001),
                            targets=dict(self.targets), done=self.done)
        atomic(self.root / 'AUTOSCALE_CONTROL.json', self.control)


def from_environment(driver):
    path = os.environ.get('OPENPI_AUTOSCALE_CONFIG')
    if not path:
        return None
    return BatchClient(driver, json.loads(Path(path).read_text()), Path(path).parent)
