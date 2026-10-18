"""Global accepted-rate controller, fed by existing pull/progress and status paths."""
from __future__ import annotations

from collections import deque
import json
import os
from pathlib import Path
import threading
import time

from .node import atomic, event, read
from .policy import Config, NODES, Tuner, place


class DriverController:
    def __init__(self, driver, campaign, directory, *, clock=time.monotonic, wall=time.time):
        self.driver, self.campaign, self.root = driver, campaign, Path(directory)
        self.config = Config(**campaign.get('autoscale', {}))
        self.clock, self.wall = clock, wall
        self.caps = {n: int(v) for n, v in campaign.get('execution_worker_nodes', campaign['worker_nodes']).items() if n in NODES}
        self.maximum = int(campaign.get('autoscale_maximum', campaign['workers']))
        if self.maximum < 1 or any(v < 1 for v in self.caps.values()):
            raise ValueError('positive desired maxima required')
        # A node lost before launch can reduce prepared capacity below the
        # configured global maximum; continue within the surviving node caps.
        self.tuner = Tuner(self.maximum, self.config)
        self.nodes, self.results = {}, deque()
        self.targets = place(self.maximum, {n: dict(cap=v, headroom=v) for n, v in self.caps.items()}, min_slice=self.config.min_slice)
        self.demand_targets = dict(self.targets)
        self.charged = dict(self.targets)
        self.revisions = dict.fromkeys(self.caps, 0)
        self.lock = threading.RLock()
        self.results_lock = threading.Lock()
        self.last_tick = self.last_window = self.clock()
        self.last_distribution = None
        self.last_counts = None
        self.active_anchor, self.stable_peak = {}, {}
        self.changed = self.clock()
        self.control = {}
        self.root.mkdir(parents=True, exist_ok=True)
        self.epoch = '%d-%d' % (self.wall() * 1000000, os.getpid())
        self.sequence = 0
        for node, maximum in self.targets.items():
            event(self.root / 'AUTOSCALE_DECISIONS.jsonl', node, maximum, 'initial_desired_maximum',
                  dict(memory_admission='node_local_before_every_spawn'), dict(target=maximum, active=0, maximum=self.maximum))
        self.publish(False)

    def ingest(self, data):
        row = data.get('node') if isinstance(data, dict) else None
        if not isinstance(row, dict) or row.get('node') not in self.caps or row.get('campaign') != self.campaign['run_name']:
            return
        node = row['node']
        if 0 <= self.wall() - row.get('time', 0) <= self.config.stale_s:
            old = self.nodes.get(node, {})
            if row['time'] >= old.get('time', 0):
                self.nodes[node] = row

    def pull(self, payload):
        with self.lock:
            data = payload.get('_autoscale', {})
            self.ingest(data)
            self.tick()
            return bool(data.get('retire')), dict(self.control)

    def result(self, payload, accepted):
        timing = payload.get('_autoscale_episode')
        if not accepted or not isinstance(timing, dict):
            return
        # This call happens after the scheduler's attempt fence and durable
        # terminal evidence, not merely after receiving a wire result.
        with self.results_lock:
            self.results.append((self.clock(), dict(timing)))
        if timing.get('node'):
            event(self.root/'AUTOSCALE_ACCEPTED.jsonl', timing['node'], 0, 'accepted_episode',
                dict(task_uid=timing.get('task_uid'), attempt=timing.get('attempt')),
                dict(accepted=1))

    def publish(self, done):
        self.sequence += 1
        self.control = dict(campaign=self.campaign['run_name'], epoch=self.epoch, sequence=self.sequence,
                            time=self.wall(), targets=dict(self.targets), demand_targets=dict(self.demand_targets),
                            maximum=self.maximum, done=done)
        atomic(self.root / 'AUTOSCALE_CONTROL.json', self.control)

    def tick(self):
        with self.lock:
            now = self.clock()
            done = self.driver.scheduler.all_done()
            if now - self.last_tick < self.config.sample_s and not (done and not self.control.get('done')):
                return
            self.last_tick = now
            for row in (read(self.root / 'AUTOSCALE_NODES.json', {}) or {}).values():
                self.ingest(dict(node=row))
            retired = set((read(self.root / 'WORKER_NODE_STATE.json', {}) or {}).get('retired', []))
            rows, active = {}, 0
            for node, cap in self.caps.items():
                s = self.nodes.get(node, {})
                fresh = 0 <= self.wall() - s.get('time', 0) <= self.config.stale_s
                active += s.get('active', 0)
                free_growth_budget = max(0, s.get('headroom', 0))
                peak = max(1, s.get('projected_peak', 1))
                # Node headroom already subtracts the future growth of ALL live
                # workers. Existing slots must be added back as slots, not RSS/P.
                # RSS/P would impose an artificial cap during the memory ramp.
                fit = s.get('active', 0) + int(free_growth_budget // peak)
                headroom = free_growth_budget + s.get('active', 0) * peak
                if not s.get('admit', False):fit = min(fit, s.get('active', 0))
                rows[node] = dict(cap=min(cap, fit), headroom=headroom, valid=fresh and node not in retired)
            counts = tuple(self.targets.items())
            observed = {n:self.nodes.get(n, {}).get('active', 0) for n in self.caps}
            within_band = all(abs(self.nodes.get(n, {}).get('active', 0) - target)
                              <= max(1, target * .05) for n, target in counts)
            # N4/N32 recycling briefly changes active by one. Requiring every
            # exact count to stay unchanged for 180s prevents all future probes.
            stable_active = all(abs(observed[n] - self.active_anchor.get(n, 0))
                                <= max(1, self.active_anchor.get(n, 0) * .05) for n in self.caps)
            if counts != self.last_counts or not stable_active:
                self.changed = now
                self.active_anchor, self.stable_peak = dict(observed), dict(observed)
            else:
                self.stable_peak = {n:max(self.stable_peak.get(n, 0), observed[n]) for n in self.caps}
            self.last_counts = counts
            with self.results_lock:
                while self.results and self.results[0][0] <= now - self.config.window_s:
                    self.results.popleft()
                window_results = list(self.results)
            reason = 'bootstrap_or_memory_placement'
            if now - self.last_window >= self.config.window_s:
                self.last_window = now
                samples = [v for _, v in window_results]
                distribution = {}
                for sample in samples:
                    key = sample.get('cohort', '')
                    distribution[key] = distribution.get(key, 0) + 1 / max(1, len(samples))
                comparable = self.last_distribution is None or all(
                    abs(distribution.get(k, 0) - self.last_distribution.get(k, 0)) <= .2
                    for k in distribution.keys() | self.last_distribution.keys())
                self.last_distribution = distribution
                if not comparable:
                    self.tuner.previous = None
                average = lambda key: sum(s.get(key, 0) for s in samples) / max(1, len(samples))
                service = (average('service_s') if samples and all(s.get('service_s') is not None for s in samples) else None)
                # Scheduler in-flight episodes is an explicitly labelled proxy,
                # not an invented count of requests inside the H100 queue.
                scheduler = self.driver.scheduler
                busy = sum(scheduler.busy().values()) if hasattr(scheduler, 'busy') else len(getattr(scheduler, '_dispatch_ts', {}))
                settled = now - self.changed >= max(0, self.config.window_s-self.config.sample_s) and all(r['valid'] for n,r in rows.items()
                    if self.targets[n] or self.nodes.get(n, {}).get('active', 0))
                measured_capacity = (sum(self.targets.values()) if within_band else sum(self.stable_peak.values())) if settled else active
                probing = bool(self.tuner.exploring_down or (self.tuner.previous and self.tuner.target > self.tuner.previous[0]))
                _, reason = self.tuner.update(now, active=measured_capacity,
                    rate=len(samples) / self.config.window_s, samples=len(samples),
                    sim_s=average('sim_render_s'), wait_s=average('server_wait_s'), service_s=service,
                    queue_inflight=busy, comparable=comparable,
                    settled=settled)
                event(self.root / 'AUTOSCALE_DECISIONS.jsonl', 'global', 0, reason,
                      dict(rate=len(samples) / self.config.window_s, samples=len(samples),
                           episode_inflight=busy, sim_s=average('sim_render_s'), wait_s=average('server_wait_s'),
                           service_s=service, comparable=comparable, settled=settled,
                           measured_capacity=measured_capacity, probing=probing,
                           active_tolerance_fraction=.05, queue_floor=self.tuner.queue_floor,
                           rate_feedback=self.tuner.feedback,
                           window_s=self.config.window_s, window_end=self.wall(),
                           calibrated=all(s.get('calibrated') for n,s in self.nodes.items()
                                          if self.targets[n] or s.get('active', 0))),
                      dict(active=active, desired=self.tuner.target, maximum=self.maximum))
            # Demand is the tuner's intent, independent of physically free RAM.
            # Borrowed capacity is still resident until an episode boundary;
            # clamping intent to current admissions would never reclaim it.
            intent_rows = {n: dict(cap=cap, headroom=max(1,
                self.nodes.get(n, {}).get('fairness', {}).get('capacity', {}).get('ram',
                    rows[n]['headroom'] or cap)), valid=n not in retired)
                for n, cap in self.caps.items()}
            self.demand_targets = place(self.tuner.target, intent_rows, min_slice=self.config.min_slice)
            desired = (dict(self.targets) if not self.nodes else
                       place(self.tuner.target, rows, min_slice=self.config.min_slice))
            # Do not move quotas on every tiny fluctuation in MemAvailable.
            # Keep current placements if they fit and the desired total is steady.
            if sum(self.targets.values()) == self.tuner.target and all(rows[n]['valid'] and self.targets[n] <= rows[n]['cap'] for n in self.targets if self.targets[n]):
                desired = dict(self.targets)
            done = self.driver.scheduler.all_done()
            if done:
                desired = dict.fromkeys(self.caps, 0)
                self.demand_targets = dict(desired)
            # A downscale is not free capacity until the node acknowledges the
            # command AND the old processes have exited. Lost/stale nodes keep
            # their reservation until retirement/reboot is externally proven.
            for n in self.caps:
                snapshot = self.nodes.get(n, {})
                if snapshot.get('control_epoch') == self.epoch and snapshot.get('control_sequence', 0) >= self.revisions[n]:
                    self.charged[n] = self.targets[n]
                self.charged[n] = max(self.charged[n], snapshot.get('active', 0))
            free = max(0, self.maximum - sum(self.charged.values()))
            previous_targets = dict(self.targets)
            for node in self.caps:
                before = self.targets[node]
                value = desired.get(node, 0)
                if value > before:
                    add = min(value - before, free)
                    value, free = before + add, free - add
                if node in retired:
                    value = 0
                self.targets[node] = value
                self.charged[node] = max(self.charged[node], value)
                if value != before:
                    self.revisions[node] = self.sequence + 1
                    event(self.root / 'AUTOSCALE_DECISIONS.jsonl', node, value - before, reason,
                          dict(memory=self.nodes.get(node, {}), requested=value),
                          dict(targets=dict(self.targets), active=active, maximum=self.maximum))
            if self.targets != previous_targets:
                self.changed, self.last_counts = now, tuple(self.targets.items())
                self.active_anchor, self.stable_peak = dict(observed), dict(observed)
            self.publish(done)


def from_environment(driver):
    path = os.environ.get('OPENPI_AUTOSCALE_CONFIG')
    if not path:
        return None
    campaign = json.loads(Path(path).read_text())
    if not campaign.get('autoscale'):
        return None
    return DriverController(driver, campaign, Path(path).parent)


def status_entries(entries, directory):
    """Piggyback on the existing 30-second lease/status call, including N=0."""
    control = read(Path(directory) / 'AUTOSCALE_CONTROL.json')
    return [dict(e, autoscale_control=control) if e['label'] == 'workers' else e for e in entries]


def status_result(request, result):
    path = request.get('env', {}).get('OPENPI_AUTOSCALE_NODE')
    if path:
        result['autoscale_node'] = read(path, {})
    return result


def control_entry(entry):
    path = entry['request'].get('env', {}).get('OPENPI_AUTOSCALE_CONTROL')
    control = entry.get('autoscale_control')
    if path and control and Path(path).parent.is_dir():
        with Path(path + '.lock').open('a') as lock:
            import fcntl
            fcntl.flock(lock, fcntl.LOCK_EX)
            old = read(path, {})
            if control.get('time', 0) > old.get('time', 0):
                atomic(path, control)


def collect_status(directory, status_cache):
    atomic(Path(directory) / 'AUTOSCALE_NODES.json', {n: s['autoscale_node'] for (n, label), s in status_cache.items()
                                                     if label == 'workers' and s.get('autoscale_node')})
