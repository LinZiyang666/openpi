"""R17 resident fleet loop and GPU slots; reuses formal_workers' owned launch/cleanup path."""
from __future__ import annotations

from collections import deque
import fcntl
import os
from pathlib import Path
import statistics
import subprocess
import time
import threading

from .node import SAMPLE_S, atomic, card_reserve

INVENTORY_S = 10
UTILIZATION_S = 5
UTILIZATION_WINDOW_S = 60


class Slots:
    """Reserve only active GPU slots. RAM is admitted by the NodeExecutor (attribute `governor`)."""
    def __init__(self, campaign, run):
        self.c, self.run = campaign, Path(run)
        self.governor = None
        self.assignments = {}
        self.c['worker_gpu_assignments'] = self.assignments
        # worker_gpu.reservations reads `autoscale`: an idle dynamic fleet holds only its assignments.
        self.record = dict(owner=dict(pid=os.getpid(), start_ticks=__import__(
            'exp.r17_integration.ops.formal_process', fromlist=['ticks']).ticks(os.getpid())),
            assignments=[], autoscale=True, worker_vram_mib=campaign.get('worker_vram_mib', 1024), released=False)
        self.save()
        self.observed, self.observed_at = [], float('-inf')
        self.utilization = deque()
        self.stop = threading.Event()
        self.observation_lock = threading.Lock()
        self.sampler = threading.Thread(target=self.sample, daemon=True)
        self.sampler.start()

    def sample(self):
        from exp.r17_integration.ops import worker_gpu as g
        inventoried = float('-inf')
        while not self.stop.is_set():
            if time.monotonic() - inventoried >= INVENTORY_S:
                inventoried = time.monotonic()
                try:
                    observed = g.inventory(self.c.get('worker_node'), self.c.get('worker_egl_devices'))
                    with self.observation_lock:
                        self.observed, self.observed_at = observed, inventoried
                except (OSError, ValueError):
                    pass  # Stale inventory never admits a new GPU context.
            self.sample_utilization()
            self.stop.wait(UTILIZATION_S)

    def sample_utilization(self):
        """Only the cards of the inventory view: protected cards are never queried."""
        view = self.view()
        if not view or not view['cards']:
            return
        try:
            output = subprocess.run(['nvidia-smi', '-i', ','.join(c['uuid'] for c in view['cards']),
                                     '--query-gpu=uuid,utilization.gpu', '--format=csv,noheader,nounits'],
                                    capture_output=True, text=True, timeout=8, check=True).stdout
        except (OSError, subprocess.SubprocessError):
            return
        row = {}
        for line in output.splitlines():
            uuid, _, value = line.partition(',')
            try:
                row[uuid.strip()] = float(value)
            except ValueError:
                pass
        now = time.monotonic()
        with self.observation_lock:
            self.utilization.append((now, row))
            while now - self.utilization[0][0] > UTILIZATION_WINDOW_S:
                self.utilization.popleft()

    def gpu_util(self):
        """Percent, mean over the last 60 s of the busiest card."""
        with self.observation_lock:
            rows = [r for t, r in self.utilization if time.monotonic() - t <= UTILIZATION_WINDOW_S]
        cards = {uuid for r in rows for uuid in r}
        return max((statistics.fmean(r[u] for r in rows if u in r) for u in cards), default=None)

    def save(self):
        self.record['assignments'] = list(self.assignments.values())
        atomic(self.run / 'GPU_RESERVATION.json', self.record)

    def view(self):
        from exp.r17_integration.ops import worker_gpu as g
        with self.observation_lock:
            if time.monotonic()-self.observed_at>20:return None
            return dict(time=self.observed_at,cards=[dict(c) for c in self.observed
                if c.get('usable') and c['device'] not in g.PROTECTED.get(self.c.get('worker_node'),set())])

    def acquire(self, index):
        from exp.r17_integration.ops import worker_gpu as g
        with (self.run.parent / '.egl_admission.lock').open('a') as lock:
            fcntl.flock(lock, fcntl.LOCK_EX)
            view = self.view()
            if view is None:
                return False
            cards = {c['device']: c for c in view['cards']}
            # Other fleets under this run root: their reservations' growth still to come.
            pending, evidence = g.reservations(self.run, cards)
            for row in evidence:
                if row['run'] == self.run.name:
                    pending[row['device']] -= row['pending_mib']
            # This fleet: per worker, so a grown worker does not offset a fresh one.
            for entry in self.governor.entries.values():
                card = cards.get(entry.get('egl_device'))
                if card is not None:
                    pids = set(entry['pids'])
                    used = sum(p.get('used_mib') or 0 for p in card['processes'] if p['pid'] in pids)
                    pending[card['device']] += max(0, self.record['worker_vram_mib'] - used)
            candidates = []
            for device, card in cards.items():
                floor = card_reserve(card['total_mib']) + self.c.get('worker_server_vram_headroom_mib', 0)
                free = card['total_mib'] - card['used_mib'] - pending[device] - floor
                if free >= self.record['worker_vram_mib']:
                    candidates.append((free, -device, card))
            if not candidates:
                return False
            _, _, card = max(candidates, key=lambda r: r[:2])
            order = self.c.get('worker_egl_devices') if self.c.get('worker_node') == 'timan1' else None
            if order:  # timan1 (owner 2026-10-18): fill GPU2 first, then GPU1
                card = min(candidates, key=lambda r: order.index(r[2]['device']) if r[2]['device'] in order else len(order))[2]
            self.assignments[index] = dict(index=index, device=card['device'], uuid=card['uuid'],
                                           worker_vram_mib=self.record['worker_vram_mib'])
            self.save()
            return True

    def release(self, index):
        self.assignments.pop(index, None)
        self.save()

    def close(self):
        self.stop.set()
        self.sampler.join(timeout=1)
        self.record['released'] = True
        self.save()


def run_fleet(campaign, run, report, children, logs, launch, stopped, draining, governor):
    from exp.r17_integration.ops.worker_lifetime import planned_recycle
    from exp.r17_integration.ops.worker_disk import admission, minimum_bytes
    slots = Slots(campaign, run)
    slots.governor = governor
    governor.slots = slots
    live, generations, failures = {}, {}, {}
    terminal_shutdown = False
    started = time.monotonic()
    last_receipt = float('-inf')
    report.update(status='RUNNING', memory_admission=dict(dynamic=True, policy='checked_per_spawn'))
    try:
        while not stopped() and time.monotonic() - started < campaign['seconds']:
            snapshot = governor.tick()
            drain = draining()
            for index, (child, entry) in list(live.items()):
                code = child.poll()
                if code is None:
                    continue
                child.wait(timeout=5)
                entry['exit_code'] = code
                planned = planned_recycle(entry, code)
                retired = entry['worker_id'] in governor.retired
                # GPU2 relay has a distinct, authenticated receipt format.
                if not planned and entry.get('node') == 'timan1':
                    from exp.r17_integration.ops.timan1_workers import planned_recycle as wrapped
                    planned = wrapped(entry, code)
                with governor.lock():
                    governor.finished(entry['worker_id'], normal=planned)
                slots.release(index)
                logs[index].close()
                del live[index]
                if code == 0 and not retired and not snapshot.get('done') and not drain:
                    # Normal shutdown is a driver-owned terminal signal. Keep
                    # supervisor alive until the driver publishes all_done.
                    entry['normal_shutdown'] = True
                    terminal_shutdown = True
                    continue
                if planned:
                    report['worker_recycles'].append(dict(index=index, generation=entry['generation'], pid=child.pid))
                elif code and not retired:
                    failures[index] = failures.get(index, 0) + 1
                    report['worker_failures'].append(dict(index=index, pid=child.pid, exit_code=code))
                    if failures[index] > campaign.get('worker_restart_limit', 2):
                        report.update(status='PARTIAL', reason='worker_restart_budget_exhausted')
                        return
            report['autoscale'] = snapshot
            report['workers_active'] = len(live)
            if time.monotonic() - last_receipt >= 30:
                atomic(Path(run) / 'receipt.json', report)
                last_receipt = time.monotonic()
            if (snapshot.get('done') or drain) and not live:
                report.update(status='PASS', reason='autoscale_drained')
                return
            # The executor's per-tick budget (1 + planned recycles) and RAM check
            # bound this loop. Disk and GPU checks remain in the existing launch envelope.
            while not drain and not terminal_shutdown:
                with governor.permit() as allowed:
                    index = next((i for i in range(campaign['node_workers']) if i not in live), None)
                    if not (allowed and index is not None and admission(run, minimum_bytes(campaign))['admitted']
                            and slots.acquire(index)):
                        break
                    generation = generations.get(index, -1) + 1
                    spawned = launch(index, generation)
                    if spawned is None:
                        slots.release(index)
                        break
                    generations[index] = generation
                    children.append(spawned)
                    live[index] = spawned
                    governor.register(spawned[1])
            time.sleep(SAMPLE_S)
        report.update(status='PARTIAL', reason='interrupted' if stopped() else 'worker_deadline')
    finally:
        # Outer formal_workers cleanup still owns processes. Keep active GPU
        # reservations until that cleanup finishes, even on exceptions.
        governor.slots = slots
