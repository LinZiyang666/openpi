"""R17 resident fleet adapter; reuses formal_workers' owned launch/cleanup path."""
from __future__ import annotations

import fcntl
import os
from pathlib import Path
import time
import threading

from .node import LocalGovernor, atomic, event

SPAWNS_PER_SAMPLE = 16


class Slots:
    """Reserve only active GPU slots. RAM is admitted independently per spawn."""
    def __init__(self, campaign, run):
        self.c, self.run = campaign, Path(run)
        self.assignments = {}
        self.c['worker_gpu_assignments'] = self.assignments
        self.record = dict(owner=dict(pid=os.getpid(), start_ticks=__import__(
            'exp.r17_integration.ops.formal_process', fromlist=['ticks']).ticks(os.getpid())),
            assignments=[], autoscale=True, worker_vram_mib=campaign.get('worker_vram_mib', 1024), released=False)
        self.save()
        self.observed, self.observed_at = [], float('-inf')
        self.stop = threading.Event()
        self.observation_lock = threading.Lock()
        self.sampler = threading.Thread(target=self.sample, daemon=True)
        self.sampler.start()

    def sample(self):
        from exp.r17_integration.ops import worker_gpu as g
        while not self.stop.is_set():
            try:
                sampled_at = time.monotonic()
                observed = g.inventory(self.c.get('worker_node'), self.c.get('worker_egl_devices'))
                with self.observation_lock:
                    self.observed, self.observed_at = observed, sampled_at
            except (OSError, ValueError):
                pass  # Stale inventory never admits a new GPU context.
            self.stop.wait(10)

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
            with self.observation_lock:
                if time.monotonic() - self.observed_at > 20:return False
                observed = [dict(r) for r in self.observed]
            cards = {r['device']: r for r in observed
                     if r.get('usable') and r['device'] not in g.PROTECTED.get(self.c.get('worker_node'), set())}
            pending, evidence = g.reservations(self.run, cards)
            governor=getattr(self,'governor',None)
            if governor is not None:
                from .fairness import pending_gpu
                covered={Path(s['run']).name for s in governor.live_states if s.get('schema',0)>=2
                         and Path(s['run']).parent==self.run.parent}
                for row in evidence:
                    if row['run'] in covered:pending[row['device']]-=row['pending_mib']
                shared=pending_gpu(governor.live_states,cards)
                pending={d:max(0,pending[d])+shared[d] for d in cards}
            candidates = []
            for device, card in cards.items():
                floor = max(512, (card['total_mib'] + 9) // 10) + self.c.get('worker_server_vram_headroom_mib', 0)
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
    slots.governor=governor
    governor.slots=slots
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
            # Disk and GPU checks remain in the existing launch envelope. GPU
            # inventory is refreshed by its independent 10s sampler. Admission
            # runs every 2s; every pending child reserves its projected growth.
            # Several admissions per pass: N32 recycling of few-second episodes
            # needs more than one replacement per 2 s sample. Every spawn is
            # still admitted on a fresh sample; growth keeps its node ramp.
            for _ in range(SPAWNS_PER_SAMPLE if not drain and not terminal_shutdown else 0):
                with governor.permit() as allowed:
                    if not (allowed and admission(run, minimum_bytes(campaign))['admitted']):break
                    index = next((i for i in range(campaign['node_workers']) if i not in live), None)
                    if index is None or not slots.acquire(index):break
                    generation = generations.get(index, -1) + 1
                    spawned = launch(index, generation)
                    if spawned is None:
                        slots.release(index)
                        break
                    generations[index] = generation
                    children.append(spawned)
                    live[index] = spawned
                    governor.register(spawned[1])
            time.sleep(governor.config.sample_s)
        report.update(status='PARTIAL', reason='interrupted' if stopped() else 'worker_deadline')
    finally:
        # Outer formal_workers cleanup still owns processes. Keep active GPU
        # reservations until that cleanup finishes, even on exceptions.
        governor.slots = slots
