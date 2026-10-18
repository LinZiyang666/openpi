"""Pure policy: bytes, monotonic seconds, accepted terminal episodes/second."""
from __future__ import annotations

from collections import deque
from dataclasses import dataclass
import math

GIB = 1024 ** 3
NODES = ('timan107', 'timan1', 'local')


@dataclass(frozen=True)
class Config:
    floor: int = 20 * GIB
    critical: int = 10 * GIB
    hysteresis: int = 4 * GIB
    hold_s: float = 60
    sample_s: float = 2
    stale_s: float = 90
    peak_margin: float = 1.15
    window_s: float = 180
    probe_s: float = 600
    marginal: float = .05
    wait_fraction: float = .65
    decrease: float = .7
    min_samples: int = 8
    min_slice: int = 4
    mode: str = 'dynamic'
    spawn_interval_s: float = 0
    external_peak_floor: int = 0
    external_peak_floor_metaworld: int = 0

    def __post_init__(self):
        if not 0 < self.critical < self.floor or self.hysteresis < 0:
            raise ValueError('require 0 < critical < floor and nonnegative hysteresis')
        if min(self.hold_s, self.sample_s, self.stale_s, self.window_s, self.probe_s) <= 0:
            raise ValueError('positive timing required')
        if self.peak_margin < 1 or not 0 < self.decrease < 1:
            raise ValueError('invalid projection/decrease')
        if self.spawn_interval_s < 0 or min(self.external_peak_floor, self.external_peak_floor_metaworld) < 0:
            raise ValueError('nonnegative spawn interval and external floors required')
        if not 0 <= self.marginal < 1 or not 0 < self.wait_fraction < 1:
            raise ValueError('invalid throughput thresholds')
        if self.mode not in ('dynamic', 'fixed') or self.min_slice < 1 or self.min_samples < 1:
            raise ValueError('invalid mode/sample/slice')


def p90(values):
    ordered = sorted(v for v in values if v > 0 and math.isfinite(v))
    return ordered[max(0, math.ceil(.9 * len(ordered)) - 1)] if ordered else 0


class Peaks:
    """Bounded, same-campaign completed generations; never mix model/benchmark keys."""
    def __init__(self, margin=1.15):
        self.history = {}
        self.margin = margin

    def add(self, campaign, peak):
        if peak > 0:
            self.history.setdefault(campaign, deque(maxlen=128)).append(peak)

    def project(self, campaign):
        return math.ceil(p90(self.history.get(campaign, ())) * self.margin)


def remaining_growth(workers):
    # Both RSS and peak are process-tree aggregates measured by their owner.
    return sum(max(0, max(w['projected_peak'], w['peak']) - w['rss']) for w in workers)


class MemoryGuard:
    def __init__(self, config=Config()):
        self.c = config
        self.blocked = False
        self.healthy_since = None

    def evaluate(self, now, available, workers, new_peak, *, valid=True, psi_full=0, external_growth=0):
        if not valid:
            self.blocked, self.healthy_since = True, None
            return dict(admit=False, retire=[], kill=[], reason='memory_unknown', headroom=0)
        growth = remaining_growth(workers)
        headroom = available - self.c.floor - growth
        # PSI is a stop-admission signal, never evidence that RAM became free.
        pressure = psi_full >= 10
        if available < self.c.floor or pressure:
            self.blocked, self.healthy_since = True, None
        elif self.blocked:
            if available >= self.c.floor + self.c.hysteresis and not pressure:
                if self.healthy_since is None:
                    self.healthy_since = now
                if now - self.healthy_since >= self.c.hold_s:
                    self.blocked = False
            else:
                self.healthy_since = None
        retire = []
        if available < self.c.critical:
            # Already draining RSS will be released; do not retire it twice.
            released = sum(w['rss'] for w in workers if w.get('draining'))
            need = self.c.floor + self.c.hysteresis - available
            for w in sorted(workers, key=lambda w: w['born'], reverse=True):
                if released >= need:
                    break
                if not w.get('draining') and w.get('owned', True):
                    retire.append(w['worker_id'])
                    released += w['rss']
        kill = []
        projected = available - external_growth
        if projected < self.c.critical:
            # Non-ledger workers are charged at their projected peak as soon as
            # their processes exist, before their RSS grows; an episode boundary
            # can be a minute away. Stop own newest workers now. Their episodes
            # are re-dispatched by the driver's disconnect fence.
            released = sum(w['rss'] for w in workers if w.get('killed'))
            need = self.c.critical + self.c.hysteresis - projected
            for w in sorted(workers, key=lambda w: w['born'], reverse=True):
                if released >= need:
                    break
                if w.get('owned', True) and not w.get('killed'):
                    kill.append(w['worker_id'])
                    released += w['rss']
        return dict(admit=not self.blocked and new_peak > 0 and headroom >= new_peak,
                    retire=retire, kill=kill, reason=('shed_kill' if kill else 'critical' if available < self.c.critical else
                    'memory_hold' if self.blocked else 'memory_growth' if headroom < new_peak else 'memory_ok'),
                    headroom=headroom, remaining_growth=growth, projected_peak=new_peak,
                    available=available, psi_full=psi_full, floor=self.c.floor, critical=self.c.critical,
                    projected_available=projected)


class Tuner:
    """One step per complete, comparable, settled accepted-rate window."""
    def __init__(self, maximum, config=Config()):
        if maximum < 1:
            raise ValueError('positive configured maximum required')
        self.maximum, self.c = maximum, config
        self.target = maximum
        self.previous = None
        self.last_probe = float('-inf')
        self.last_change = float('-inf')
        self.plateau = False
        self.queue_previous = None
        self.queue_floor = 0
        self.queue_floor_rate = 0.
        self.feedback = None
        self.exploring_down = False

    def update(self, now, *, active, rate, samples, sim_s, wait_s, service_s,
               queue_inflight=0, comparable=True, settled=True):
        before = self.target
        reason = 'hold'
        self.feedback = None
        if self.c.mode == 'fixed':
            return before, reason
        if not comparable:
            self.previous = self.queue_previous = None
            self.queue_floor = 0
            self.exploring_down = False
        wait_ratio = wait_s / max(sim_s + wait_s, 1e-9)
        queueing = (service_s is not None and wait_ratio >= self.c.wait_fraction
                    and wait_s > 1.5 * service_s and queue_inflight > 0)
        if (not comparable or samples < self.c.min_samples or rate <= 0
                or now - self.last_change < self.c.window_s):
            return before, 'insufficient_or_unsettled_window'
        # Backpressure is actionable while spawning/draining. Requiring a whole
        # settled window here doubles each shrink interval during a real ramp.
        useful = max(1, math.ceil(rate * (sim_s + (service_s or 0)) * 1.1))
        # Episode in-flight is only a queue proxy: network latency can dominate
        # even below server saturation. Reject a shrink that loses accepted
        # throughput instead of repeatedly shrinking toward a biased estimate.
        previous_queue = self.queue_previous
        rollback = bool(previous_queue and active < previous_queue[0]
                        and rate < previous_queue[1] * (1 - self.c.marginal))
        if previous_queue and active < previous_queue[0]:
            self.queue_previous = None  # this decrease has now been measured
        if (settled and self.queue_floor and active >= self.queue_floor
                and before <= self.queue_floor and rate < self.queue_floor_rate * .8):
            self.queue_floor = 0  # A slower service/workload may have a new optimum.
        proposed = max(useful, math.floor(min(active, before) * self.c.decrease),
                       min(self.queue_floor, active, before))
        probe_result = self.previous and (active > self.previous[0] or (settled and before > self.previous[0]))
        if rollback:
            old_n, old_rate = previous_queue
            reason = 'throughput_probe_rollback' if self.exploring_down else 'queue_rate_rollback'
            self.target = min(self.maximum, old_n)
            self.queue_floor, self.queue_floor_rate = self.target, old_rate
            self.feedback = dict(previous_active=old_n, previous_rate=old_rate,
                                 reduced_active=active, reduced_rate=rate)
            self.previous = self.queue_previous = None
            self.exploring_down = False
            self.plateau, self.last_probe = True, now
        elif probe_result:
            if not settled:
                return before, 'insufficient_or_unsettled_window'
            old_n, old_rate = self.previous
            marginal = (rate - old_rate) / old_rate if old_rate > 0 else 1
            self.queue_previous = None
            self.exploring_down = False
            if marginal < self.c.marginal:
                self.target = max(1, old_n)
                self.queue_floor, self.queue_floor_rate = self.target, old_rate
                self.previous = None
                self.plateau, self.last_probe = True, now
                reason = 'marginal_plateau'
            else:
                self.previous = (active, rate)
                self.target = min(self.maximum, active + max(1, active // 10))
                reason = 'measured_gain'
        elif queueing and proposed < min(active, before):
            self.queue_previous = (active, rate)
            self.target = proposed
            self.last_probe = now
            self.plateau = True
            self.previous = None
            self.exploring_down = False
            reason = 'server_queue'
        elif not settled:
            return before, 'insufficient_or_unsettled_window'
        elif active > 1 and (self.exploring_down or (not self.plateau and self.previous is None)):
            # Starting at the maximum leaves no upward gradient to measure.
            # A rate-only downward experiment also works when timing is absent
            # or the bottleneck is simulation/CPU rather than the server queue.
            self.queue_previous = (active, rate)
            self.exploring_down = True
            self.target = max(1, math.floor(min(active, before) * self.c.decrease))
            self.last_probe = now
            reason = 'throughput_probe_down'
        elif self.plateau and now - self.last_probe >= self.c.probe_s:
            self.previous = (active, rate)
            self.last_probe = now
            self.target = min(self.maximum, active + max(1, active // 10))
            reason = 'periodic_probe'
        elif now - self.last_probe >= self.c.probe_s:
            # Probing is required even when waiting dominates: the bottleneck
            # may have improved since the previous measurement.
            self.previous = (active, rate)
            self.last_probe = now
            self.target = min(self.maximum, active + max(1, active // 10))
            reason = 'periodic_probe' if self.plateau else 'initial_probe'
        elif not self.plateau and wait_ratio < self.c.wait_fraction:
            self.previous = (active, rate)
            self.target = min(self.maximum, active + max(1, active // 10))
            reason = 'additive_increase'
        self.target = min(self.maximum, max(1, self.target))
        if before != self.target:
            self.last_change = now
        return self.target, reason


def place(target, rows, *, min_slice=4):
    """Headroom-weighted capped allocation; consolidate tiny residual nodes.

    A tiny total lane may live on one node. Multi-node lanes never get a new
    1..min_slice-1 slice. Caps are prepared deployment ceilings, not live counts.
    """
    eligible = {n: r for n, r in rows.items() if n in NODES and r.get('valid', True)
                and r['cap'] > 0 and r['headroom'] > 0}
    result = {n: 0 for n in rows}
    left = min(target, sum(r['cap'] for r in eligible.values()))
    # Integer weighted water filling, then consolidate sub-minimum slices.
    while left:
        candidates = [n for n in eligible if result[n] < eligible[n]['cap']]
        if not candidates:break
        n = max(candidates, key=lambda n: (eligible[n]['headroom'] / (result[n] + 1), n))
        result[n] += 1
        left -= 1
    live = [n for n in eligible if result[n]]
    if len(live) <= 1:return result
    removed = 0
    for n in live:
        if result[n] < min_slice:
            removed += result[n]
            result[n] = 0
    if not any(result.values()) and eligible:
        n = max(eligible, key=lambda n: (eligible[n]['headroom'], n))
        result[n] = min(removed, eligible[n]['cap'])
        removed -= result[n]
    while removed:
        candidates = [n for n in eligible if result[n] >= min_slice and result[n] < eligible[n]['cap']]
        if candidates:
            n = max(candidates, key=lambda n: (eligible[n]['headroom'] / (result[n] + 1), n))
            result[n] += 1
            removed -= 1
        else:
            candidates = [n for n in eligible if not result[n] and eligible[n]['cap'] >= min_slice]
            if not candidates or removed < min_slice:break
            n = max(candidates, key=lambda n: (eligible[n]['headroom'], n))
            result[n] = min(removed, eligible[n]['cap'])
            removed -= result[n]
    return result
