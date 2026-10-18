"""Deterministic CPU curves; these are assumptions, not measured trial results."""
import json
from .policy import Config, GIB, MemoryGuard, Tuner


def simulate():
    guard = MemoryGuard()
    memory, workers = [], []
    available = 220 * GIB
    for now in range(0, 401, 2):
        if now == 210:available -= 2 * GIB
        if now == 240:available += 12 * GIB
        decision = guard.evaluate(now, available, workers, 2 * GIB)
        if decision['admit'] and len(workers) < 150:
            workers.append(dict(worker_id=str(now), born=now, rss=2 * GIB, peak=2 * GIB, projected_peak=2 * GIB))
            available -= 2 * GIB
        memory.append(dict(second=now, active=len(workers), maximum=150, available_gib=available / GIB, **decision))
    tuner = Tuner(250)
    throughput = []
    for now in range(180, 14401, 180):
        active = tuner.target
        # Historical-shaped hypothesis:85 workers ~55/min,44 ~86/min.
        capacity = 86 / 60 if now < 7200 else 144 / 60
        optimum = capacity * 30
        rate = min(active / 30, capacity) / (1 + .01375 * max(0, active - optimum))
        wait = max(28, active / rate - 2)
        target, reason = tuner.update(now, active=active, rate=rate, samples=int(180 * rate),
                                     sim_s=2, wait_s=wait, service_s=28, queue_inflight=active)
        throughput.append(dict(second=now, active=active, target=target, accepted_per_min=rate * 60,
                               server_capacity_per_min=capacity * 60, reason=reason))
    convergence = {}
    for maximum in (150, 250):
        t = Tuner(maximum);trace = [maximum]
        for now in range(180, 1201, 180):
            n, _ = t.update(now, active=t.target, rate=86 / 60, samples=258,
                sim_s=8, wait_s=100, service_s=19.8, queue_inflight=t.target, settled=False)
            trace.append(n)
            if n <= 44:break
        convergence[maximum] = dict(seconds=now, counts=trace)
    return dict(synthetic=True, memory=memory, throughput=throughput, convergence=convergence)


if __name__ == '__main__':print(json.dumps(simulate(), indent=2))
