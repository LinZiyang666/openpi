import math

from exp.offline_search.rounds.r17.autoscale_1017.policy import Config, GIB, MemoryGuard, Peaks, Tuner, place


def worker(name, born=0, rss=4, peak=6, **kw):
    return dict(worker_id=name, born=born, rss=rss * GIB, peak=peak * GIB, projected_peak=6 * GIB, **kw)


def test_p90_recent_same_campaign_generations():
    peaks = Peaks(1.15)
    for value in range(1, 11):peaks.add('cosmos', value * GIB)
    peaks.add('other-model', 200 * GIB)
    assert peaks.project('cosmos') == math.ceil(9 * GIB * 1.15)
    assert peaks.project('unknown') == 0
    for _ in range(128):peaks.add('cosmos', 3 * GIB)
    assert peaks.project('cosmos') == math.ceil(3 * GIB * 1.15)


def test_remaining_growth_is_not_double_counted_and_recycled_slots_recheck():
    guard = MemoryGuard()
    workers = [worker('old', rss=5), worker('new', rss=1)]
    # Available34 - floor20 - remaining(1+5) =8; a6GiB new generation fits.
    assert guard.evaluate(0, 34 * GIB, workers, 6 * GIB)['admit']
    assert not guard.evaluate(2, 30 * GIB, workers, 6 * GIB)['admit']
    assert not guard.evaluate(4, 19 * GIB, workers, 6 * GIB)['admit']
    # Recycle has no special exemption.
    assert not guard.evaluate(6, 19 * GIB, workers[:1], 6 * GIB)['admit']
    assert not guard.evaluate(8, 26 * GIB, [], 6 * GIB)['admit']
    assert not guard.evaluate(67, 26 * GIB, [], 6 * GIB)['admit']
    assert guard.evaluate(68, 26 * GIB, [], 6 * GIB)['admit']


def test_critical_retires_newest_at_boundary_with_pending_drains_credited():
    guard = MemoryGuard()
    workers = [worker('older', 1, rss=8, peak=8), worker('newer', 2, rss=8, peak=8), worker('newest', 3, rss=8, peak=8)]
    d = guard.evaluate(0, 9 * GIB, workers, 8 * GIB)
    assert d['retire'] == ['newest', 'newer']
    workers[1]['draining'] = workers[2]['draining'] = True
    assert guard.evaluate(2, 9 * GIB, workers, 8 * GIB)['retire'] == []
    workers[0]['owned'] = False
    assert guard.evaluate(4, 2 * GIB, workers, 8 * GIB)['retire'] == []


def test_psi_unknown_and_hysteresis_reset():
    g = MemoryGuard()
    assert not g.evaluate(0, 80 * GIB, [], 4 * GIB, psi_full=12)['admit']
    assert not g.evaluate(10, 80 * GIB, [], 4 * GIB)['admit']
    assert not g.evaluate(50, 23 * GIB, [], 4 * GIB)['admit']
    assert not g.evaluate(70, 80 * GIB, [], 4 * GIB)['admit']
    assert g.evaluate(130, 80 * GIB, [], 4 * GIB)['admit']
    assert g.evaluate(132, 80 * GIB, [], 4 * GIB, valid=False)['reason'] == 'memory_unknown'


def test_memory_curve_ramp_floor_critical_recovery():
    g = MemoryGuard(Config(hold_s=6))
    curve = [(0, 60), (2, 45), (4, 28), (6, 19), (8, 9), (10, 30), (12, 23), (14, 30), (18, 30), (20, 30)]
    records = [g.evaluate(t, free * GIB, [worker('old', 1), worker('new', 2)], 6 * GIB) for t, free in curve]
    assert [r['admit'] for r in records] == [True, True, False, False, False, False, False, False, False, True]
    assert records[4]['retire'][0] == 'new'


def test_server_plateau_and_later_improvement():
    tuner = Tuner(24, Config(window_s=60, probe_s=180))
    trace = []
    for now in range(60, 3601, 60):
        active = tuner.target
        capacity = .8 if now < 1800 else 1.6
        rate = min(active / 10, capacity)
        wait = max(8, active / rate - 2)
        target, reason = tuner.update(now, active=active, rate=rate,
                                     samples=30, sim_s=2, wait_s=wait, service_s=7, queue_inflight=active)
        trace.append((now, target, reason))
    assert max(n for t, n, _ in trace if 1000 < t < 1800) <= 9
    assert max(n for t, n, _ in trace if t > 2200) >= 16
    assert all(1 <= n <= 24 for _, n, _ in trace)
    assert any(reason == 'marginal_plateau' for _, _, reason in trace)
    assert any(reason == 'periodic_probe' for _, _, reason in trace)


def test_queue_shrinks_toward_littles_law_but_still_probes():
    tuner = Tuner(85, Config(window_s=60, probe_s=180))
    tuner.target = 44
    history = []
    for now in range(60, 901, 60):
        n, reason = tuner.update(now, active=tuner.target, rate=4, samples=240,
                                sim_s=2, wait_s=8, service_s=.5, queue_inflight=40)
        history.append((n, reason))
    assert min(n for n, _ in history) == 11  # ceil(rate * unqueued_cycle *1.1)
    assert any(r == 'periodic_probe' for _, r in history)
    assert history[0][0] < 44


def test_unknown_service_never_invents_unqueued_cycle_and_fixed_keeps_max():
    tuner = Tuner(85)
    tuner.target = 44
    n, reason = tuner.update(180, active=44, rate=3, samples=20, sim_s=1, wait_s=30,
                            service_s=None, queue_inflight=44)
    assert reason != 'server_queue'
    fixed = Tuner(12, Config(mode='fixed'))
    assert fixed.update(1000, active=3, rate=.1, samples=100, sim_s=1, wait_s=20, service_s=.2) == (12, 'hold')


def test_place_headroom_no_local_two_no_dead_node():
    rows = {'timan107': dict(cap=64, headroom=100), 'timan1': dict(cap=64, headroom=200),
            'local': dict(cap=2, headroom=2), 'timan108': dict(cap=1000, headroom=100000)}
    result = place(44, rows)
    assert sum(result.values()) == 44
    assert result['local'] == result['timan108'] == 0
    assert result['timan1'] > result['timan107']
    assert place(2, rows)['timan1'] == 2
    assert place(3, {'local': dict(cap=2, headroom=2)})['local'] == 2


def test_large_initial_target_is_maximum_even_with_sparse_or_unsettled_evidence():
    for maximum in (150, 250, 350):
        tuner = Tuner(maximum)
        assert tuner.target == maximum
        assert tuner.update(600, active=1, rate=1 / 180, samples=1,
                            sim_s=30, wait_s=90, service_s=None)[0] == maximum
        assert tuner.update(780, active=2, rate=0, samples=0,
                            sim_s=0, wait_s=0, service_s=None)[0] == maximum


def test_small_headroom_is_not_promoted_to_an_artificial_four_worker_slice():
    rows = {'timan107': dict(cap=12, headroom=100), 'timan1': dict(cap=12, headroom=200),
            'local': dict(cap=12, headroom=1)}
    assert place(12, rows) == dict(timan107=4, timan1=8, local=0)


def test_convergence_from_owner_maxima_under_twenty_minutes_while_draining():
    for maximum, minutes in ((150, 12), (250, 15)):
        t = Tuner(maximum)
        trace = []
        for now in range(180, 1201, 180):
            n = t.target
            target, reason = t.update(now, active=n, rate=86 / 60, samples=258,
                sim_s=8, wait_s=100, service_s=19.8, queue_inflight=n, settled=False)
            trace.append(target)
            if target <= 44:break
        assert trace[0] == int(maximum * .7)
        assert target == 44 and now == minutes * 60
        assert reason == 'server_queue'


def test_220gib_node_150_max_ramps_to_memory_envelope_and_recovers():
    guard = MemoryGuard(Config(hold_s=60))
    peak, available, workers = 2 * GIB, 220 * GIB, []
    for i in range(150):
        decision = guard.evaluate(i * 2, available, workers, peak)
        if not decision['admit']:break
        workers.append(dict(worker_id=str(i), born=i, rss=peak, peak=peak, projected_peak=peak))
        available -= peak
    assert len(workers) == 100 and available == 20 * GIB
    assert decision['reason'] == 'memory_growth' and not decision['admit']
    # A separate 2GiB resident load starts; no recycle refill below the floor.
    assert not guard.evaluate(202, 18 * GIB, workers[:-1], peak)['admit']
    assert not guard.evaluate(204, 30 * GIB, workers[:-1], peak)['admit']
    assert guard.evaluate(264, 30 * GIB, workers[:-1], peak)['admit']


def test_owner_scale_synthetic_plateau_then_faster_server_recovers():
    from exp.offline_search.rounds.r17.autoscale_1017.simulate import simulate
    data = simulate()
    assert data['throughput'][0]['active'] == 250
    plateau = [r for r in data['throughput'] if 3600 <= r['second'] < 7200]
    improved = [r for r in data['throughput'] if r['second'] >= 10800]
    assert all(41 <= r['active'] <= 45 and r['accepted_per_min'] >= .95 * 86 for r in plateau)
    assert all(r['active'] >= 69 and r['accepted_per_min'] >= .95 * 144 for r in improved)
    assert any(r['reason'] == 'periodic_probe' for r in plateau)
    assert any(not r['admit'] and r['reason'] == 'memory_hold' for r in data['memory'])
    assert any(r['admit'] and r['second'] >= 300 for r in data['memory'])


def test_network_latency_proxy_cannot_shrink_throughput_to_one_worker():
    tuner = Tuner(100)
    rows = []
    for now in range(180, 14401, 180):
        n = tuner.target
        capacity = 1. if now < 5400 else 2.
        rate = min(n / 10, capacity)
        target, reason = tuner.update(now, active=n, rate=rate, samples=int(rate * 180),
            sim_s=1, wait_s=n / rate - 1, service_s=.1, queue_inflight=n)
        rows.append((now, n, rate, target, reason))
    assert any(r[4]=='queue_rate_rollback' for r in rows)
    assert all(r[2]>=.95 for r in rows if 3600<=r[0]<5400)
    assert all(r[2]>=1.9 for r in rows if r[0]>=10800)
    assert min(r[1] for r in rows)>1 and max(r[3] for r in rows)<=100
    assert any(r[4]=='periodic_probe' for r in rows)


def test_rate_only_descent_finds_cpu_plateau_without_server_timing():
    tuner=Tuner(100)
    rows=[]
    for now in range(180,14401,180):
        n=tuner.target
        capacity=1. if now<5400 else 2.
        rate=min(n/10,capacity)/(1+.002*max(0,n-capacity*10))
        target,reason=tuner.update(now,active=n,rate=rate,samples=int(rate*180),
            sim_s=n/rate-1,wait_s=1,service_s=None,queue_inflight=n)
        rows.append((now,n,rate,target,reason))
    assert rows[0][4]=='throughput_probe_down'
    assert all(r[2]>=.95 for r in rows if 3600<=r[0]<5400)
    assert all(r[2]>=1.9 for r in rows if r[0]>=10800)
    assert any(r[4]=='throughput_probe_rollback' for r in rows)
