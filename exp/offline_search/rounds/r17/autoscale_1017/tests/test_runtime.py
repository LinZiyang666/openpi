import os
from pathlib import Path
from types import SimpleNamespace
import pytest

from exp.offline_search.rounds.r17.autoscale_1017.driver import DriverController, control_entry, status_result
from exp.offline_search.rounds.r17.autoscale_1017.node import LocalGovernor, atomic, identity, process_tree, read
from exp.offline_search.rounds.r17.autoscale_1017.policy import GIB


class Clock:
    value = 0.
    def __call__(self):return self.value
    def wall(self):return 1700000000 + self.value
    def advance(self, seconds=2):self.value += seconds


def test_node_memory_growth_pause_and_recovery_no_refill(tmp_path, monkeypatch):
    clock = Clock()
    monkeypatch.setattr('exp.offline_search.rounds.r17.autoscale_1017.node.time.time', clock.wall)
    available = [70 * GIB]
    sizes = {}
    g = LocalGovernor(dict(run_name='lane', model='cosmos', domain='libero', node_workers=8,
                            worker_node='local', autoscale=dict(hold_s=6)), tmp_path / 'run',
                      ledger=tmp_path / 'ledger', clock=clock,
                      reader=lambda: (available[0], {'full': dict(avg10=0)}),
                      process_reader=lambda pid, ticks: sizes.get(pid))
    atomic(g.history_path, [dict(peak=6 * GIB, complete=True)])
    atomic(g.control_path, dict(campaign='lane', time=clock.wall(), targets=dict(local=8)))
    for i in range(3):
        with g.permit() as allowed:
            assert allowed
            sizes[i + 10] = (4 * GIB, 6 * GIB)
            g.register(dict(pid=i + 10, start_ticks='1', worker_id=str(i), born=i))
    available[0] = 19 * GIB
    with g.permit() as allowed:assert not allowed
    g.finished('2', normal=True)
    with g.permit() as allowed:assert not allowed
    available[0] = 60 * GIB
    clock.advance(2)
    with g.permit() as allowed:assert not allowed
    clock.advance(6)
    with g.permit() as allowed:assert allowed
    lines = [__import__('json').loads(s) for s in g.events.read_text().splitlines()]
    assert {r['reason'] for r in lines} >= {'spawn', 'generation_exit', 'memory_hold'}
    assert all({'time', 'node', 'delta', 'reason', 'signals', 'counts'} <= r.keys() for r in lines)


def test_one_unknown_canary_per_campaign_across_fleets(tmp_path, monkeypatch):
    clock = Clock()
    monkeypatch.setattr('exp.offline_search.rounds.r17.autoscale_1017.node.time.time', clock.wall)
    c = dict(run_name='lane', node_workers=4, worker_node='local', autoscale={})
    kwargs = dict(ledger=tmp_path / 'ledger', clock=clock,
                  reader=lambda: (100 * GIB, {'full': dict(avg10=0)}), process_reader=lambda *args: (GIB, GIB))
    a, b = (LocalGovernor(c, tmp_path / name, **kwargs) for name in ('a', 'b'))
    for g in (a, b):atomic(g.control_path, dict(campaign='lane', time=clock.wall(), targets=dict(local=4)))
    with a.permit() as allowed:
        assert allowed
        a.register(dict(pid=os.getpid(), start_ticks=identity(os.getpid()), worker_id='a'))
    with b.permit() as allowed:assert not allowed
    clock.advance(20)
    assert b.tick()['reason'] == 'memory_unknown'  # stalled live owner


def test_cgroup_and_pid_identity_sampling_is_cpu_only(tmp_path):
    proc = tmp_path / 'proc'
    for pid, children, rss, peak in ((10, '11', 20, 40), (11, '', 10, 30)):
        (proc / str(pid) / 'task' / str(pid)).mkdir(parents=True)
        (proc / str(pid) / 'stat').write_text(str(pid) + ' (name with spaces) S ' + ' '.join(['0'] * 18 + ['123']))
        (proc / str(pid) / 'status').write_text('VmRSS:\t%d kB\nVmHWM:\t%d kB\n' % (rss, peak))
        (proc / str(pid) / 'task' / str(pid) / 'children').write_text(children)
    assert process_tree(10, '123', proc) == (30 * 1024, 70 * 1024)
    assert process_tree(10, 'wrong', proc) is None


def controller(tmp_path, clock, mode='fixed', maximum=8):
    sched = SimpleNamespace(all_done=lambda: False, busy=lambda: dict(server=3))
    d = SimpleNamespace(scheduler=sched)
    c = dict(run_name='lane', workers=maximum, worker_nodes=dict(timan107=4, timan1=4), autoscale=dict(mode=mode))
    return DriverController(d, c, tmp_path, clock=clock, wall=clock.wall)


def node(clock, name, active=0, headroom=100, target=0, **kw):
    return dict(node=name, campaign='lane', time=clock.wall(), headroom=headroom * GIB,
                projected_peak=4 * GIB, active=active, target=target, workers=[], admit=True, **kw)


def test_control_ttl_zero_recovery_and_duplicate_accepted_fences(tmp_path):
    clock = Clock()
    d = controller(tmp_path, clock)
    clock.advance()
    d.ingest(dict(node=node(clock, 'timan107')))
    d.tick()
    assert d.targets['timan107'] == 4
    assert sum(d.targets.values()) <= 8
    d.result(dict(_autoscale_episode=dict(sim_render_s=1, server_wait_s=2, service_s=.2)), False)
    assert len(d.results) == 0
    d.result(dict(_autoscale_episode=dict(sim_render_s=1, server_wait_s=2, service_s=.2)), True)
    assert len(d.results) == 1
    # Zero workers can still carry a command through the existing status path.
    path = tmp_path / 'node_control.json'
    request = dict(env=dict(OPENPI_AUTOSCALE_CONTROL=str(path)))
    control_entry(dict(request=request, autoscale_control=d.control))
    assert read(path)['targets']['timan107'] == 4
    older = dict(d.control, time=d.control['time'] - 1, targets={})
    control_entry(dict(request=request, autoscale_control=older))
    assert read(path)['targets']['timan107'] == 4
    request['env']['OPENPI_AUTOSCALE_NODE'] = str(path)
    assert status_result(request, {})['autoscale_node'] == read(path)


def test_downscale_requires_ack_and_actual_exit_before_moving_quota(tmp_path):
    clock = Clock()
    d = controller(tmp_path, clock)
    clock.advance()
    d.ingest(dict(node=node(clock, 'timan107')))
    d.tick()
    assert d.targets == dict(timan107=4, timan1=0)
    # Simulate an older in-flight grant; no ACK means no reclaimed quota.
    d.charged.update(timan107=8, timan1=0)
    # Old grant8 may still be in transit: cannot treat active0 as free.
    clock.advance(100)
    d.ingest(dict(node=node(clock, 'timan1')))
    d.tick()
    assert d.targets == dict(timan107=0, timan1=0)
    clock.advance()
    d.tick()
    assert d.targets['timan1'] == 0
    # Acknowledged lower command, but8 old children still exist.
    d.ingest(dict(node=node(clock, 'timan107', active=8, headroom=0, target=0,
                            control_epoch=d.epoch, control_sequence=d.revisions['timan107'])))
    clock.advance()
    d.tick()
    assert d.targets['timan1'] == 0
    # All exits observed; destination can now use capacity.
    d.ingest(dict(node=node(clock, 'timan107', active=0, headroom=0, target=0,
                            control_epoch=d.epoch, control_sequence=d.revisions['timan107'])))
    clock.advance()
    d.tick()
    assert d.targets['timan1'] == 4


def test_done_publishes_even_within_sample_interval(tmp_path):
    clock = Clock()
    d = controller(tmp_path, clock)
    d.driver.scheduler.all_done = lambda: True
    d.tick()
    assert d.control['done']
    assert not any(d.targets.values())


def test_actual_driver_large_initial_quotas_and_five_window_backoff(tmp_path):
    clock = Clock()
    d = DriverController(SimpleNamespace(scheduler=SimpleNamespace(all_done=lambda: False, busy=lambda: dict(server=100))),
                         dict(run_name='lane', workers=250, worker_nodes=dict(timan1=150, timan107=100), autoscale=dict(mode='dynamic')),
                         tmp_path, clock=clock, wall=clock.wall)
    assert d.tuner.target == 250 and d.control['targets'] == dict(timan1=150, timan107=100)
    for _ in range(5):
        clock.advance(180)
        for name, n in d.targets.items():
            d.ingest(dict(node=node(clock, name, active=n, target=n, headroom=1000,
                calibrated=True, control_epoch=d.epoch, control_sequence=d.sequence)))
        for _ in range(258):
            d.result(dict(_autoscale_episode=dict(sim_render_s=8, server_wait_s=100, service_s=19.8, cohort='formal')), True)
        d.tick()
        assert d.targets['timan1'] <= 150 and d.targets['timan107'] <= 100
        assert sum(d.targets.values()) <= 250
    assert d.tuner.target == 44 and clock.value == 900
    import json
    windows = [r for r in map(json.loads, (tmp_path / 'AUTOSCALE_DECISIONS.jsonl').read_text().splitlines()) if 'rate' in r['signals']]
    assert len(windows) == 5 and all(r['signals']['samples'] == 258 for r in windows)
    assert all(r['reason'] == 'server_queue' for r in windows)


@pytest.mark.parametrize('gpu_ceiling',[None,30])
def test_periodic_recycling_does_not_prevent_stable_windows_or_upward_probes(tmp_path,gpu_ceiling):
    import json
    clock = Clock()
    d = DriverController(SimpleNamespace(scheduler=SimpleNamespace(all_done=lambda:False,busy=lambda:dict(server=44))),
        dict(run_name='lane',workers=100,worker_nodes=dict(timan107=100),autoscale=dict(mode='dynamic')),
        tmp_path,clock=clock,wall=clock.wall)
    d.tuner.target=44;d.targets=dict(timan107=44);d.charged=dict(d.targets)
    d.publish(False)
    credits=0.
    for step in range(1,721):
        clock.advance(2)
        n=min(d.targets['timan107'],gpu_ceiling or 100)
        d.ingest(dict(node=node(clock,'timan107',active=n-(step%3==0),target=n,headroom=1000,
            calibrated=True,control_epoch=d.epoch,control_sequence=d.sequence)))
        # Keep throughput proportional to occupied slots below the initial
        # useful capacity, so descending too far must be rejected.
        credits+=min(n,gpu_ceiling or 44)/(gpu_ceiling or 44)
        while credits>=1:
            d.result(dict(_autoscale_episode=dict(sim_render_s=20,server_wait_s=2,service_s=10,cohort='formal')),True)
            credits-=1
        d.tick()
    windows=[r for r in map(json.loads,(tmp_path/'AUTOSCALE_DECISIONS.jsonl').read_text().splitlines()) if 'rate' in r['signals']]
    assert sum(r['signals']['settled'] for r in windows)>=5
    assert any(r['reason']=='periodic_probe' for r in windows)
    assert d.tuner.target>=min(44,gpu_ceiling or 100)


def test_driver_does_not_double_reserve_live_workers_remaining_growth(tmp_path):
    clock = Clock()
    d = DriverController(SimpleNamespace(scheduler=SimpleNamespace(all_done=lambda: False)),
                         dict(run_name='lane', workers=100, worker_nodes=dict(timan107=100), autoscale=dict(mode='dynamic')),
                         tmp_path, clock=clock, wall=clock.wall)
    clock.advance(2)
    d.ingest(dict(node=dict(node='timan107', campaign='lane', time=clock.wall(), active=40,
        target=100, headroom=80 * GIB, projected_peak=3 * GIB, admit=True,
        workers=[dict(rss=GIB, peak=GIB, projected_peak=3 * GIB)] * 40)))
    d.tick()
    assert d.targets['timan107'] == 66  # 40 live + floor(80/3) new, not40 RSS/P slots.


def test_node_lost_before_launch_does_not_turn_desired_maximum_into_requirement(tmp_path):
    clock = Clock()
    d = DriverController(SimpleNamespace(scheduler=SimpleNamespace(all_done=lambda: False)),
                         dict(run_name='lane', workers=250, autoscale_maximum=250,
                              worker_nodes=dict(timan1=150, timan107=100), execution_worker_nodes=dict(timan1=150),
                              autoscale=dict(mode='dynamic')), tmp_path, clock=clock, wall=clock.wall)
    assert d.tuner.target == 250 and d.targets == dict(timan1=150)
