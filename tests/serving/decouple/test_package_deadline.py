"""CPU proofs: clocks, admission, owned-group teardown; no actual child launches."""
from datetime import datetime, timezone
import json
import signal
import time

import pytest

from exp.decouple.gpu_package import check_ports, estimate_arm_seconds, resolve_deadline, run_package
from exp.decouple.package_config import REPO, make_arm
from exp.decouple.package_deadline import (DEFAULT_DEADLINE, DEADLINE_ENV, DeadlineBudget,
    PackageDeadlineReached, deadline_watchdog, parse_deadline, stop_owned_groups)


@pytest.fixture
def config():
    return json.loads((REPO / 'exp/decouple/benchmark.local.json').read_text())


def state(config):
    return dict(package_id='cpu_deadline', config_sha256=config['config_sha256'], config_path='sealed.json',
                gates={}, completed={}, selected={})


def result(spec, rate=65):
    return dict(name=spec['name'], logical_arm=spec['logical_arm'], model=spec['model'],
                fronts=spec['fronts'], model_processes=6 if spec['model_processes']=='auto' else spec['model_processes'],
                episodes_completed=300, episodes_per_min=rate, elapsed_s=300,
                gpu_util_mean_pct=50, system_cpu_mean_pct=10, cpu_proc_mean_pct=300,
                gpu_memory_peak_mb=40000, latency_p50_ms=10, latency_p99_ms=100,
                manifest_sha256=spec['manifest_sha256'], within_budget=True)


def gate(*args):
    return dict(passed=True)


def test_default_deadline_is_1515_cdt_and_override_precedence(config):
    expected = datetime(2026, 10, 8, 20, 15, tzinfo=timezone.utc).timestamp()
    assert parse_deadline(DEFAULT_DEADLINE) == expected
    assert parse_deadline('2026-10-08T15:15:00') == expected
    assert parse_deadline('2026-10-08T20:15:00Z') == expected
    assert parse_deadline('2026-10-08T15:15:00-05:00') == expected
    environment = {DEADLINE_ENV:'2026-10-08T14:15:00-05:00'}
    assert resolve_deadline(config, environment=environment) == expected - 3600
    assert resolve_deadline(config, '2026-10-08T16:15:00-05:00', environment) == expected + 3600
    with pytest.raises(ValueError):
        resolve_deadline(config, environment={DEADLINE_ENV:''})


@pytest.mark.parametrize('epoch,expected', [(20000,7080), (1000,880)])
def test_earliest_absolute_or_two_hour_cutoff(epoch, expected):
    budget = DeadlineBudget(epoch, 7200, 120, clock=lambda:0, wall_clock=lambda:0)
    assert budget.remaining() == expected
    assert budget.fits(expected) and not budget.fits(expected+1)


def test_clock_adjustments_never_extend_and_forward_adjustment_shortens():
    wall = [0]
    budget = DeadlineBudget(1000, 7200, 120, clock=lambda:0, wall_clock=lambda:wall[0])
    wall[0] = -100
    assert budget.remaining() == 880
    wall[0] = 500
    assert budget.remaining() == 380


@pytest.mark.parametrize('port', [23180,23198,23199,23178])
def test_unregistered_public_port_rejected(config, port):
    import hashlib
    from exp.decouple.package_config import canonical, validate
    config['public_port'] = port
    config.pop('config_sha256')
    config['config_sha256'] = hashlib.sha256(canonical(config)).hexdigest()
    with pytest.raises(AssertionError):
        validate(config, check_inputs=False)


def test_occupied_port_preflight_preserved_without_real_socket(config):
    calls = []
    class Probe:
        def __enter__(self):
            return self
        def __exit__(self, *args):
            pass
        def bind(self, address):
            calls.append(address)
            if address[1] == 23179:
                raise OSError('owned by another run')
    with pytest.raises(RuntimeError, match='23179 occupied; no listener was touched'):
        check_ports(config, Probe)
    assert calls == [('0.0.0.0',23100),('0.0.0.0',23179)]


def test_late_start_skips_slow_control_without_root_and_keeps_comparison(config, tmp_path, capsys):
    now, calls = [0.], []
    current = state(config)
    def arm(spec, *args):
        calls.append(spec['logical_arm'])
        assert spec['package_deadline'] == 1180
        assert spec['package_wall_shutdown_deadline'] == 1300
        now[0] += 500
        return result(spec)
    code = run_package(config, current, tmp_path, gate_runner=gate, arm_runner=arm,
                       clock=lambda:now[0], wall_clock=lambda:now[0], deadline_epoch=1300, watchdog=False)
    assert code == 2 and calls == ['current_max','split_8']
    assert len(current['completed']) == 2
    assert 'groot/initial/current_1' not in current.get('attempts', {})
    assert current['selected']['groot']['current'] == 'current_max'
    output = capsys.readouterr().out
    assert 'PKG_ARM_SKIPPED_DEADLINE groot_initial_current_1' in output
    assert 'PKG_DEADLINE_STOP' in output and 'PKG_ARM_FAIL' not in output
    summary = (tmp_path/'SUMMARY.md').read_text()
    assert 'groot/initial/current_max' in summary and '1/3 pairs' in summary
    assert 'PKG_DEADLINE_STOP' in summary


def test_repeat_pair_admission_prevents_starting_only_one_half(config, tmp_path, capsys):
    current = state(config)
    for model in ('groot','pi05'):
        for kind in config['initial_arms']:
            current['completed'][f'{model}/initial/{kind}'] = result(make_arm(config, model, kind, 'fixture'))
        current['selected'][model] = dict(current='current_max',split='split_8',current_copies=6)
    def forbidden(*args):
        pytest.fail('started a repeat without budget for its full pair')
    required = estimate_arm_seconds(config, current['completed'], 'groot','current_max')
    assert required < 800 < 2*required
    assert run_package(config, current, tmp_path, gate_runner=gate, arm_runner=forbidden,
                       clock=lambda:0, wall_clock=lambda:0, deadline_epoch=920, watchdog=False) == 2
    assert len(current['skipped_deadline']) == 8
    assert 'PKG_ARM_SKIPPED_DEADLINE groot_repeat1_split' in capsys.readouterr().out


@pytest.mark.parametrize('phase', ['gate','arm'])
def test_cutoff_interrupt_keeps_summary_and_uses_deadline_marker(config, tmp_path, capsys, phase):
    current = state(config)
    def cutoff(*args):
        raise PackageDeadlineReached('fake cutoff')
    assert run_package(config, current, tmp_path, gate_runner=cutoff if phase=='gate' else gate,
                       arm_runner=cutoff, clock=lambda:0, wall_clock=lambda:0,
                       deadline_epoch=10000, watchdog=False) == 2
    assert (tmp_path/'SUMMARY.md').exists() and current['status']=='deadline_stop'
    output = capsys.readouterr().out
    assert 'PKG_DEADLINE_STOP' in output and 'PKG_GATE_FAIL' not in output and 'PKG_ARM_FAIL' not in output


def test_expired_start_never_enters_gate_arm_or_fleet(config, tmp_path):
    def forbidden(*args):
        pytest.fail('launch/preflight after expired shutdown cutoff')
    assert run_package(config, state(config), tmp_path, gate_runner=forbidden, arm_runner=forbidden,
                       fleet_context=forbidden, clock=lambda:0, wall_clock=lambda:1000,
                       deadline_epoch=1000, watchdog=False) == 2


def test_watchdog_interrupts_blocking_work_without_launching_processes():
    budget = DeadlineBudget(time.time()+0.05, 7200, 0)
    with pytest.raises(PackageDeadlineReached):
        with deadline_watchdog(budget):
            time.sleep(1)


def test_package_watchdog_stops_blocking_gate_and_writes_summary(config, tmp_path, capsys):
    current = state(config)
    def blocked(*args):
        time.sleep(1)
    assert run_package(config, current, tmp_path, gate_runner=blocked,
                       deadline_epoch=time.time()+config['shutdown_reserve_s']+0.05) == 2
    assert current['status']=='deadline_stop' and not current['gates']
    assert 'PKG_DEADLINE_STOP' in (tmp_path/'SUMMARY.md').read_text()
    assert 'PKG_DEADLINE_STOP' in capsys.readouterr().out


def test_cleanup_terms_all_groups_before_waiting_and_bounds_worker_delay():
    now, events, killed = [0.], [], set()
    class Process:
        def __init__(self, pid):
            self.pid = pid
        def poll(self):
            return -9 if self.pid in killed else None
        def wait(self, timeout):
            assert self.pid in killed
    def killpg(pid, sig):
        events.append((pid,sig,now[0]))
        if sig==signal.SIGKILL:
            killed.add(pid)
    stop_owned_groups([('server',Process(1)),('relay',Process(2)),('worker',Process(3))],
                      hard_deadline=30, wall_deadline=30, clock=lambda:now[0], wall_clock=lambda:now[0],
                      sleep=lambda delay:now.__setitem__(0,now[0]+delay), killpg=killpg)
    assert events[:3] == [(1,signal.SIGTERM,0),(2,signal.SIGTERM,0),(3,signal.SIGTERM,0)]
    assert all(t <= 10.001 for pid,sig,t in events if pid in (1,2) and sig==signal.SIGKILL)
    assert max(t for _,_,t in events) <= 25.001


def test_deadline_skips_can_resume_with_new_explicit_reservation(config, tmp_path):
    current = state(config)
    assert run_package(config, current, tmp_path, gate_runner=gate, arm_runner=lambda spec,*args:result(spec),
                       clock=lambda:0, wall_clock=lambda:0, deadline_epoch=200, watchdog=False) == 2
    assert current['skipped_deadline'] and not current['completed']
    assert run_package(config, current, tmp_path, gate_runner=gate, arm_runner=lambda spec,*args:result(spec),
                       clock=lambda:0, wall_clock=lambda:0, deadline_epoch=10000, watchdog=False) == 0
    assert len(current['completed'])==18 and not current['skipped_deadline']
