"""Package lifecycle proofs with stubs: no listeners, GPU or remote launches."""
import hashlib
import json
from pathlib import Path

import pytest

from exp.decouple.benchmark_gpu import accepted_cohort, expand_tokens
from exp.decouple.current_fleet import can_admit, replace_option, warm_copy
from exp.decouple.gpu_package import best_arms, gate_argv, repeat_order, run_package as _run_package
from exp.decouple.package_config import REPO, canonical, make_arm, materialize, validate, worker_command
from exp.decouple.package_driver import BenchmarkStrategy, agent_lease, measurement_uids, remote_preflight
from exp.decouple.summarize_gpu import paired_ratios
from openpi.conductor.task import ServerEndpoint


def run_package(*args, **kwargs):
    # Lifecycle stubs remain independent of the date on which tests are run.
    kwargs.setdefault('wall_clock', lambda:0)
    kwargs.setdefault('watchdog', False)
    return _run_package(*args, **kwargs)


@pytest.fixture
def config():
    return json.loads((REPO / 'exp/decouple/benchmark.local.json').read_text())


def seal(config):
    config.pop('config_sha256')
    config['config_sha256'] = hashlib.sha256(canonical(config)).hexdigest()
    return config


def state(config):
    return dict(package_id='cpu_fixture', config_sha256=config['config_sha256'],
                config_path=str(REPO / 'exp/decouple/benchmark.local.json'),
                gates={}, completed={}, selected={})


def result(spec):
    rates = dict(current_1=10, current_max=65, split_4=35, split_8=90, split_16=70)
    return dict(name=spec['name'], logical_arm=spec['logical_arm'], model=spec['model'],
                fronts=spec['fronts'], model_processes=6 if spec['model_processes']=='auto' else spec['model_processes'],
                episodes_completed=300, episodes_per_min=rates[spec['logical_arm']], elapsed_s=300,
                gpu_util_mean_pct=50, system_cpu_mean_pct=10, cpu_proc_mean_pct=300,
                gpu_memory_peak_mb=40000, latency_p50_ms=10, latency_p99_ms=100,
                manifest_sha256=spec['manifest_sha256'], within_budget=True,
                episode_failures=2, episode_error_records=0, warmup_excluded=True)


def test_frozen_config_and_all_arms_share_b_manifest(config):
    validate(config, check_inputs=False)
    for model, case in config['cases'].items():
        for kind in config['initial_arms']:
            arm = make_arm(config, model, kind, 'fixture')
            assert arm['manifest_sha256'] == case['manifest_sha256']
            assert '--batch-size' in arm['server_command'] if arm['split'] else '--copies' in arm['server_command']
        assert case['row']['kwargs']['task_levels'] == case['manifest']['task_levels']
        argv = worker_command(config, model, 'lease', 'token')
        assert argv[argv.index('--phase')+1] == 'dev'
        assert argv[argv.index('--workers')+1] == '64'
        assert config['worker_island'] in config['worker_environment']['PYTHONPATH']


@pytest.mark.parametrize('mutation', ['pool', 'port', 'duplicates', 'task_levels', 'manifest_hash'])
def test_schema_rejects_unfair_or_reserved_inputs(config, mutation):
    case = config['cases']['groot']
    if mutation == 'pool':
        case['manifest']['init_pool'] = 'A'
    elif mutation == 'port':
        config['public_port'] = 23179
    elif mutation == 'duplicates':
        case['manifest']['pairs'][1] = case['manifest']['pairs'][0]
    elif mutation == 'task_levels':
        case['manifest']['task_levels']['0'] = 'off'
    else:
        case['manifest_sha256'] = 'wrong'
    with pytest.raises((AssertionError, ValueError)):
        validate(seal(config), check_inputs=False)


def test_json_argv_is_not_shell_or_format_code():
    value = '{"task_levels":{"0":"ir45"},"label":"$(false) `false`"}'
    assert expand_tokens(value, port=23100, out='root') == value
    assert expand_tokens('{out}/x:{backend_port}', out='/safe path', backend_port=23400) == '/safe path/x:23400'


def test_gate_and_arm_order_selection_and_alternating_pairs(config, tmp_path, capsys):
    calls = []
    def gate(case, k, out, timeout):
        calls.append((case['model'], k))
        return dict(passed=True, path=str(out))
    def arm(spec, root, index, budget):
        calls.append(spec['name'])
        if 'repeat' in spec['name'] and spec['logical_arm'].startswith('current'):
            assert spec['model_processes'] == 6
            assert spec['server_command'][spec['server_command'].index('--copies')+1] == '6'
        return result(spec)
    current = state(config)
    assert run_package(config, current, tmp_path, gate_runner=gate, arm_runner=arm) == 0
    assert calls[:4] == [('pi05',1),('pi05',2),('groot',1),('groot',2)]
    assert len(calls) == 22 and len(current['completed']) == 18
    assert all('groot_initial' in str(call) for call in calls[4:9])
    assert all('pi05_initial' in str(call) for call in calls[9:14])
    assert [name.split('_initial_')[1].split('_try')[0] for name in calls[4:9]] == ['current_1','current_max','split_8','split_4','split_16']
    assert all('groot_repeat' in str(call) for call in calls[14:18])
    assert all('pi05_repeat' in str(call) for call in calls[18:22])
    for model in ('groot','pi05'):
        assert current['selected'][model] == dict(current='current_max', split='split_8', current_copies=6)
        assert paired_ratios(current, model) == [90/65]*3
    assert repeat_order(1) == ['split','current'] and repeat_order(2) == ['current','split']
    assert 'PKG_DONE' in capsys.readouterr().out
    assert '| groot/initial/current_1 |' in (tmp_path / 'SUMMARY.md').read_text()


def test_gate_failure_prevents_every_arm(config, tmp_path, capsys):
    def fail(*args):
        raise ValueError('nonexact')
    def forbidden(*args):
        pytest.fail('arm launched after gate failure')
    current = state(config)
    assert run_package(config, current, tmp_path, gate_runner=fail, arm_runner=forbidden) == 1
    assert not current['completed'] and 'PKG_GATE_FAIL' in capsys.readouterr().out


def test_arm_failure_and_resume_fresh_attempt_skip_completed(config, tmp_path, capsys):
    calls = []
    def gate(case, k, out, timeout):
        return dict(passed=True, path=str(out))
    def first(spec, *args):
        calls.append(spec['name'])
        if spec['logical_arm'] == 'current_max':
            raise RuntimeError('front failed')
        return result(spec)
    current = state(config)
    assert run_package(config, current, tmp_path, gate_runner=gate, arm_runner=first) == 1
    assert len(calls)==2 and len(current['completed'])==1
    assert 'PKG_ARM_FAIL groot_initial_current_max' in capsys.readouterr().out
    calls.clear()
    def resumed(spec, *args):
        calls.append(spec['name'])
        return result(spec)
    def no_gate(*args):
        pytest.fail('reran completed gate')
    assert run_package(config, current, tmp_path, gate_runner=no_gate, arm_runner=resumed) == 0
    assert len(calls)==17 and calls[0].endswith('try02')
    assert not any('groot_initial_current_1' in name for name in calls)
    assert len(current['failed_attempts'])==1


def test_wall_limit_stops_without_shortening_cohort(config, tmp_path, capsys):
    ticks = iter([0,0])
    def clock():
        return next(ticks, 10000)
    current = state(config)
    assert run_package(config, current, tmp_path, clock=clock) == 2
    assert not current['completed'] and current['status'] == 'deadline_stop'
    assert 'PKG_DEADLINE_STOP' in capsys.readouterr().out


def test_warmup_barrier_and_only_measured_uids(config, tmp_path):
    case = config['cases']['groot']
    strategy = BenchmarkStrategy({}, case, tmp_path)
    endpoint = ServerEndpoint('fixture', 23100)
    graph = strategy.plan([case['row']['arm']], {case['row']['arm']:endpoint})
    assert set(graph.upstreams('eval')) == {'warmup'}
    warm = graph.stages['warmup'].episodes
    measured = graph.stages['eval'].episodes
    assert len(warm)==10 and len(measured)==300
    assert not set(ep.task_uid for ep in warm) & measurement_uids(case)
    assert all(ep.phase=='eval' and ep.extra['init_pool']=='B' for ep in warm+measured)
    rows = [dict(task_uid=ep.task_uid, attempt=1, status='done', accepted=True, success=True) for ep in warm+measured]
    rows += [dict(task_uid=measured[0].task_uid, attempt=0, status='done', accepted=False, success=False)]
    accepted = accepted_cohort(rows, measurement_uids(case))
    assert len(accepted)==300 and all(r['success'] for r in accepted.values())


def test_summary_rejects_mismatched_pair(config):
    current = state(config)
    current['selected']['groot'] = dict(current='current_1', split='split_4')
    for kind in ('current_1', 'split_4'):
        current['completed'][f'groot/initial/{kind}'] = result(make_arm(config, 'groot', kind, 'fixture'))
    current['completed']['groot/initial/split_4']['manifest_sha256'] = 'different'
    with pytest.raises(ValueError, match='manifest'):
        paired_ratios(current, 'groot')


def test_current_copy_admission_and_unique_argv():
    assert can_admit(39000, 46000, 6656)
    assert not can_admit(41000, 46000, 6656)
    assert replace_option(['--server-tag','old','--port','23400'], '--server-tag', 23401) == ['--server-tag','23401','--port','23400']


def test_capacity_workspace_probe_uses_original_protocol_and_closes(config):
    calls = []
    class Client:
        def __getattr__(self, name):
            def invoke(*args, **kwargs):
                calls.append((name,args,kwargs))
                return {'actions':[[0]*7]} if name == 'infer' else {'__ack__':'ok'}
            return invoke
    case = config['cases']['groot']
    warm_copy(23401, {'entries':[{'row':case['row']}]}, 'groot',
              client_factory=lambda port:Client(), observation_loader=lambda model:{'prompt':'fixture'})
    assert [name for name,_,_ in calls] == ['load_cache_config','select_bundle','episode_start','infer','episode_end','close']
    assert calls[2][2]['extra_metadata']['task_uid'] == 'B:decouple_capacity:23401'


def test_capacity_probe_failure_still_closes(config):
    closed = []
    class Client:
        def __getattr__(self, name):
            return lambda *args,**kwargs:{}
        def close(self):
            closed.append(True)
    case = config['cases']['pi05']
    with pytest.raises(ValueError, match='produce actions'):
        warm_copy(23401, {'entries':[{'row':case['row']}]}, 'pi05',
                  client_factory=lambda port:Client(), observation_loader=lambda model:{'prompt':'fixture'})
    assert closed == [True]


@pytest.mark.parametrize('model', ['pi05','groot'])
def test_capacity_probe_recorded_b_fixture_is_valid_without_model(config, model):
    observed = []
    class Client:
        def __getattr__(self, name):
            return lambda *args,**kwargs:{}
        def infer(self, observation):
            observed.append(observation)
            return {'actions':[[0]*7]}
    case = config['cases'][model]
    warm_copy(23401, {'entries':[{'row':case['row']}]}, model, client_factory=lambda port:Client())
    size = 256 if model=='groot' else 224
    assert observed[0]['observation/image'].shape == (size,size,3)
    assert observed[0]['observation/state'].shape == (8,)
    assert isinstance(observed[0]['prompt'], str)


def test_remote_preflight_is_read_only_and_lease_stops_only_own_agent(config, tmp_path, monkeypatch):
    calls = []
    remote = lambda host, argv, **kwargs:calls.append((host, argv))
    remote_preflight(config, remote)
    assert len(calls)==3 and all(host=='timan107' for host, _ in calls)
    assert 'sha256sum --quiet -c worker.sha256' in calls[1][1][-1]
    calls.clear()
    with agent_lease(config, 'groot', tmp_path, remote):
        pass
    assert any('nohup env' in argv[-1] for _,argv in calls)
    assert any('active=False' in argv[-1] for _,argv in calls)
    assert not any('pkill' in str(argv) for _,argv in calls)


def test_materialized_group_and_frozen_method_argv_parse(config, tmp_path):
    from exp.offline_search.rounds.r16.conductor import read_group
    for model in ('groot', 'pi05'):
        out = tmp_path / model
        out.mkdir()
        case = config['cases'][model]
        group = materialize(case, out)
        assert read_group(out / 'group.json') == group
        assert group['init_pool']=='B' and group['entries'][0]['row']['dev']
        assert str(out / 'evidence') in Path(group['entries'][0]['row']['yaml']).read_text()
        assert Path(group['entries'][0]['row']['yaml']).stem == case['row']['arm']
        argv = gate_argv(case, 2, out / 'gate.json')
        assert argv[argv.index('--k')+1] == '2' and argv[argv.index('--batch-size')+1] == '1'
