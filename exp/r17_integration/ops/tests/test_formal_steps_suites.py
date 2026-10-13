"""CPU-only step-study dispatch and exact original-identity suite selection."""
from dataclasses import asdict
import copy
import json

import numpy as np
import pytest
import yaml

from exp.r17_integration.ops import formal_eval as launch
from exp.r17_integration.ops.formal_driver import plan
from exp.r17_integration.ops.tests.test_formal_horizon import fixture
from exp.r17_integration.common.config import build_shared_storage
from exp.r17_integration.common.serving import make_connection_factory


def step_fixture(tmp_path, model, k):
    provider, campaign, old = fixture(tmp_path, model, 32 if model == 'fastwam' else 16)
    campaign.update(denoising_steps=k, official_50=True)
    config = launch.full_config(provider, campaign)
    config.recipe = old.recipe
    (tmp_path/'full.yaml').write_text(yaml.safe_dump(asdict(config)))
    sampler = provider.sampler_for_steps(k)
    (tmp_path/'execution_sampler.json').write_text(json.dumps(asdict(sampler)))
    return provider, campaign, config, sampler


@pytest.mark.parametrize('model,k', [('dp', 5), ('dp', 10), ('dp', 100),
    ('fastwam', 1), ('fastwam', 2), ('fastwam', 10), ('cosmos', 1), ('cosmos', 3), ('cosmos', 5)])
def test_step_binding_plan_real_core_and_actual_nfe(tmp_path, model, k):
    provider, campaign, config, sampler = step_fixture(tmp_path, model, k)
    _, _, tasks, endpoints = plan(campaign)
    commit = provider.model_spec.layout.replan_steps
    assert len(tasks) == 2000
    assert {t.extra['num_steps'] for t in tasks} == {k}
    assert {t.extra['commit_horizon'] for t in tasks} == {commit}
    assert endpoints[0].allowed_k == (k,)
    assert config.adapter_binding['sampler_identity'] == sampler.identity
    policy = make_connection_factory(provider, config, build_shared_storage(config))()
    task = tasks[0]
    policy.on_episode_start(extra_metadata=dict(task.extra, original_init_idx=0, task_id=0,
        attempt=1, seed=7, explicit_seed=42, ack_protocol='r17.execution.v1'))
    obs = {'state': np.arange(7, dtype=np.float32)}
    rows = []
    for seq in range(2):
        request = dict(obs)
        if seq:
            request['__execution__'] = dict(decision_seq=seq-1, executed_count=commit, terminal=False)
        result = policy.infer(request)
        assert len(result['actions']) == commit
        rows.append(dict(result['__hit_meta__'], task_uid=task.task_uid))
    policy.on_episode_end(execution=dict(decision_seq=1, executed_count=1, terminal=True))
    assert policy.adapter.counts['nfe'] == 2*k
    assert launch.horizon_execution_verified(json.loads(json.dumps(rows)), campaign, asdict(provider.model_spec))
    sealed = asdict(sampler)
    sealed['num_steps'] += 1
    (tmp_path/'execution_sampler.json').write_text(json.dumps(sealed))
    with pytest.raises(ValueError, match='sealed'):
        plan(campaign)


@pytest.mark.parametrize('model,horizon', [('dp', 16), ('fastwam', 32), ('cosmos', 16)])
def test_suite_filter_retains_all_500_original_episode_identities(tmp_path, model, horizon):
    _, campaign, config = fixture(tmp_path, model, horizon)
    campaign['official_50'] = True
    benchmark, sealed_model, original, _ = plan(campaign)
    before = (tmp_path/'full.yaml').read_bytes()
    campaign['suites'] = ['libero_spatial', 'libero_10']
    selected_benchmark, selected_model, selected, _ = plan(campaign)
    expected = [t for t in original if t.experiment in campaign['suites']]
    assert [asdict(t) for t in selected] == [asdict(t) for t in expected]
    assert selected_benchmark.identity == benchmark.identity
    assert selected_model == sealed_model and (tmp_path/'full.yaml').read_bytes() == before
    assert len(selected) == 1000 and len(original) == 2000
    for suite in campaign['suites']:
        cohort = [t for t in selected if t.experiment == suite]
        assert len(cohort) == 500
        assert len({t.task_id for t in cohort}) == 10
        for task_id in {t.task_id for t in cohort}:
            assert {t.orig_init_state_idx for t in cohort if t.task_id == task_id} == set(range(50))


@pytest.mark.parametrize('argv', [
    ['--model', 'cosmos_libero', '--denoising-steps', '4'],
    ['--model', 'dp_libero', '--denoising-steps', '1'],
    ['--model', 'fastwam_libero', '--denoising-steps', '3'],
    ['--model', 'dp_mw', '--denoising-steps', '1'],
    ['--denoising-steps', '10', '--commit-horizon', '5'],
    ['--suites', 'libero_spatial'],
    ['--official-50', '--suites', 'unknown'],
    ['--official-50', '--suites', 'libero_10', 'libero_10'],
    ['--official-50', '--smoke', '--suites', 'libero_spatial'],
    ['--model', 'cosmos_libero', '--native-parity-only', '--denoising-steps', '3'],
])
def test_cli_rejects_confounding_or_invalid_study_options(argv):
    with pytest.raises(SystemExit):
        launch.parse_args(['--prepare-only', *argv])


def test_flags_are_default_off_and_reject_library_campaigns():
    args = launch.parse_args(['--prepare-only'])
    assert args.denoising_steps is None and args.suites is None
    with pytest.raises(ValueError):
        launch.execution_steps(dict(model='dp', domain='libero', library=True, denoising_steps=10), 100)


def test_aggregate_checks_selected_sampler_identity_and_observed_nfe(tmp_path):
    provider, campaign, _, sampler = step_fixture(tmp_path, 'dp', 10)
    _, _, tasks, _ = plan(campaign)
    task = asdict(tasks[0])
    (tmp_path/'driver').mkdir()
    def write(name, value):
        (tmp_path/name).write_text(json.dumps(value))
    write('campaign.json', campaign)
    write('EPISODE_PLAN.json', dict(tasks=[task], model=asdict(provider.model_spec), benchmark_protocol='fixture'))
    write('driver/DRIVER_RECEIPT.json', dict(status='PASS', elapsed_seconds=1))
    write('driver/journal.jsonl', dict(task_uid=task['task_uid'], accepted=True, status='done', attempt=1, success=True))
    row = dict(task_uid=task['task_uid'], accepted=True, attempt=1, decision_seq=0, cursor=0,
               commit_horizon=5, capture_enabled=False, source='policy', actual_nfe=10,
               stage_calls=[1,1,10], sampler_identity=sampler.identity)
    write('driver/per_step.jsonl', row)
    result = launch.aggregate(tmp_path)
    assert result['status'] == 'PASS' and result['denoising_steps'] == 10
    for field, value in [('actual_nfe', 100), ('sampler_identity', 'wrong'), ('cursor', 5)]:
        bad = copy.deepcopy(row)
        bad[field] = value
        write('driver/per_step.jsonl', bad)
        assert launch.aggregate(tmp_path)['status'] == 'PARTIAL'


@pytest.mark.parametrize('k', [5, 10])
def test_selected_dp_ddim_grid_and_updates_match_diffusers(k):
    import torch
    from pathlib import Path
    from diffusers import DDIMScheduler
    from exp.r17_integration.models.dp.sampler import SamplerProfile
    cfg = json.loads((Path(__file__).parents[2]/'models/dp/samples/audited_libero_metadata.json').read_text())['config']['model']
    profile = SamplerProfile(num_steps=k)
    scheduler = profile.scheduler(cfg)
    reference = DDIMScheduler.from_config(scheduler.config)
    reference.set_timesteps(k)
    assert profile.kind == 'ddim' and profile.eta == 0
    assert tuple(scheduler.timesteps.tolist()) == tuple(range(99, -1, -(100//k)))
    x = torch.linspace(-.7, .7, 16*7).reshape(1,16,7)
    for t in scheduler.timesteps:
        prediction = .2*x
        expected = reference.step(prediction, t, x, eta=0).prev_sample
        x = profile.step(scheduler, prediction, t, x, torch.Generator())
        assert torch.equal(x, expected)


def test_dp_graph_startup_check_receives_selected_sampler_on_cpu(monkeypatch):
    from types import SimpleNamespace as NS
    import torch
    from exp.mw_finetune.dp import cuda_graph
    from exp.r17_integration.common import serving
    from exp.r17_integration.common.fake_adapter import fake_sampler
    from exp.r17_integration.ops.formal_server import enable_dp_graph
    selected = fake_sampler(5)
    calls = []
    def full_call(adapter, obs, request, sampler=None):
        calls.append(sampler)
        return None, NS(actions=np.zeros((16,7)))
    monkeypatch.setattr(serving, 'full_call', full_call)
    monkeypatch.setattr(cuda_graph, 'DenoiserGraph', lambda model: object())
    monkeypatch.setattr(cuda_graph, 'check_observations', lambda domain, stats:
        [dict(seed=i, prompt=str(i)) for i in range(3)])
    adapter = NS(domain='libero', model=NS(_policy=NS(model=torch.nn.Identity().eval())),
                 legacy_policy=NS(stats={}, infer=lambda obs: dict(actions=np.zeros((16,7))),
                                  _infer=lambda obs, **kwargs: dict(actions=np.zeros((16,7)))))
    result = enable_dp_graph(adapter, launch.benchmark(True), sampler=selected)
    assert calls == [selected]*6 and result['execution_sampler'] == asdict(selected)
    assert result['passed']
