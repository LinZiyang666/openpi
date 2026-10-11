"""CPU-only witnesses for the default-off full-policy commitment study."""
from dataclasses import asdict, replace
import copy
import json
import uuid

import numpy as np
import pytest
import yaml

from exp.r17_integration.ops import formal_eval as launch
from exp.r17_integration.ops.formal_driver import plan
from exp.r17_integration.common.benchmarks import register_benchmark
from exp.r17_integration.common.config import build_shared_storage
from exp.r17_integration.common.fake_adapter import fake_provider, fake_spec
from exp.r17_integration.common.qualification import config_for_probe
from exp.r17_integration.common.serving import make_connection_factory


def fixture(tmp_path, model, horizon, commit=None):
    benchmark = launch.benchmark(True)
    register_benchmark(benchmark)
    spec = replace(fake_spec(horizon=horizon, action_dim=7), allowed_benchmark_protocols=(benchmark.identity,))
    if model == 'cosmos':
        spec = replace(spec, layout=replace(spec.layout, replan_steps=16))
    provider = fake_provider('horizon.' + uuid.uuid4().hex, spec)
    campaign = dict(model=model, domain='libero', local_assets=str(tmp_path), endpoint_host='unused',
                    public_port=23160, smoke=False, run_name='horizon_cpu', workers=1)
    if commit is not None:
        campaign['commit_horizon'] = commit
    config = launch.full_config(provider, campaign)
    config.recipe.ordered_tasks = list(benchmark.ordered_tasks)
    config.recipe.task_table_identity = benchmark.task_table_identity
    (tmp_path/'full.yaml').write_text(yaml.safe_dump(asdict(config)))
    (tmp_path/'benchmark.json').write_text(json.dumps(benchmark.manifest()))
    (tmp_path/'model_spec.json').write_text(json.dumps(asdict(spec)))
    return provider, campaign, config


def test_default_configuration_is_unchanged(tmp_path):
    provider, campaign, config = fixture(tmp_path, 'dp', 16)
    old = config_for_probe(provider, 4, 5)
    old.enabled, old.checkpoints, old.recipe.enabled = False, {}, False
    old.recipe.ordered_tasks = config.recipe.ordered_tasks
    old.recipe.task_table_identity = config.recipe.task_table_identity
    assert asdict(old) == asdict(config)
    assert launch.parse_args(['--prepare-only']).commit_horizon is None
    _, _, tasks, _ = plan(campaign)
    assert len(tasks) == 2000
    assert {t.extra['commit_horizon'] for t in tasks} == {5}


@pytest.mark.parametrize('model,horizon,commit', [
    ('dp', 16, 5), ('dp', 16, 10), ('dp', 16, 15), ('dp', 16, 16),
    ('fastwam', 32, 10), ('fastwam', 32, 16), ('fastwam', 32, 20), ('fastwam', 32, 32),
    ('cosmos', 16, 16)])
def test_sealed_plan_and_real_core_execute_exact_chunk_tail(tmp_path, model, horizon, commit):
    provider, campaign, config = fixture(tmp_path, model, horizon, commit)
    _, _, tasks, _ = plan(campaign)
    assert len(tasks) == 2000 and {t.orig_init_state_idx for t in tasks} == set(range(50))
    assert {t.extra['commit_horizon'] for t in tasks} == {commit}
    policy = make_connection_factory(provider, config, build_shared_storage(config))()
    task = tasks[0]
    policy.on_episode_start(extra_metadata=dict(task.extra, original_init_idx=0, task_id=0, attempt=1, seed=7, explicit_seed=42,
        ack_protocol='r17.execution.v1', partial_tail_protocol='r17.partial_tail.v1'))
    replan = provider.model_spec.layout.replan_steps
    obs = {'state': np.arange(7, dtype=np.float32)}
    seq, cursor, previous, first = 0, 0, None, None
    rows, controls = [], []
    while cursor < commit:
        request = dict(obs)
        if previous is not None:
            request['__execution__'] = previous
        output = policy.infer(request)
        if first is None:
            first = output['actions'].copy()
        assert output['actions'].tobytes() == first[cursor:].tobytes()
        count = min(replan, commit-cursor)
        controls.extend(output['actions'][:count])
        rows.append(dict(output['__hit_meta__'], task_uid=task.task_uid))
        previous = dict(decision_seq=seq, executed_count=count, terminal=False)
        cursor += count
        seq += 1
    assert np.array_equal(np.asarray(controls), first)
    assert policy.adapter.counts['stage1'] == 1
    assert policy.adapter.counts['nfe'] == 4
    # A fresh inference happens only after all committed controls were acknowledged.
    output = policy.infer(dict(obs, __execution__=previous))
    assert policy.adapter.counts['stage1'] == 2
    rows.append(dict(output['__hit_meta__'], task_uid=task.task_uid))
    policy.on_episode_end(execution=dict(decision_seq=seq, executed_count=1, terminal=True))
    # Serialize tuples exactly as the real journal does.
    rows = json.loads(json.dumps(rows))
    assert launch.horizon_execution_verified(rows, campaign, asdict(provider.model_spec))
    for field, value in [('source', 'cache'), ('actual_nfe', 1), ('cursor', 1), ('commit_horizon', commit+1)]:
        bad = copy.deepcopy(rows)
        bad[0][field] = value
        assert not launch.horizon_execution_verified(bad, campaign, asdict(provider.model_spec))
    assert not launch.horizon_execution_verified(rows[1:], campaign, asdict(provider.model_spec))
    assert not launch.horizon_execution_verified(rows+[rows[0]], campaign, asdict(provider.model_spec))
    campaign['commit_horizon'] = commit+1 if commit < horizon else replan-1
    with pytest.raises(ValueError):
        plan(campaign)


@pytest.mark.parametrize('model,commit', [('dp_libero', 4), ('dp_libero', 17),
    ('fastwam_libero', 33), ('cosmos_libero', 15), ('cosmos_libero', 17), ('dp_mw', 10)])
def test_cli_rejects_unsupported_commitment(model, commit):
    with pytest.raises(SystemExit):
        launch.parse_args(['--prepare-only', '--model', model, '--commit-horizon', str(commit)])


def test_partial_tail_plan_cannot_silently_drop_negotiation(tmp_path):
    _, campaign, config = fixture(tmp_path, 'dp', 16, 16)
    config.adapter_binding.pop('allow_partial_commit')
    (tmp_path/'full.yaml').write_text(yaml.safe_dump(asdict(config)))
    with pytest.raises(ValueError):
        plan(campaign)


def test_library_campaign_cannot_opt_into_horizon_study():
    with pytest.raises(ValueError, match='plain'):
        launch.execution_horizon(dict(domain='libero', model='dp', library=True, commit_horizon=10),
                                 dict(replan_steps=5, horizon=16))


def test_aggregate_accepts_only_full_policy_and_verified_tails(tmp_path):
    provider, campaign, _ = fixture(tmp_path, 'dp', 16, 16)
    _, _, tasks, _ = plan(campaign)
    task = asdict(tasks[0])
    driver = tmp_path/'driver'
    driver.mkdir()
    def write(name, data):
        (tmp_path/name).write_text(json.dumps(data))
    write('campaign.json', campaign)
    write('EPISODE_PLAN.json', dict(tasks=[task], model=asdict(provider.model_spec), benchmark_protocol='fixture'))
    write('driver/DRIVER_RECEIPT.json', dict(status='PASS', elapsed_seconds=1))
    write('driver/journal.jsonl', dict(task_uid=task['task_uid'], accepted=True, status='done', attempt=1, success=True))
    rows = [dict(task_uid=task['task_uid'], accepted=True, attempt=1, decision_seq=seq,
                 cursor=cursor, commit_horizon=16, capture_enabled=False,
                 source='policy' if cursor == 0 else 'policy_tail', actual_nfe=4 if cursor == 0 else 0,
                 stage_calls=[1,1,4] if cursor == 0 else [0,0,0])
            for seq, cursor in enumerate([0,5,10,15,0])]
    def steps(values):
        (driver/'per_step.jsonl').write_text(''.join(json.dumps(r)+'\n' for r in values))
    steps(rows)
    summary = launch.aggregate(tmp_path)
    assert summary['status'] == 'PASS'
    assert (summary['policy_calls'], summary['policy_tail_requests']) == (2,3)
    for bad in [rows[1:], rows+[rows[0]], [dict(rows[0], source='cache')]+rows[1:]]:
        steps(bad)
        assert launch.aggregate(tmp_path)['status'] == 'PARTIAL'
