"""Canonical task numbering is an explicit, content-bound benchmark contract."""
from dataclasses import replace
import json

import pytest

from exp.r17_integration.common.contracts import digest
from exp.r17_integration.benchmarks.metaworld import FORMAL_SPEC, MetaWorldBenchmarkAdapter
from exp.r17_integration.conductor.manifest import episode_tasks, load_benchmark_spec
from exp.r17_integration.tests.benchmarks.test_libero import spec as libero_spec
from exp.r17_integration.tests.benchmarks.test_metaworld import Client, Env
from openpi.conductor.task import ServerEndpoint


@pytest.mark.parametrize('libero', [False, True], ids=['mw', 'libero'])
def test_canonical_table_identity_is_published_and_not_a_subset(tmp_path, libero):
    spec = libero_spec() if libero else FORMAL_SPEC
    names = tuple(r[0] for r in spec.tasks)
    from exp.r17_integration.common.benchmarks import register_benchmark, task_table_identity
    register_benchmark(spec)
    assert spec.ordered_tasks == names
    assert spec.task_table_identity == task_table_identity(spec.identity)
    document = spec.manifest()
    assert tuple(document['ordered_tasks']) == names
    assert document['task_table_identity'] == spec.task_table_identity
    assert digest(tuple(reversed(names))) != spec.task_table_identity
    assert digest(names[:2]) != spec.task_table_identity
    # Additive metadata must not change the pre-existing benchmark protocol.
    assert spec.identity == spec.to_contract().identity
    path = tmp_path/'spec.json'
    path.write_text(json.dumps(document))
    assert load_benchmark_spec(path).task_table_identity == spec.task_table_identity
    bad = {**document, 'task_table_identity': digest(tuple(reversed(names)))}
    path.write_text(json.dumps(bad))
    with pytest.raises(ValueError, match='task table'):
        load_benchmark_spec(path)


def test_mw_episode_and_runner_publish_same_full_task_table_for_one_selected_task():
    spec, client = FORMAL_SPEC, Client()
    task = episode_tasks(campaign='r2', arm='plain', bundle_id='plain', endpoint=ServerEndpoint('local', 1),
        model_identity='m', norm_identity='n', layout_identity='l', partition='smoke',
        names=['reach-v3'], indices=[7])[0]
    assert task.extra['task_table_identity'] == spec.task_table_identity
    runner = MetaWorldBenchmarkAdapter(env_factory=lambda *args: Env(success=1)).make_episode_runner(
        client_factory=lambda _: client)
    outcome = runner.run(task, lambda *args: None)
    metadata = next(data for kind, data in client.calls if kind == 'start')['extra_metadata']
    assert metadata['task_table_identity'] == spec.task_table_identity
    assert client.requests[0]['__request__']['task_name'] == 'reach-v3'
    assert client.requests[0]['__request__']['task_id'] == FORMAL_SPEC.ordered_tasks.index('reach-v3')
    assert client.requests[0]['__request__']['task_table_identity'] == spec.task_table_identity
    assert all(row['task_table_identity'] == spec.task_table_identity
               for row in outcome.per_step_rows if row.get('_kind') != 'episode_summary')
    bad = replace(task, extra={**task.extra, 'task_table_identity': digest(('reach-v3',))})
    with pytest.raises(ValueError, match='task table'):
        runner.run(bad, lambda *args: None)
