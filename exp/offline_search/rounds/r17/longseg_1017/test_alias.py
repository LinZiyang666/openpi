"""Stage-time execution-host rebind: routing fields only, untouched without the alias."""
import copy
import json

import pytest

from . import io_budget

ALIAS = dict(R17_H100_NODE='h100-temp', R17_H100_HOST='149.165.172.126', R17_DRIVER_PORT='23194')


def job(tmp_path, monkeypatch):
    from .ports import route
    monkeypatch.setattr(io_budget, 'LOCK', tmp_path/'reader.lock')
    for key in ALIAS:
        monkeypatch.delenv(key, raising=False)
    c = route(dict(run_name='fa_ls_x', server_node='h100', public_port=23170, driver_port=23179, replicas=2,
                   worker_nodes={'timan1': 1}, execution_worker_nodes={'timan1': 1, 'timan107': 1},
                   cache_study='S', longseg_1017={}), tmp_path)
    tasks = [dict(task_uid=f'u:eval:{i}:0', server_host=c['endpoint_host'], server_port=c['endpoint_port'],
                  orig_init_state_idx=i, extra=dict(seed=7+i)) for i in range(3)]
    (tmp_path/'campaign.json').write_text(json.dumps(c))
    (tmp_path/'EPISODE_PLAN.json').write_text(json.dumps(dict(schema='p', tasks=tasks, endpoint=c['endpoint'])))
    return c, tasks


def test_without_alias_nothing_changes(tmp_path, monkeypatch):
    from .ports import execution_host
    c, _ = job(tmp_path, monkeypatch)
    before = {p.name: p.read_bytes() for p in tmp_path.iterdir()}
    assert execution_host(copy.deepcopy(c), tmp_path) == c
    assert {p.name: p.read_bytes() for p in tmp_path.iterdir()} == before
    assert c['endpoint'] == '149.165.153.233:23170' and c['ports'][-1] == c['driver_port'] == 23199


def test_alias_rebinds_routing_only(tmp_path, monkeypatch):
    from .ports import execution_host
    c, tasks = job(tmp_path, monkeypatch)
    for key, value in ALIAS.items():
        monkeypatch.setenv(key, value)
    new = execution_host(copy.deepcopy(c), tmp_path)
    assert new['endpoint'] == '149.165.172.126:23170' and new['execution_host'] == 'h100-temp'
    assert new['driver_port'] == new['driver_public_port'] == new['ports'][-1] == 23194
    assert new['ports'][:-1] == c['ports'][:-1] and new['longseg_1017'] == c['longseg_1017']
    changed = {k for k in set(c) | set(new) if c.get(k) != new.get(k)}
    assert changed == {'ports', 'driver_port', 'driver_public_port', 'endpoint_host', 'endpoint',
                       'execution_host', 'execution_host_rebind'}
    plan = json.loads((tmp_path/'EPISODE_PLAN.json').read_text())
    assert [dict(t, server_host='149.165.172.126') for t in tasks] == plan['tasks']
    assert json.loads((tmp_path/'campaign.json').read_text()) == new
    assert len(list(tmp_path.glob('*.before_alias_*'))) == 2
    assert execution_host(copy.deepcopy(new), tmp_path) == new  # repeated prepare
    (tmp_path/'PREPARED.json').write_text('{}')
    with pytest.raises(ValueError, match='reset it before moving'):
        execution_host(copy.deepcopy(c), tmp_path)
    for key in ALIAS:
        monkeypatch.delenv(key)
    with pytest.raises(ValueError, match='bound to execution host h100-temp'):
        execution_host(copy.deepcopy(new), tmp_path)


def test_deferred_definition_built_under_alias_gets_the_lane_port(tmp_path, monkeypatch):
    """longseg_1017.workflow prepare builds the levels unit (route) and stages it in one process."""
    from .ports import execution_host, route
    monkeypatch.setattr(io_budget, 'LOCK', tmp_path/'reader.lock')
    for key, value in ALIAS.items():
        monkeypatch.setenv(key, value)
    c = route(dict(run_name='fa_ls_y', server_node='h100', public_port=23420, driver_port=23179, replicas=2,
                   worker_nodes={'timan1': 1}, longseg_1017={}), tmp_path)
    assert c['endpoint'] == '149.165.172.126:23420' and c['driver_port'] == 23199
    tasks = [dict(task_uid='u:eval:0:0', server_host=c['endpoint_host'], server_port=23420)]
    (tmp_path/'campaign.json').write_text(json.dumps(c))
    (tmp_path/'EPISODE_PLAN.json').write_text(json.dumps(dict(tasks=tasks, endpoint=c['endpoint'])))
    new = execution_host(c, tmp_path)
    assert (new['driver_port'], new['ports'][-1], new['execution_host']) == (23194, 23194, 'h100-temp')
    assert json.loads((tmp_path/'EPISODE_PLAN.json').read_text())['tasks'] == tasks
