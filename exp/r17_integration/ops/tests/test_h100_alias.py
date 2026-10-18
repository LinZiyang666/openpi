"""Process-level h100 alias: unset is the H100 itself; set reroutes every h100 effect."""
from types import SimpleNamespace

import pytest

from exp.r17_integration.ops import direct_transfer, formal_eval as f

ALIAS = dict(R17_H100_NODE='h100-temp', R17_H100_HOST='149.165.172.126', R17_DRIVER_PORT='23194')


@pytest.fixture
def alias(monkeypatch):
    for key, value in ALIAS.items():
        monkeypatch.setenv(key, value)


@pytest.fixture(autouse=True)
def unset(monkeypatch):
    for key in ALIAS:
        monkeypatch.delenv(key, raising=False)


def tether(monkeypatch):
    calls = []
    def execute(argv, seconds=60, env=None, cwd=None):
        calls.append(argv);return '{}'
    monkeypatch.setattr(f, 'execute', execute)
    return calls


def test_unset_is_the_h100(monkeypatch):
    calls = tether(monkeypatch)
    assert f.h100_alias() is None and f.physical_node('h100') == 'h100' and f.h100_host() == '149.165.153.233'
    f.remote('h100', ['true']);f.remote('timan107', ['true'])
    assert [a[a.index('--')-1] for a in calls] == ['h100', 'timan107']
    f.check_execution_host(dict(server_node='h100'))
    with pytest.raises(ValueError, match='bound to execution host h100-temp'):
        f.check_execution_host(dict(server_node='h100', execution_host='h100-temp'))


def test_alias_reroutes_tether_rsync_and_endpoints(monkeypatch, tmp_path, alias):
    calls = tether(monkeypatch)
    f.remote('h100', ['true']);f.remote('timan1', ['true'])
    assert [a[a.index('--')-1] for a in calls] == ['h100-temp', 'timan1']
    calls.clear()
    def daemon(*a):return dict(pid=1, root='/stub')
    monkeypatch.setattr(direct_transfer, 'daemon', daemon)
    monkeypatch.setattr(direct_transfer, 'fcntl', SimpleNamespace(flock=lambda *a: None, LOCK_EX=0))
    (tmp_path/'a').write_text('a')
    direct_transfer.transfer(['rsync', 'push', str(tmp_path/'a'), 'h100:/tmp/x/a'], remote=f.remote, execute=f.execute)
    assert {a[a.index('--')-1] for a in calls if a[0] == 'tether'} == {'h100-temp'}
    assert [w for a in calls if a[0] == 'rsync' for w in a if w.startswith('rsync://')][0].startswith('rsync://149.165.172.126:')
    c = dict(server_node='h100', public_port=23170, driver_port=23194, worker_nodes={'timan1': 1}, cache_study='S')
    f.open_tunnels(c, tmp_path)
    assert c['endpoint'] == '149.165.172.126:23170' and f.backend_control_host(c) == '149.165.172.126'
    with pytest.raises(ValueError, match='bound to execution host h100'):
        f.check_execution_host(dict(server_node='h100'))
    f.check_execution_host(dict(server_node='h100', execution_host='h100-temp'))


@pytest.mark.parametrize('env', [dict(R17_H100_NODE='h100-temp'), dict(ALIAS, R17_DRIVER_PORT='23420'),
                                 dict(ALIAS, R17_H100_NODE='h100')])
def test_partial_or_invalid_alias_is_refused(monkeypatch, env):
    for key, value in env.items():
        monkeypatch.setenv(key, value)
    with pytest.raises(ValueError, match='together'):
        f.physical_node('h100')
