"""CPU-only native retry after a closed dynamic R15 attempt; no node contacts."""
import copy
import hashlib
import json
from pathlib import Path
import types

import pytest

from exp.offline_search.closed_loop.ops.h100 import control, node, test_robustness
from exp.offline_search.rounds.r17.autoscale_1017 import legacy_release, switch

BASE = Path(switch.__file__).parent
BACKUP = Path('/home/weiland/.claude/jobs/35f2869b/tmp/autoscale_backup')


@pytest.mark.parametrize('version', ['before', 'after'])
def test_native_chain_ignores_dynamic_release_and_retries_with_fresh_identity(tmp_path, monkeypatch, version):
    assert not switch.enabled()
    c = control
    if version == 'before':
        name = 'exp/offline_search/closed_loop/ops/h100/control.py'
        source = (BACKUP / name.replace('/', '__')).read_bytes()
        pins = json.loads((BASE / 'PATCH_BASE.json').read_text())
        assert hashlib.sha256(source).hexdigest() == pins[name]['sha256']
        c = types.ModuleType('exp.offline_search.closed_loop.ops.h100.retry_baseline')
        c.__file__ = control.__file__
        c.__package__ = control.__package__
        exec(compile(source, c.__file__, 'exec'), c.__dict__)
    monkeypatch.setattr(test_robustness, 'control', c)
    root, events, _ = test_robustness.mock_chain(tmp_path, monkeypatch, counts=['0 0 0', '4 2 4'])
    monkeypatch.setenv('PORTS', '23230,23231,23232')
    monkeypatch.setenv('WPS', '13')
    monkeypatch.setenv('SERVER_HOST', 'h100')
    monkeypatch.setattr(c, 'legacy_chains', lambda **kw: [])
    (root / 'state').mkdir()
    (root / 'state/CHAIN.ERROR').write_text('old dynamic ARM_FAILED')
    (root / 'state/first.ERROR').write_text('old dynamic ARM_FAILED')
    (root / 'LEGACY_AUTOSCALE.json').write_text(json.dumps(dict(remote='/old/dynamic/overlay', mode='dynamic', maximum=104)))
    old_release = (root / 'LEGACY_AUTOSCALE.json').read_bytes()
    starts = []
    rpc = c.rpc
    def capture(host, action, spec, **kwargs):
        if action == 'start' and spec.get('role') == 'driver':
            starts.append(copy.deepcopy(spec))
        return rpc(host, action, spec, **kwargs)
    monkeypatch.setattr(c, 'rpc', capture)
    c.chain(root, ['first'])
    assert len(starts) == 2
    assert starts[0]['launch_id'] != starts[1]['launch_id']
    for spec in starts:
        assert spec['workers'] == [13, 13, 13]
        assert not any(k.startswith('OPENPI_AUTOSCALE') or k in ('OPENPI_WORKER_GOVERNOR', 'OPENPI_WORKER_OBSERVER') for k in spec['env'])
        assert legacy_release.rpc_command('/native/node.py', 'start', '/spec.json', spec) == ['python3', '/native/node.py', 'start', '/spec.json']
    assert (root / 'state/CHAIN.DONE').exists()
    assert not (root / 'state/CHAIN.ERROR').exists()
    assert (root / 'LEGACY_AUTOSCALE.json').read_bytes() == old_release


@pytest.mark.parametrize('old_session_alive', [False, True])
def test_real_node_launch_with_dynamic_owner_receipt(tmp_path, monkeypatch, old_session_alive):
    monkeypatch.setattr(node, 'ISLAND', tmp_path)
    out = tmp_path / 'os_cl/runs/fa_pi_f_u_o_l50/arm'
    out.mkdir(parents=True)
    old = dict(role='driver', out=str(out), arm='arm', suite='libero_10',
               servers=['server:23230'], workers=[39], launch_id='failed-dynamic',
               env={'OPENPI_AUTOSCALE_OVERLAY': '/old/overlay'})
    receipt = out / 'driver.owner.json'
    receipt.write_text(json.dumps(dict(spec=old, finished=True, stopped=True)))
    evidence = out / 'autoscale/w00.recycle.json'
    evidence.parent.mkdir()
    evidence.write_text('{"completed_episodes":4,"reason":"episode_boundary"}\n')
    saved = evidence.read_bytes()
    spec = dict(old, launch_id='native-retry', env={'WORKER_HOST': 'timan107'})
    calls = []
    monkeypatch.setattr(node, 'session_exists', lambda _: old_session_alive)
    monkeypatch.setattr(node, 'wait_workers', lambda: calls.append('wait_workers'))
    monkeypatch.setattr(node, 'call', lambda args, **kw: calls.append(args) or '')
    if old_session_alive:
        with pytest.raises(RuntimeError, match='occupied session'):
            node.launch(spec)
        assert json.loads(receipt.read_text())['spec'] == old
        assert calls == []
    else:
        node.launch(spec)
        assert json.loads(receipt.read_text())['spec'] == spec
        job = json.loads((out / 'driver.job.json').read_text())
        assert job['cmd'][:2] == ['bash', str(tmp_path / 'os_cl/run_arm.sh')]
        assert job['env'] == spec['env']
        assert calls[0] == 'wait_workers'
        assert calls[1][0] == 'tmux'  # captured only; no process is started
    assert evidence.read_bytes() == saved
