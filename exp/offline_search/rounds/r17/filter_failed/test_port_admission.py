"""CPU-only regression for the cross-lane port failure, with no listeners."""
import json
import socket
from unittest.mock import patch

import pytest

from exp.r17_integration.ops.formal_ports import local_ports, prepare_admission
from . import reset, stage


def test_port_collision_fails_before_staging_and_reset_preserves_evidence(tmp_path):
    with socket.socket() as owner:
        owner.bind(('0.0.0.0', 0))  # Reserve only; never listen or start a server.
        busy = owner.getsockname()[1]
        with socket.socket() as available:
            available.bind(('0.0.0.0', 0))
            free = available.getsockname()[1]
        campaign = dict(run_name='cpu_port_test', server_node='h100',
                        driver_port=busy, ports=[free, busy],
                        filter_failed_source=str(tmp_path/'source'))
        assert local_ports(campaign) == [busy]
        with pytest.raises(ValueError, match='LOCAL_PORT_BUSY'):
            prepare_admission(campaign)
        # An occupied H100 server-port number is harmless on this host.
        remote = dict(campaign, driver_port=free, ports=[busy, free])
        prepare_admission(remote)
        assert local_ports(dict(campaign, server_node='local')) == sorted([free, busy])
        with pytest.raises(ValueError, match='LOCAL_PORT_BUSY'):
            prepare_admission(dict(remote, server_node='local'))
        (tmp_path/'campaign.json').write_text(json.dumps(campaign))
        with patch('exp.r17_integration.ops.immutable_release.install',
                   side_effect=AssertionError('install before admission')), \
             patch('exp.r17_integration.ops.formal_eval.remote',
                   side_effect=AssertionError('remote before admission')), \
             patch('exp.r17_integration.ops.formal_eval.transfer',
                   side_effect=AssertionError('transfer before admission')):
            with pytest.raises(ValueError, match='LOCAL_PORT_BUSY'):
                stage.stage(tmp_path)
        assert not (tmp_path/'PREPARED.json').exists()

    receipt = dict(status='PARTIAL', all_owned_closed=True, owned_local=[],
                   owned_remote=[], cleanup=[],
                   error="TimeoutError('ports did not release within 120s: [23159]')")
    for name, value in {'RECEIPT.json': receipt, 'LAUNCH.json': dict(pid=99999999),
                        'PORT_RELEASE.json': dict(status='FAIL'),
                        'PREPARED.json': dict(status='PASS')}.items():
        (tmp_path/name).write_text(json.dumps(value))
    (tmp_path/'PHASE_TIMINGS.jsonl').write_text(json.dumps(
        dict(phase='prepared_validation', status='PASS'))+'\n')
    originals = {p.name: p.read_bytes() for p in tmp_path.iterdir() if p.is_file()}
    # PARTIAL without port evidence remains ineligible for reset.
    with patch.object(reset, 'read', side_effect=lambda p: dict(status='PASS')
                      if p.name == 'PORT_RELEASE.json' else json.loads(p.read_text())):
        with pytest.raises(ValueError, match='closed failed startup'):
            reset.prestart_failure(tmp_path)
    with patch.object(stage, '_stage', side_effect=AssertionError('staging forbidden')), \
         patch('exp.r17_integration.ops.formal_eval.remote',
               side_effect=AssertionError('remote mutation forbidden')):
        reset.reset(tmp_path, local_only=True)
    record = json.loads(next(tmp_path.glob('RESET.failed_*.json')).read_text())
    assert record['status'] == 'UNSTAGED' and record['external_renames'] == []
    assert record['driver_started'] is False
    for name, saved in record['renamed'].items():
        assert (tmp_path/saved).read_bytes() == originals[name]
    for name in ('RECEIPT.json', 'LAUNCH.json', 'PREPARED.json'):
        assert not (tmp_path/name).exists()
