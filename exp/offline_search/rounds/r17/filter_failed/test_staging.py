"""CPU regressions for the fixed-name launcher inputs and pre-start reset."""
import json
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import patch

import pytest

from . import stage, reset


@pytest.mark.parametrize('node', ['local', 'timan107', 'timan1', 'timan108'])
def test_fixed_benchmark_retains_source_bytes_beside_content_addressed_controls(tmp_path, node):
    assets = tmp_path/'assets'
    assets.mkdir()
    # The real bug: valid equivalent JSON with different whitespace hashes.
    raw = b'{\n  "suite": "libero_10"\n}\n'
    (assets/'benchmark.json').write_bytes(raw)
    (tmp_path/(node+'.json')).write_text('{"worker_benchmark":"control_b.json"}')
    controls = tmp_path/'control_inputs'
    controls.mkdir()
    (controls/'b.json').write_bytes(b'{"suite":"libero_10"}\n')
    (controls/'c.json').write_text('{}')
    files = {}
    def transfer(argv):
        assert argv[:2] == ['rsync', 'push']
        files[argv[3]] = Path(argv[2]).read_bytes()
    launch = SimpleNamespace(remote=lambda *a, **k: None, transfer=transfer)
    directory = '/tmp/test/'+node
    item = dict(stage=directory, benchmark_transfer=dict(path=directory+'/control_b.json', sha256='b'),
                campaign_transfer=dict(path=directory+'/control_c.json', sha256='c'))
    stage.stage_worker_inputs(launch, dict(local_assets=str(assets)), tmp_path, {node: item})
    assert files[node+':'+directory+'/benchmark.json'] == raw
    assert files[node+':'+directory+'/control_b.json'] != raw
    assert files[node+':'+directory+'/campaign.json'] == (tmp_path/(node+'.json')).read_bytes()
    assert files[node+':'+directory+'/launch_campaign.json'] == (tmp_path/(node+'.json')).read_bytes()


def failed_run(root):
    values = {'RECEIPT.json': dict(status='FAIL', gpu_bound=False, all_owned_closed=True),
              'LAUNCH.json': dict(pid=99999999), 'PREPARED.json': dict(status='PASS')}
    for name, value in values.items():
        (root/name).write_text(json.dumps(value))
    (root/'PHASE_TIMINGS.jsonl').write_text(json.dumps(dict(phase='prepared_validation', status='FAIL'))+'\n')
    (root/'.startup_inputs').mkdir()
    (root/'.startup_inputs'/'evidence').write_bytes(b'original snapshot')
    return values


def test_reset_preserves_evidence_and_restages_without_launching(tmp_path):
    failed_run(tmp_path)
    originals = {p.name: p.read_bytes() for p in tmp_path.iterdir() if p.is_file()}
    def restage(root):
        assert not (root/'LAUNCH.json').exists()
        assert not (root/'RECEIPT.json').exists()
        assert not (root/'.startup_inputs').exists()
        assert (root/'PREPARED.json').read_bytes() == originals['PREPARED.json']
    with patch.object(stage, '_stage', side_effect=restage) as prepare:
        reset.reset(tmp_path)
    prepare.assert_called_once_with(tmp_path)
    record = json.loads(next(tmp_path.glob('RESET.failed_*.json')).read_text())
    assert record['status'] == 'PREPARED'
    for name, saved in record['renamed'].items():
        if name == '.startup_inputs':
            assert (tmp_path/saved/'evidence').read_bytes() == b'original snapshot'
        else:
            assert (tmp_path/saved).read_bytes() == originals[name]


def test_closed_model_load_failure_preserves_worker_and_replica_evidence(tmp_path):
    failed_run(tmp_path)
    stage_dir=tmp_path/'worker_stage';stage_dir.mkdir()
    (stage_dir/'workers.process.json').write_text('{"status":"closed"}')
    worker_run=tmp_path/'worker_run';worker_run.mkdir()
    (worker_run/'receipt.json').write_text('{"closed":true}')
    (tmp_path/'servers').mkdir()
    (tmp_path/'servers/failed').write_text('model load failed')
    (tmp_path/'replica_0.log').write_text('identity mismatch')
    (tmp_path/'WORKER_DEPLOYMENT.json').write_text(json.dumps({'local':dict(stage=str(stage_dir),worker_run=str(worker_run))}))
    receipt=dict(status='FAIL',all_owned_closed=True,
                 owned_local=[dict(label='replica_0',pid=99999999,start_ticks='0')],
                 owned_remote=[dict(label='workers',node='local')],
                 cleanup=[dict(label='replica_0',closure_confirmed=True),
                          dict(label='workers',closure_confirmed=True,alive=False,owned_worker_survivors=[])])
    (tmp_path/'RECEIPT.json').write_text(json.dumps(receipt))
    with patch.object(stage,'_stage'):
        reset.reset(tmp_path)
    record=json.loads(next(tmp_path.glob('RESET.failed_*.json')).read_text())
    assert len(record['external_renames'])==2
    assert not (tmp_path/'servers').exists() and not worker_run.exists()
    assert (tmp_path/record['renamed']['servers']/'failed').read_text()=='model load failed'
    assert (tmp_path/record['renamed']['replica_0.log']).read_text()=='identity mismatch'


def test_server_python_override_is_scoped_to_the_campaign():
    from exp.r17_integration.ops.formal_eval import host_python
    assert host_python(dict(model='dp',server_python='/chosen/existing/python'))=='/chosen/existing/python'
    assert host_python(dict(model='dp'))=='/data/mw_finetune/dp/env/bin/python'


@pytest.mark.parametrize('condition', ['owned', 'gpu', 'validation_passed', 'execution_files', 'live_pid'])
def test_reset_refuses_execution_or_incomplete_prestart_proof(tmp_path, condition):
    values = failed_run(tmp_path)
    if condition in ('owned', 'gpu'):
        receipt = values['RECEIPT.json']
        receipt.update(owned_local=[dict(pid=1)] if condition == 'owned' else [], gpu_bound=condition == 'gpu')
        (tmp_path/'RECEIPT.json').write_text(json.dumps(receipt))
    if condition == 'validation_passed':
        (tmp_path/'PHASE_TIMINGS.jsonl').write_text(json.dumps(dict(phase='prepared_validation', status='PASS'))+'\n')
    if condition == 'execution_files':
        (tmp_path/'servers').mkdir()
    if condition == 'live_pid':
        import os
        (tmp_path/'LAUNCH.json').write_text(json.dumps(dict(pid=os.getpid())))
    with pytest.raises(ValueError):
        reset.prestart_failure(tmp_path)
    assert (tmp_path/'LAUNCH.json').exists()
    assert not list(tmp_path.glob('RESET*'))


@pytest.mark.parametrize('local_worker', [False, True])
def test_repeated_prepare_repairs_staging_and_refuses_existing_launch(tmp_path, local_worker):
    (tmp_path/'campaign.json').write_text(json.dumps(dict(run_name='cpu_test',filter_failed_source=str(tmp_path/'source'),
                                                        server_node='local', ports=[],
                                                        runtime='/same/frozen/runtime', runtime_identity='pin',
                                                        local_worker_exception={'owner_authorized': True} if local_worker else None)))
    (tmp_path/'source').mkdir()
    original_receipt = b'{"archive_sha256":"same-frozen-worker-archive"}\n'
    (tmp_path/'source/LOCAL_WORKER_ORCHESTRATION.json').write_bytes(original_receipt)
    (tmp_path/'PREPARED.json').write_text('{}')
    (tmp_path/'WORKER_DEPLOYMENT.json').write_text('{}')
    with patch.object(stage, 'stage_worker_inputs') as workers, patch.object(stage, 'stage_server_inputs') as server, \
            patch('exp.r17_integration.ops.formal_prestage.seal_prepared'), \
            patch('exp.r17_integration.ops.immutable_release.install'), \
            patch('exp.r17_integration.ops.immutable_release.sha', return_value='archive_pin'):
        stage.stage(tmp_path)
    workers.assert_called_once()
    server.assert_called_once()
    assert json.loads((tmp_path/'campaign.json').read_text())['coordinator_runtime'] == '/same/frozen/runtime'
    if local_worker:
        assert (tmp_path/'LOCAL_WORKER_ORCHESTRATION.json').read_bytes() == original_receipt
    (tmp_path/'LAUNCH.json').write_text('{}')
    with pytest.raises(ValueError, match='already launched'):
        stage.stage(tmp_path)


def test_h100_stages_both_campaign_names_with_current_metadata(tmp_path):
    assets = tmp_path/'assets'
    assets.mkdir()
    (assets/'manifest.json').write_text('{}')
    campaign = dict(server_node='h100', run_name='new_run', assets='/remote/assets', local_assets='/original/assets',
                    runtime='/same/frozen/runtime', final_worker_metadata=True)
    files = {}
    def transfer(argv):
        files[argv[3]] = Path(argv[2]).read_bytes()
    launch = SimpleNamespace(remote=lambda *a, **k: None, transfer=transfer,
                             pythonpath=lambda r:r, host_python=lambda c:'python3')
    stage.stage_server_inputs(launch, campaign, tmp_path)
    prefix = 'h100:/tmp/r17_formal/new_run/server/'
    assert files[prefix+'campaign.json'] == (tmp_path/'campaign.json').read_bytes()
    assert files[prefix+'launch_campaign.json'] == files[prefix+'campaign.json']
    assert json.loads(files[prefix+'campaign.json'])['final_worker_metadata'] is True
