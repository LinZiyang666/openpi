"""CPU campaign configuration and continuation of native stage batching."""
import copy
import io
import json
from pathlib import Path
import tarfile

import pytest

from exp.r17_integration.common.batch_runtime import BatchConfig
from exp.r17_integration.ops import cache_control, cache_resume, formal_eval, formal_prestage, queue_retry


@pytest.mark.parametrize('model', ['pi05_mw','pi05_libero','groot_mw','groot_libero'])
def test_flags_recorded_in_campaign_and_worker_streams(model, tmp_path, monkeypatch):
    monkeypatch.setattr(formal_eval, 'FORMAL_ROOT', tmp_path)
    monkeypatch.setattr(formal_eval, 'prepare', lambda campaign, directory: {})
    monkeypatch.setattr(formal_eval, 'close_tunnels', lambda *args: None)
    name = model+'_batch'
    assert formal_eval.main(['--model',model,'--max-batch','8','--batch-wait-ms','7',
        '--replicas','2','--server-node','local','--prepare-only','--run-name',name]) == 0
    campaign = json.loads((tmp_path/name/'campaign.json').read_text())
    assert campaign['stage_batching'] == BatchConfig(8,7).metadata()
    assert campaign['replicas'] == 2
    env = formal_eval.server_env(campaign, str(Path.cwd()))
    assert env['OPENPI_DISABLE_STAGE_STREAMS'] == '0'
    assert all(env[f'BATCHING_STAGE{i}_WORKERS'] == '1' for i in (1,2,3))
    flags = formal_eval.parse_args(['--model',model,'--max-batch','8','--mps','--replicas','2','--prepare-only'])
    assert flags.mps and flags.max_batch == 8


def test_retry_clone_retains_batching_and_cache_resource_history(tmp_path, monkeypatch):
    snapshot = tmp_path/'snapshot'
    snapshot.mkdir()
    for name, value in [('PREPARED.json',dict(files=['EPISODE_PLAN.json'])), ('EPISODE_PLAN.json',dict(tasks=[]))]:
        (snapshot/name).write_text(json.dumps(value))
    for name in ('runtime.tar','workers.tar'):
        (snapshot/name).write_bytes(b'CPU fixture')
    campaign = dict(run_name='old',model='pi05',server_node='local',cache_study='study',cache_control='control',
                    runtime='frozen',runtime_identity='pin',replicas=2,stage_batching=BatchConfig(8,5).metadata())
    monkeypatch.setattr(queue_retry,'validate_inputs',lambda path:(copy.deepcopy(campaign),{}))
    monkeypatch.setattr(cache_control,'check',lambda c:None)
    monkeypatch.setattr(formal_eval,'FORMAL_ROOT',tmp_path)
    monkeypatch.setattr(formal_eval,'prepare_workers',lambda *args:{})
    monkeypatch.setattr(formal_prestage,'seal_prepared',lambda *args:None)
    monkeypatch.setattr(formal_prestage,'validate_prepared',lambda *args:None)
    destination = queue_retry.clone(snapshot,'new',dict(status='PASS'))
    cloned = json.loads((destination/'campaign.json').read_text())
    assert cloned['stage_batching'] == campaign['stage_batching']
    assert cache_resume.segment_resources(cloned,destination)[0]['batching'] == campaign['stage_batching']


def test_cache_resume_preserves_batch_settings(tmp_path, monkeypatch):
    import shutil
    from exp.r17_integration.tests.ops.test_cache_lane import fake_failed
    source = fake_failed(tmp_path)
    campaign = json.loads((source/'campaign.json').read_text())
    campaign['stage_batching'] = BatchConfig(8,7).metadata()
    (source/'campaign.json').write_text(json.dumps(campaign))
    destination = tmp_path/'continued'
    def clone(*args, **kwargs):
        shutil.copytree(source,destination)
        (destination/'WORKER_DEPLOYMENT.json').write_text('{}')
        return destination
    monkeypatch.setattr(queue_retry,'clone',clone)
    monkeypatch.setattr(formal_prestage,'seal_prepared',lambda *args:None)
    assert cache_resume.prepare(source,'continued') == destination
    assert json.loads((destination/'campaign.json').read_text())['stage_batching'] == campaign['stage_batching']


def test_batched_groot_cache_runtime_includes_native_batch_hooks_only(tmp_path):
    def archive(path, files):
        with tarfile.open(path,'w') as output:
            for name, value in files.items():
                member = tarfile.TarInfo(name)
                member.size = len(value)
                output.addfile(member, io.BytesIO(value))
    native = ('src/openpi/cache/groot/batcher.py','src/openpi/cache/groot/staged.py')
    base, fresh, destination = [tmp_path/name for name in ('base.tar','fresh.tar','out.tar')]
    files = {**{name:b'old' for name in native}, 'src/openpi/unrelated.py':b'control'}
    archive(base,files)
    files.update({name:b'new stage batching' for name in native})
    files['src/openpi/unrelated.py'] = b'worktree change'
    from exp.r17_integration.ops.cache_study import stall_runtime_files
    files.update({name:b'Q3' for name in stall_runtime_files()})
    archive(fresh,files)
    result = cache_control.overlay_archive(base,fresh,destination,batch_groot=True)
    with tarfile.open(destination) as output:
        assert all(output.extractfile(name).read()==b'new stage batching' for name in native)
        assert output.extractfile('src/openpi/unrelated.py').read() == b'control'
    assert result['non_r17_exact'] is False
