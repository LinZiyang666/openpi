"""Incomplete CPU staging must retain native pins and refuse unsafe adoption."""
import hashlib
from pathlib import Path

import pytest

from exp.r17_integration.ops import cache_cpu_prepare as recovery, cache_study as study
from exp.r17_integration.ops import formal_eval, formal_prestage, cache_control, driver_release
from exp.r17_integration.ops.build_client_bundle import json_bytes
from exp.r17_integration.search.store import atomic_json


@pytest.fixture
def incomplete(tmp_path,monkeypatch):
    directory=tmp_path/'cpu';directory.mkdir();assets=directory/'assets';assets.mkdir()
    manifest=tmp_path/'STUDY.json';manifest.write_text('{}')
    sealed=dict(source_study_digest='study');sealed['digest']=study.digest(sealed)
    atomic_json(assets/'CACHE_STUDY.json',sealed)
    snapshot=dict(schema='r17.snapshot.v1',partials=[],files={'CACHE_STUDY.json':study.file_sha(assets/'CACHE_STUDY.json')})
    snapshot['digest']=hashlib.sha256(json_bytes(snapshot)).hexdigest()
    atomic_json(assets/'TRANSFER_SHA256.json',snapshot)
    atomic_json(directory/'SERVER_ASSET_TRANSFER.json',dict(status='verified',digest=snapshot['digest'],partials=[]))
    runtime={'files':{'a':'pin'}};runtime_pin=recovery.identity(runtime)
    (directory/'runtime.tar').write_bytes(b'frozen')
    atomic_json(directory/'RUNTIME_RELEASE.json',dict(archive_sha256=study.file_sha(directory/'runtime.tar'),
        identity=runtime_pin,root=str(tmp_path/'native')))
    campaign=dict(prepare_only=True,cache_study=str(manifest),cache_control='control',worker_nodes={'timan107':8},
        local_assets=str(assets),runtime=str(tmp_path/'native'),runtime_identity=runtime_pin,replicas=2,ports=[23440])
    atomic_json(directory/'campaign.json',campaign)
    atomic_json(directory/'ORCHESTRATION.json',dict(driver_runtime=str(tmp_path/'driver'),
        selected=dict(module='driver',mode='legacy')))
    atomic_json(directory/'EPISODE_PLAN.json',dict(tasks=[],expected_episodes=0,model={'checkpoint_identity':'model'},benchmark_protocol='official'))
    monkeypatch.setattr(formal_eval,'FORMAL_ROOT',tmp_path)
    monkeypatch.setattr(study,'validate_manifest',lambda p:dict(digest='study'))
    monkeypatch.setattr(recovery,'archive_manifest',lambda p:runtime)
    monkeypatch.setattr(recovery,'verify',lambda *a:dict(present=True))
    monkeypatch.setattr(cache_control,'check',lambda c:None)
    monkeypatch.setattr(driver_release,'validate',lambda *a:None)
    monkeypatch.setattr(driver_release,'planned_tasks',lambda *a:[])
    attempts=[]
    def workers(c,d):
        attempts.append(c)
        atomic_json(d/'timan107_MEMORY_ADMISSION.json',dict(admitted=True))
        atomic_json(d/'timan107.json',c)
        atomic_json(d/'WORKER_DEPLOYMENT.json',dict(timan107={}))
        return dict(timan107={})
    monkeypatch.setattr(formal_eval,'prepare_workers',workers)
    return directory,manifest,attempts,workers


@pytest.mark.parametrize('marker',['LAUNCH.json','RECEIPT.json'])
def test_dispatched_campaign_cannot_enter_cpu_recovery(incomplete,marker):
    directory,manifest,attempts,_=incomplete
    (directory/marker).write_text('{}')
    with pytest.raises(ValueError,match='undispatched incomplete'):recovery.finish(directory,manifest)
    assert not attempts


def test_snapshot_mutation_refused_before_worker_import(incomplete):
    directory,manifest,attempts,_=incomplete
    (directory/'assets/CACHE_STUDY.json').write_text('{}')
    with pytest.raises((ValueError,KeyError)):recovery.finish(directory,manifest)
    assert not attempts


def test_frozen_plan_mutation_refused_before_worker_preparation(incomplete):
    directory,manifest,attempts,_=incomplete
    plan=study.read(directory/'EPISODE_PLAN.json');plan['tasks']=[{'task_uid':'changed'}]
    atomic_json(directory/'EPISODE_PLAN.json',plan)
    with pytest.raises(ValueError,match='frozen official-A plan'):recovery.finish(directory,manifest)
    assert not attempts


def test_runtime_mutation_refused_before_worker_preparation(incomplete):
    directory,manifest,attempts,_=incomplete
    (directory/'runtime.tar').write_bytes(b'changed')
    with pytest.raises(ValueError,match='native runtime changed'):recovery.finish(directory,manifest)
    assert not attempts


def test_disk_failure_preserves_native_inputs_for_a_cpu_only_retry(incomplete,monkeypatch):
    directory,manifest,attempts,workers=incomplete
    pins={n:study.file_sha(directory/n) for n in ('runtime.tar','EPISODE_PLAN.json','ORCHESTRATION.json','assets/CACHE_STUDY.json')}
    def disk_refused(*a):raise RuntimeError('worker disk admission refused')
    monkeypatch.setattr(formal_eval,'prepare_workers',disk_refused)
    with pytest.raises(RuntimeError,match='disk admission refused'):recovery.finish(directory,manifest)
    assert not (directory/'PREPARED.json').exists() and not attempts
    assert all(study.file_sha(directory/n)==pin for n,pin in pins.items())
    monkeypatch.setattr(formal_eval,'prepare_workers',workers)
    recovery.finish(directory,manifest)
    formal_prestage.validate_prepared(directory)
    assert len(attempts)==1 and all(study.file_sha(directory/n)==pin for n,pin in pins.items())
    proof=study.read(directory/'CPU_WORKER_PREPARE_RECOVERY.json')
    assert proof['status']=='PASS' and proof['cpu_only'] and not proof['gpu_launched']


def test_existing_worker_archive_is_preserved_with_only_recycle_helper_refresh(incomplete):
    from exp.r17_integration.ops.build_client_bundle import build_client_bundle
    directory,manifest,attempts,_=incomplete
    source=directory/'client_source';source.mkdir();(source/'client.py').write_text('pass\n')
    build_client_bundle(source,['client.py'],directory/'workers.tar')
    original=study.file_sha(directory/'workers.tar')
    recovery.finish(directory,manifest)
    refresh=attempts[0]['cache_worker_refresh']
    assert study.file_sha(Path(refresh['archive']))==refresh['archive_sha256']==original
    assert set(refresh['files'])=={'exp/r17_integration/ops/formal_workers.py',
        'exp/r17_integration/ops/worker_gpu.py','exp/r17_integration/ops/worker_disk.py'}
