"""Cleanup must be closed, SHA-verified and confined to explicit stage files."""
import json
from pathlib import Path

import pytest

from exp.r17_integration.ops import library_cleanup as cleanup
from exp.r17_integration.ops.build_client_bundle import sha256


@pytest.fixture
def plan(tmp_path,monkeypatch):
    monkeypatch.setattr(cleanup,'TRANSFER_ROOT',tmp_path/'tmp/r17_formal')
    monkeypatch.setattr(cleanup,'Q_ROOT',tmp_path/'tmp/r17_Q_qualification')
    server=cleanup.TRANSFER_ROOT/'CPU/server';server.mkdir(parents=True)
    (server/'campaign.json').write_text(json.dumps(dict(library=True,server_node='h100',run_name='CPU')))
    (server/'replica_0.process.json').write_text(json.dumps(dict(status='CLOSED')))
    archive=server/'runtime.tar';archive.write_bytes(b'verified-runtime')
    qroot=cleanup.Q_ROOT/'CPU_shards';qroot.mkdir(parents=True)
    staged=qroot/('workers_'+sha256(archive)+'_'+'a'*32+'.tar');staged.write_bytes(archive.read_bytes())
    marker=staged.with_suffix('.stage.json');marker.write_text(json.dumps(dict(run_key='CPU_shards',kind='workers',
        archive=str(staged),archive_sha256=sha256(staged),stage_sha_verified=True,stage_removed=False)))
    request=dict(campaign='CPU',verified_archive_sha256s=[sha256(archive)],files=[
        dict(path=str(p),sha256=sha256(p)) for p in (archive,staged,marker)])
    return request,server


def test_closed_cleanup_removes_only_verified_list_retains_logs_and_adjacent_dp(plan):
    request,server=plan
    log=server/'replica_0.log';log.write_text('retain evidence')
    other=cleanup.TRANSFER_ROOT/'dp_libero_libero_spatial/server/shards/other.tar'
    other.parent.mkdir(parents=True);other.write_bytes(b'live-DP-stage')
    dry=cleanup.cleanup_verified_stages(request)
    assert dry['applied'] is False and dry['removed_bytes']==0
    assert all(Path(e['path']).exists() for e in request['files'])
    result=cleanup.cleanup_verified_stages(request,apply=True)
    assert result['removed_bytes']==sum(e['bytes'] for e in result['files'])
    assert all(not Path(e['path']).exists() for e in request['files'])
    assert log.read_text()=='retain evidence' and other.read_bytes()==b'live-DP-stage'
    assert (server/'campaign.json').exists() and (server/'replica_0.process.json').exists()


@pytest.mark.parametrize('mutation',('live','unclosed','sha','foreign','symlink','marker','unverified','missing_receipts'))
def test_any_unverified_or_foreign_item_rejects_entire_plan_before_unlink(plan,mutation,monkeypatch):
    request,server=plan
    if mutation=='live':
        (server/'replica_0.process.json').write_text(json.dumps(dict(status='CLOSED',child={'pid':123})))
        monkeypatch.setattr(cleanup,'alive',lambda entry:True)
    elif mutation=='unclosed':(server/'replica_0.process.json').write_text('{"status":"RUNNING"}')
    elif mutation=='sha':request['files'][-1]['sha256']='0'*64
    elif mutation=='foreign':
        foreign=server.parent.parent/'DP/server/runtime.tar';foreign.parent.mkdir(parents=True);foreign.write_bytes(b'DP')
        request['files'].append(dict(path=str(foreign),sha256=sha256(foreign)))
    elif mutation=='symlink':
        link=server/'shards'/('0'*64+'.tar');link.parent.mkdir();link.symlink_to(Path(request['files'][0]['path']))
        request['files'].append(dict(path=str(link),sha256=sha256(link)))
    elif mutation=='marker':
        path=Path(request['files'][-1]['path']);row=json.loads(path.read_text());row['run_key']='DP';path.write_text(json.dumps(row))
        request['files'][-1]['sha256']=sha256(path)
    elif mutation=='unverified':request['verified_archive_sha256s']=['0'*64]
    else:(server/'replica_0.process.json').unlink()
    with pytest.raises(ValueError):cleanup.cleanup_verified_stages(request,apply=True)
    assert all(Path(e['path']).exists() for e in request['files'])
